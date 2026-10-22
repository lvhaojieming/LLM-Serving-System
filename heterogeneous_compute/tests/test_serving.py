import copy
import json
from pathlib import Path
import pytest
from heterogeneous_compute.serving import commands, expand_config, fingerprint, gateway_config, generation_workers, live, local, native_spec, serve_signature


def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/serving.json").read_text())


def test_gateway_aliases_keep_embedding_separate_from_generation_pool():
    cfg = gateway_config(config())
    assert cfg["learned_routing"] == {"enabled": False}
    assert cfg["pools"]["embedding"]["task"] == "embed"
    assert cfg["pools"]["awq"]["base_url"] == "http://127.0.0.1:18601/v1"


def test_unrelated_service_change_does_not_invalidate_gateway_identity():
    cfg = config()
    changed = copy.deepcopy(cfg)
    changed["services"]["ascend-awq"]["cudagraph_metrics"] = False
    assert fingerprint(cfg, "gateway") == fingerprint(changed, "gateway")
    changed["backends"]["ascend-awq"]["base_url"] = "http://192.168.162.24:18428/v1"
    assert fingerprint(cfg, "gateway") != fingerprint(changed, "gateway")
    assert fingerprint(cfg, "router") != fingerprint(changed, "router")


def test_stop_does_not_signal_a_reused_or_foreign_pid(tmp_path, monkeypatch):
    import os
    import heterogeneous_compute.serving as serving
    cfg = config()
    cfg["result_dir"] = str(tmp_path)
    marker = tmp_path / "embedding/process.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps({"pid": os.getpid(), "stamp": ["foreign-boot", "0"]}))
    monkeypatch.setattr(os, "killpg", lambda *args: (_ for _ in ()).throw(AssertionError("foreign PID signalled")))
    assert not live(marker)
    local(cfg, "stop", "embedding")


def test_native_process_matching_accepts_shebang_and_equivalent_numeric_arguments():
    a = ['/usr/bin/python', '/usr/local/bin/vllm', 'serve', '/models/Qwen3-14B-AWQ', '--dtype', 'half', '--port', '18497', '--gpu-memory-utilization', '0.60']
    b = ['vllm', 'serve', '/models/Qwen3-14B-AWQ', '--port', '18497', '--gpu-memory-utilization', '0.6', '--dtype', 'float16']
    assert serve_signature(a) == serve_signature(b)
    b[b.index('18497')] = '18498'
    assert serve_signature(a) != serve_signature(b)


def test_native_process_matching_rejects_other_models_and_non_vllm_commands():
    assert serve_signature(['python3', '-m', 'another_server', 'serve', '/models/Qwen3-14B-AWQ']) is None
    a = ['vllm', 'serve', '/models/Qwen3-14B-AWQ', '--quantization', 'awq']
    b = ['vllm', 'serve', '/models/Qwen3-14B', '--quantization', 'awq']
    assert serve_signature(a) != serve_signature(b)


@pytest.mark.parametrize('name,vendor', [('musa-awq', 'musa'), ('iluvatar-awq', 'iluvatar')])
def test_native_lifecycle_uses_retained_vendor_configuration(name, vendor):
    argv, env = commands(config(), name)
    assert argv[argv.index('--quantization') + 1] == 'awq'
    assert argv[argv.index('--tensor-parallel-size') + 1] == '2'
    # These existing native profiles explicitly retain eager; no implicit mode switch.
    assert '--enforce-eager' in argv
    if vendor == 'musa':
        assert env['MUSA_VISIBLE_DEVICES'] == '6,7' and '--block-size' in argv
    else:
        assert env['CUDA_VISIBLE_DEVICES'] == '2,3' and env['KERNEL_BACKEND'] == 'awq_gemm:cuinfer'


def test_start_refuses_live_retired_experiments(monkeypatch):
    import heterogeneous_compute.serving as serving
    monkeypatch.setattr(serving, 'legacy_processes', lambda: [{'pid': 123, 'argv': ['old-pp']}])
    with pytest.raises(RuntimeError, match='123'):
        local(config(), 'start', 'musa-awq')


def test_expansion_respects_six_node_awq_budgets_and_embedding_reservations():
    cfg = expand_config(config())
    assert len(generation_workers(cfg)) == 32
    used = {}
    for worker in cfg['workers'].values():
        entries = used.setdefault(worker['node'], [])
        assert not set(entries).intersection(worker['devices'])
        entries.extend(worker['devices'])
    assert cfg['services']['embedding']['node'] == 'iluvatar150-01'
    assert cfg['services']['embedding']['runtime_profile']['tp'] == 2
    embedding_devices = cfg['services']['embedding']['runtime_profile']['logical_devices']
    assert embedding_devices == [12, 13]
    assert not set(embedding_devices).intersection(used['iluvatar150-01'])
    assert not {1, 2, 15}.intersection(used['iluvatar150-01'])
    assert all(len(w['devices']) == 2 for w in cfg['workers'].values())
    assert {n:len(d) for n,d in used.items()} == {'ascend910c-01':16,'ascend910c-02':16,'s5000-01':8,'s5000-02':8,'iluvatar150-02':8,'iluvatar150-01':8}
    assert len(gateway_config(cfg)['worker_registry']['expected_urls']) == 32


def test_expansion_rejects_overlapping_and_missing_device_partitions():
    cfg = config()
    cfg['replica_groups'][0]['device_sets'][1] = [1,2]
    with pytest.raises(ValueError, match='Overlapping'):
        expand_config(cfg)
    cfg = config()
    cfg['replica_groups'][0]['device_sets'].pop()
    with pytest.raises(ValueError, match='incomplete'):
        expand_config(cfg)


def test_replica_overrides_keep_vendor_runtime_and_select_new_node_devices():
    cfg = expand_config(config())
    spec, target = native_spec(cfg['services']['musa-96-2'])
    assert spec['tp'] == 2 and target['host'] == '192.168.162.96' and target['port'] == 18762
    assert target['env']['MUSA_VISIBLE_DEVICES'] == '4,5'
    assert target['env']['GLOO_SOCKET_IFNAME'] == 'ens21f0'
    assert cfg['services']['ascend-24-1']['overrides']['tp'] == 2
    assert cfg['services']['ascend-24-1']['overrides']['devices'] == [2,3]


def test_worker_acceptance_retains_failures_and_tests_reachable_workers(monkeypatch):
    import urllib.error
    import heterogeneous_compute.serving as serving
    cfg = {'workers': {
        'working': {'base_url': 'http://working/v1', 'model': 'AWQ', 'devices': [0, 1]},
        'offline': {'base_url': 'http://offline/v1', 'model': 'AWQ', 'devices': [2, 3]},
        'wrong_answer': {'base_url': 'http://wrong/v1', 'model': 'AWQ', 'devices': [4, 5]},
    }}
    def request(url, body, timeout):
        if 'offline' in url:
            raise urllib.error.URLError('connection refused')
        return {'choices': [{'message': {'content': '5' if 'working' in url else '4'}}]}
    monkeypatch.setattr(serving, 'http', request)
    results = {r['worker']: r for r in serving.verify_workers(cfg)}
    assert results['working']['passed'] is True
    assert results['offline']['passed'] is False and 'refused' in results['offline']['error']
    assert results['wrong_answer']['passed'] is False and results['wrong_answer']['answer'] == '4'


def test_stop_continues_after_an_ascend_startup_has_already_exited(tmp_path, monkeypatch):
    import heterogeneous_compute.serving as serving
    import heterogeneous_compute.quantization.deploy as deploy
    cfg = config()
    cfg['result_dir'] = str(tmp_path)
    marker = tmp_path / 'ascend-awq/awq/server.json'
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({'pid': 12345}))
    monkeypatch.setattr(serving, 'stamp', lambda pid: None)
    monkeypatch.setattr(deploy, 'stop_server', lambda *args: (_ for _ in ()).throw(AssertionError('exited PID signalled')))
    serving.local(cfg, 'stop', 'ascend-awq')


def test_all_musa_replicas_use_standard_mccl_and_disable_custom_allreduce():
    cfg = expand_config(config())
    names = [name for name, service in cfg['services'].items() if service.get('vendor') == 'musa']
    assert len(names) == 8
    for name in names:
        argv, _ = commands(cfg, name)
        assert '--disable-custom-all-reduce' in argv
        assert '--enforce-eager' in argv and argv[argv.index('--tensor-parallel-size') + 1] == '2'


def test_new_iluvatar_node_defaults_to_graph_without_changing_retained_eager_workers():
    cfg = expand_config(config())
    for name in ['iluvatar-58-0', 'iluvatar-58-1', 'iluvatar-58-2', 'iluvatar-58-3']:
        argv, env = commands(cfg, name)
        assert '--enforce-eager' not in argv
        assert '--disable-custom-all-reduce' in argv and '--cudagraph-metrics' in argv
        assert env['NCCL_SOCKET_IFNAME'] == 'enp147s0f0'
        assert not {1, 2, 12, 13, 15}.intersection(map(int, env['CUDA_VISIBLE_DEVICES'].split(',')))
    retained, _ = commands(cfg, 'iluvatar-awq')
    assert '--enforce-eager' in retained
    embedding = cfg['services']['embedding']
    assert embedding['argv'][embedding['argv'].index('--runner') + 1] == 'pooling'
    assert '--enforce-eager' not in embedding['argv']
    assert embedding['env']['CUDA_VISIBLE_DEVICES'] == '12,13'
    assert 'HETEROSERVE_ILUVATAR_POOLING_ONLY' not in embedding['env']
