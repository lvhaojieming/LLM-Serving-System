import copy
import json
from pathlib import Path
import pytest
from heterogeneous_compute.serving import admission_config, commands, expand_config, fingerprint, gateway_config, generation_workers, live, local, native_spec, router_configs, router_workers, serve_signature


def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/serving.json").read_text())


def test_admission_keeps_public_address_and_loopback_application_distinct():
    cfg = expand_config(config())
    assert list(cfg['services'])[-1] == 'admission'
    assert cfg['gateway']['port'] == 18600
    assert gateway_config(cfg)['port'] == 18610
    assert gateway_config(cfg)['host'] == '127.0.0.1'
    altered = copy.deepcopy(cfg)
    altered['admission']['max_queue'] = 0
    assert fingerprint(cfg, 'admission') != fingerprint(altered, 'admission')
    assert fingerprint(cfg, 'gateway') == fingerprint(altered, 'gateway')
    altered['gateway']['internal_port'] = 18611
    assert fingerprint(cfg, 'gateway') != fingerprint(altered, 'gateway')
    with pytest.raises(ValueError):
        admission_config({**cfg, 'admission': {**cfg['admission'], 'max_queue': -1}})


def test_admission_preserves_cleanup_headroom_and_configured_time_budgets():
    cfg=expand_config(config())
    rendered=admission_config(cfg)
    assert 'maxconn 512 maxqueue 128' in rendered
    assert 'timeout queue 10s' in rendered
    assert 'timeout client 180s' in rendered and 'timeout server 180s' in rendered
    assert 'timeout http-keep-alive 75s' in rendered
    assert 'hard-stop-after 190s' in rendered
    assert cfg['gateway']['max_inflight']==640
    cfg['admission']['max_inflight']=641
    with pytest.raises(ValueError,match='internal Gateway budget'):admission_config(cfg)
    cfg['admission'].pop('max_inflight')
    assert 'maxconn 640 maxqueue 128' in admission_config(cfg)


@pytest.mark.parametrize('queue_size', [0, 2])
def test_official_haproxy_queue_stream_cancel_overflow_and_timeout(tmp_path, queue_size):
    """Exercise the real proxy, not a reimplementation of its scheduler.

    HETEROSERVE_HAPROXY selects the retained Linux binary. Dependencies stay in
    that binary's configured LD_LIBRARY_PATH. Temporary HTTP servers use no GPU.
    """
    import concurrent.futures
    import http.server
    import os
    import signal
    import socket
    import subprocess
    import threading
    import time
    import httpx
    from heterogeneous_compute.benchmark_telemetry import parse_metrics

    binary = os.environ.get('HETEROSERVE_HAPROXY')
    if not binary:
        pytest.skip('Select the retained HAProxy binary for Linux integration')
    gate = threading.Event()
    counts = {'active': 0, 'peak': 0}
    lock = threading.Lock()

    class Backend(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with lock:
                counts['active'] += 1
                counts['peak'] = max(counts['peak'], counts['active'])
            try:
                if payload.get('stream'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    self.wfile.write(b'data: {"choices":[]}\n\n'); self.wfile.flush()
                    while not gate.wait(.02):
                        self.wfile.write(b': heartbeat\n\n'); self.wfile.flush()
                    self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
                else:
                    assert gate.wait(5)
                    self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with lock: counts['active'] -= 1

    def port():
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); return s.getsockname()[1]

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Backend)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cfg = config()
    cfg['gateway'].update(host='127.0.0.1', port=port(), internal_port=server.server_port, max_inflight=1)
    cfg['admission'].update(max_inflight=1, max_queue=queue_size, queue_timeout_seconds=1, metrics_port=port())
    path = tmp_path/'haproxy.cfg'; path.write_text(admission_config(cfg))
    subprocess.run([binary, '-c', '-f', str(path)], check=True)
    log = (tmp_path/'proxy.log').open('w')
    process = subprocess.Popen([binary, '-db', '-f', str(path)], stdout=log, stderr=log)
    endpoint = f"http://127.0.0.1:{cfg['gateway']['port']}"
    metrics = f"http://127.0.0.1:{cfg['admission']['metrics_port']}/metrics"

    def queued():
        rows = parse_metrics(httpx.get(metrics, trust_env=False).text)
        return sum(r['value'] for r in rows if r['name'] == 'haproxy_backend_current_queue' and r['labels'].get('proxy') == 'gateway')

    def wait_for(predicate):
        deadline = time.monotonic() + .7
        while time.monotonic() < deadline:
            try:
                if predicate(): return
            except httpx.TransportError: pass
            time.sleep(.01)
        raise AssertionError('Proxy did not reach expected state')

    def post():
        return httpx.post(endpoint+'/v1/chat/completions', json={}, timeout=4, trust_env=False)

    try:
        wait_for(lambda: httpx.get(metrics, trust_env=False).status_code == 200)
        with httpx.Client(timeout=5, trust_env=False) as client, concurrent.futures.ThreadPoolExecutor(16) as pool:
            if not queue_size:
                with client.stream('POST', endpoint+'/v1/chat/completions', json={'stream': True}) as active:
                    assert post().status_code == 429 and queued() == 0
                    process.send_signal(signal.SIGUSR1)
                    gate.set()
                    assert b'[DONE]' in active.read()
                process.wait(timeout=5)
                return
            with client.stream('POST', endpoint+'/v1/chat/completions', json={'stream': True}) as active:
                assert active.status_code == 200
                pending = [pool.submit(post) for _ in range(2)]
                wait_for(lambda: queued() == 2)
                # SSE headers have already arrived. Its generation still owns
                # the only backend slot; all excess requests must be rejected.
                rejected = list(pool.map(lambda _: post(), range(32)))
                assert all(r.status_code == 429 and r.headers['retry-after'] == '1' for r in rejected)
                assert client.get(endpoint+'/ready').status_code == 200
                assert client.post(endpoint+'/%64rain').status_code == 403
                assert counts['active'] == 1
                gate.set()
                assert b'[DONE]' in active.read()
                responses = [future.result() for future in pending]
                assert all(r.status_code == 200 and float(r.headers['x-heteroserve-queue-ms']) > 0 for r in responses)
            wait_for(lambda: counts['active'] == 0 and queued() == 0)
            gate.clear()
            with client.stream('POST', endpoint+'/v1/chat/completions', json={'stream': True}):
                start = time.monotonic()
                expired = post()
                assert expired.status_code == 503 and .8 <= time.monotonic()-start < 3
                assert queued() == 0
                with socket.create_connection(('127.0.0.1', cfg['gateway']['port'])) as cancelled:
                    cancelled.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}')
                    wait_for(lambda: queued() == 1)
                wait_for(lambda: queued() == 0)
            wait_for(lambda: counts['active'] == 0)
            assert counts['peak'] == 1
            with client.stream('POST', endpoint+'/v1/chat/completions', json={'stream': True}) as active:
                pending = pool.submit(post)
                wait_for(lambda: queued() == 1)
                process.send_signal(signal.SIGUSR1)
                gate.set()
                assert b'[DONE]' in active.read()
                assert pending.result().status_code == 200
            process.wait(timeout=5)
    finally:
        gate.set(); process.terminate(); process.wait(timeout=5)
        log.close(); server.shutdown(); server.server_close()


def test_gateway_aliases_keep_embedding_separate_from_generation_pool():
    cfg = gateway_config(expand_config(config()))
    assert cfg["learned_routing"] == {"enabled": False}
    assert cfg["pools"]["embedding"]["task"] == "embed"
    assert "awq" not in cfg["pools"]
    assert cfg["random_routes"] == {"awq": ["ascend-awq", "musa-awq", "iluvatar-awq"]}
    assert cfg["route_embeddings"] == {"awq": "embedding"}
    assert [cfg["pools"][name]["base_url"] for name in cfg["random_routes"]["awq"]] == [
        "http://127.0.0.1:18601/v1", "http://127.0.0.1:18603/v1", "http://127.0.0.1:18605/v1"]
    assert len(cfg["worker_registry"]["urls"]) == 3


def test_unrelated_service_change_does_not_invalidate_gateway_identity():
    cfg = expand_config(config())
    changed = copy.deepcopy(cfg)
    changed["services"]["ascend-awq"]["cudagraph_metrics"] = False
    assert fingerprint(cfg, "gateway") == fingerprint(changed, "gateway")
    changed["workers"]["ascend-awq"]["base_url"] = "http://192.168.162.24:18428/v1"
    assert fingerprint(cfg, "gateway") != fingerprint(changed, "gateway")
    assert fingerprint(cfg, "router-ascend") != fingerprint(changed, "router-ascend")
    assert fingerprint(cfg, "router-musa") == fingerprint(changed, "router-musa")


def test_pool_generation_does_not_mutate_source_configuration():
    cfg = expand_config(config())
    original = copy.deepcopy(cfg)
    gateway_config(cfg)
    assert cfg == original


def test_official_router_pools_are_disjoint_and_keep_all_current_devices():
    cfg = expand_config(config())
    pools = router_configs(cfg)
    assert {name: len(router_workers(cfg, name)) for name in pools} == {
        'router-ascend': 16, 'router-musa': 8, 'router-iluvatar': 8}
    assigned = []
    for name, spec in pools.items():
        argv, _ = commands(cfg, name)
        assert argv[0] and '--worker-urls' in argv[2]
        assert "'--port', '" + str(spec['port']) + "'" in argv[2]
        assigned.extend(router_workers(cfg, name))
    assert len(assigned) == len(set(assigned)) == 32
    assert cfg == expand_config(cfg)


def test_router_pools_reject_overlap_and_missing_nodes():
    cfg = config()
    cfg['router']['pools']['musa']['nodes'].append('ascend910c-01')
    with pytest.raises(ValueError, match='overlapping'):
        expand_config(cfg)
    cfg = config()
    cfg['router']['pools']['ascend']['nodes'].remove('ascend910c-01')
    with pytest.raises(ValueError, match='cover'):
        expand_config(cfg)


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
    # Graphs are the default; MUSA retains a documented vendor exception.
    assert ('--enforce-eager' in argv) == (vendor == 'musa')
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


def test_ascend_start_reuses_a_port_after_an_owned_connection_closes(tmp_path, monkeypatch):
    import socket
    from types import SimpleNamespace
    from heterogeneous_compute.quantization import deploy
    listener=socket.socket();listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    listener.bind(('127.0.0.1',0));listener.listen();port=listener.getsockname()[1]
    client=socket.create_connection(('127.0.0.1',port));accepted,_=listener.accept()
    accepted.shutdown(socket.SHUT_WR);accepted.close();client.recv(1);client.close();listener.close()
    cfg=json.loads((Path(__file__).resolve().parents[1]/'quantization/deployment.json').read_text())
    cfg.update(host='127.0.0.1',result_dir=str(tmp_path))
    cfg['profiles']['awq']['port']=port
    monkeypatch.setattr(deploy.subprocess,'Popen',lambda *args,**kwargs:SimpleNamespace(pid=123456))
    deploy.start_server(cfg,'awq')
    assert json.loads((tmp_path/'awq/server.json').read_text())['pid']==123456


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
    assert cfg['services']['ascend-24-1']['overrides']['max_model_len'] == 4096
    assert cfg['services']['ascend-awq']['overrides']['max_model_len'] == 4096
    assert cfg['services']['ascend-23-0']['overrides']['max_model_len'] == 4096


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
        argv, env = commands(cfg, name)
        assert '--disable-custom-all-reduce' in argv
        assert '--enforce-eager' in argv and argv[argv.index('--tensor-parallel-size') + 1] == '2'
        assert env['MCCL_SHM_DISABLE'] == '0' and env['MCCL_P2P_DISABLE'] == '1'


def test_iluvatar_nodes_use_graphs_and_keep_embedding_device_isolation():
    cfg = expand_config(config())
    for name in ['iluvatar-58-0', 'iluvatar-58-1', 'iluvatar-58-2', 'iluvatar-58-3']:
        argv, env = commands(cfg, name)
        assert '--enforce-eager' not in argv
        assert '--disable-custom-all-reduce' in argv and '--cudagraph-metrics' in argv
        assert env['NCCL_SOCKET_IFNAME'] == 'enp147s0f0'
        assert not {1, 2, 12, 13, 15}.intersection(map(int, env['CUDA_VISIBLE_DEVICES'].split(',')))
    for name in ['iluvatar-awq', 'iluvatar-60-0', 'iluvatar-60-2', 'iluvatar-60-3']:
        argv, _ = commands(cfg, name)
        assert '--enforce-eager' not in argv and '--cudagraph-metrics' in argv
    embedding = cfg['services']['embedding']
    assert embedding['argv'][embedding['argv'].index('--runner') + 1] == 'pooling'
    assert '--enforce-eager' not in embedding['argv']
    assert embedding['env']['CUDA_VISIBLE_DEVICES'] == '12,13'
    assert 'HETEROSERVE_ILUVATAR_POOLING_ONLY' not in embedding['env']


def test_service_mode_override_takes_precedence_over_vendor_exception():
    cfg = expand_config(config())
    cfg['services']['musa-96-2']['overrides']['enforce_eager'] = False
    argv, _ = commands(cfg, 'musa-96-2')
    assert '--enforce-eager' not in argv
    other, _ = commands(cfg, 'musa-96-3')
    assert '--enforce-eager' in other
