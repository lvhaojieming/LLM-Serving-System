import asyncio
import hashlib
import json
import threading
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from heteroserve.routing.client import RemoteRouterClient
from heteroserve.routing.service import create_app
from heteroserve.routing.runtime import RoutingDecision, RouterSettings
from heteroserve.gateway import create_app as gateway_app
import manage_l1
import manage_gateway
from manage_pool import configuration

TEMPLATE = "canonical-template"
TEMPLATE_SHA = hashlib.sha256(TEMPLATE.encode()).hexdigest()


def service_config():
    cfg = manage_gateway.config()
    cfg.update(instance_id="l1-01", max_inflight=1, chat_template_sha256=TEMPLATE_SHA)
    return cfg


class Runtime:
    chat_template = TEMPLATE
    def route(self, messages, generation, kwargs):
        return RoutingDecision("awq", {"awq": .3, "gptq": .7}, 28, 7.)


def identity(name="l1-01"):
    return {"pod_uid": "uid-"+name, "node": "node-"+name, "instance_id": name,
            "checkpoint_sha256": manage_gateway.config()["router"]["checkpoint_sha256"],
            "chat_template_sha256": TEMPLATE_SHA, "encoder_device":"npu:0", "head_device":"npu:0"}


def payload():
    return {"messages":[{"role":"user","content":"hello"}],"max_new_tokens":32,"template_kwargs":{"enable_thinking":False}}


def test_probability_service_preserves_checkpoint_selection_not_argmax():
    app=create_app(service_config(),Runtime())
    with TestClient(app) as client:
        result=client.post('/route',json=payload()).json()
        assert result['decision']['expert']=='awq'
        assert result['decision']['probabilities']['gptq']>.5
        assert result['chat_template']==TEMPLATE
        assert client.get('/ready').status_code==200


def test_service_cancelled_caller_does_not_release_running_npu_lease():
    async def scenario():
        started,finish=threading.Event(),threading.Event()
        class Blocking(Runtime):
            def route(self,*args):
                started.set();finish.wait(5);return super().route(*args)
        app=create_app(service_config(),Blocking())
        async with app.router.lifespan_context(app),httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://service') as client:
            first=asyncio.create_task(client.post('/route',json=payload()))
            for _ in range(100):
                if started.is_set():break
                await asyncio.sleep(.01)
            assert started.is_set()
            first.cancel()
            with pytest.raises(asyncio.CancelledError):await first
            assert len(app.state.l1['tasks'])==1
            assert (await client.post('/route',json=payload())).status_code==429
            finish.set()
            for _ in range(100):
                if not app.state.l1['tasks']:break
                await asyncio.sleep(.01)
            assert not app.state.l1['tasks']
    asyncio.run(scenario())


def test_remote_client_distributes_to_both_replicas_and_preserves_decisions():
    async def scenario():
        seen=[]
        def backend(request):
            seen.append(request.url.host)
            name='l1-01' if request.url.host.endswith('1') else 'l1-02'
            return httpx.Response(200,json={'identity':identity(name),'chat_template':TEMPLATE,
                'decision':{'expert':'awq','probabilities':{'awq':.3,'gptq':.7},'input_tokens':28,'elapsed_ms':7}})
        async def resolve():return ['http://worker1:8000','http://worker2:8000']
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
            runtime=RemoteRouterClient(RouterSettings(**manage_gateway.config()['router']),{'base_url':'http://service:8000','chat_template_sha256':TEMPLATE_SHA},client,resolve)
            for _ in range(4):assert (await runtime.route(payload()['messages'],32,payload()['template_kwargs'])).expert=='awq'
        assert seen==['worker1','worker2','worker1','worker2']
    asyncio.run(scenario())


def test_remote_client_skips_overloaded_replica_and_rejects_wrong_version():
    async def scenario():
        async def resolve():return ['http://worker1:8000','http://worker2:8000']
        def backend(request):
            if request.url.host=='worker1':return httpx.Response(429)
            wrong=identity('l1-02');wrong['checkpoint_sha256']='f'*64
            return httpx.Response(200,json={'identity':wrong})
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
            runtime=RemoteRouterClient(RouterSettings(**manage_gateway.config()['router']),{'base_url':'http://service:8000','chat_template_sha256':TEMPLATE_SHA},client,resolve)
            with pytest.raises(HTTPException,match='contract mismatch'):
                await runtime.route(payload()['messages'],32,payload()['template_kwargs'])
    asyncio.run(scenario())


def test_l1_deployments_are_independent_one_card_and_headless_ready_only():
    cluster,pool,npu=configuration()
    cfg=manage_l1.config()
    objects=manage_l1.objects(cluster,pool,npu,cfg,manage_gateway.config())
    headless=objects[0]
    assert headless['spec']['clusterIP']=='None' and not headless['spec'].get('publishNotReadyAddresses')
    deployments=[v for v in objects if v['kind']=='Deployment']
    assert len(deployments)==2
    for deployment in deployments:
        pod=deployment['spec']['template']['spec']
        assert pod['containers'][0]['resources']['limits']=={npu['resource']:'1'}
        assert deployment['spec']['strategy']['rollingUpdate']=={'maxSurge':1,'maxUnavailable':0}
        assert any('heteroserve.routing.service' in value for value in pod['containers'][0]['args'])


def test_gateway_concurrent_remote_routes_keep_per_request_instance_identity():
    async def scenario():
        async def resolve():return ['http://worker1:8000','http://worker2:8000']
        def route_backend(request):
            name='l1-01' if request.url.host=='worker1' else 'l1-02'
            return httpx.Response(200,json={'identity':identity(name),'chat_template':TEMPLATE,
                'decision':{'expert':'awq','probabilities':{'awq':.3,'gptq':.7},'input_tokens':28,'elapsed_ms':7}})
        async def inference_backend(request):
            await asyncio.sleep(.02)
            return httpx.Response(200,json={'model':'backend','choices':[]})
        routing_client=httpx.AsyncClient(transport=httpx.MockTransport(route_backend))
        runtime=RemoteRouterClient(RouterSettings(**manage_gateway.config()['router']),{'base_url':'http://service:8000','chat_template_sha256':TEMPLATE_SHA},routing_client,resolve)
        app=gateway_app(manage_gateway.config(),httpx.MockTransport(inference_backend),runtime)
        async with app.router.lifespan_context(app),httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://gateway') as client:
            body={'model':'auto','messages':[{'role':'user','content':'hello'}]}
            responses=await asyncio.gather(client.post('/v1/chat/completions',json=body),client.post('/v1/chat/completions',json=body))
            assert {r.headers['x-moqe-l1-instance'] for r in responses}=={'l1-01','l1-02'}
            assert all(r.headers['x-moqe-expert']=='awq' for r in responses)
            assert app.state.gateway['inflight']==0
    asyncio.run(scenario())


def test_l1_verification_waits_for_rollout_and_checks_desired_node(monkeypatch,tmp_path):
    from types import SimpleNamespace
    cfg=manage_l1.config();cfg['instances']={'l1-01':{'node':'heteroserve-lab-209','enabled':True}}
    pod={'metadata':{'name':'new','uid':'new-uid','labels':{'heteroserve.io/instance':'l1-01'}},
         'spec':{'nodeName':'heteroserve-lab-209'},'status':{'conditions':[{'type':'Ready','status':'True'}]}}
    response=identity();response.update(pod_uid='new-uid',node='heteroserve-lab-209',physical_devices=[1])
    calls=[]
    def execute(cluster,args,**kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=json.dumps({'identity':response,'route':{}}) if args[0]=='exec' else '')
    monkeypatch.setattr(manage_l1,'kubectl',execute)
    monkeypatch.setattr(manage_l1,'get_pods',lambda *a:[pod])
    monkeypatch.setattr(manage_l1,'ROOT',tmp_path)
    result=manage_l1.verify({},cfg,manage_gateway.config())
    assert calls[0][:2]==['rollout','status'] and result['passed']
    response['node']='heteroserve-lab-210'
    pod['spec']['nodeName']='heteroserve-lab-210'
    with pytest.raises(RuntimeError,match='verification failed'):
        manage_l1.verify({},cfg,manage_gateway.config())
