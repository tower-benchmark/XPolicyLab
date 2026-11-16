import asyncio
import socket
import threading
from types import SimpleNamespace
import sys

import pytest
from client_server.ws.model_server import PolicyServerConfig
from client_server.tcp.model_server import ModelServer
from client_server.tcp.model_client import ModelClient
from XPolicyLab.policy.OpenWAM_TOWER.model import Model
from XPolicyLab.utils.process_data import prepare_obs_images
from test_raw_images import DefaultModel, RGB, observation, encoded

@pytest.mark.parametrize('value', ['false', 0, 1, None, [], {}])
@pytest.mark.parametrize('factory', [PolicyServerConfig, lambda **kw: ModelServer(DefaultModel(), **kw)])
def test_strict_bool(value, factory):
    with pytest.raises(ValueError, match='raw_rgb_images_only must be a bool'):
        factory(raw_rgb_images_only=value)

@pytest.mark.parametrize('protocol', ['ws', 'legacy_tcp'])
@pytest.mark.parametrize('value', [True, False, 'absent'])
def test_setup_forwarding(monkeypatch, protocol, value):
    import setup_policy_server as setup
    import client_server.ws.model_server as ws
    seen = []
    class Fake:
        def __init__(self, model, config=None, **kw):
            seen.append(config.raw_rgb_images_only if config else kw['raw_rgb_images_only'])
        async def serve_forever(self): pass
        def start(self): pass
        def stop(self): pass
    monkeypatch.setattr(setup, 'eval_function_decorator', lambda *a: lambda cfg: DefaultModel())
    monkeypatch.setattr(ws, 'PolicyServer', Fake)
    monkeypatch.setattr(setup, 'ModelServer', Fake)
    monkeypatch.setattr(setup.time, 'sleep', lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    cfg = dict(policy_name='stub', protocol=protocol, port=0)
    if value != 'absent': cfg['raw_rgb_images_only'] = value
    setup.main(cfg)
    assert seen == [True if value == 'absent' else value]

@pytest.mark.parametrize('value', [True, False, 'absent'])
def test_ws_main_forwarding(monkeypatch, tmp_path, value):
    import json
    import client_server.ws.model_server as ws
    cfg = dict(policy_name='stub')
    if value != 'absent': cfg['raw_rgb_images_only'] = value
    path = tmp_path/'deploy.json'; path.write_text(json.dumps(cfg))
    seen = []
    class Fake:
        def __init__(self, model, config): seen.append(config.raw_rgb_images_only)
        async def serve_forever(self): pass
    monkeypatch.setattr(ws, 'PolicyServer', Fake)
    monkeypatch.setattr(ws.importlib, 'import_module', lambda _: SimpleNamespace(Model=lambda _: DefaultModel()))
    monkeypatch.setattr(sys, 'argv', ['server', '--config-path', str(path)])
    ws.main()
    assert seen == [True if value == 'absent' else value]

def test_tower_refuses_opt_out():
    with pytest.raises(ValueError, match='OpenWAM_TOWER refuses raw_rgb_images_only: false'):
        Model({'raw_rgb_images_only': False})

@pytest.mark.parametrize('bad', [False, True])
def test_tcp_client_error_and_none_over_socket(bad):
    class ModelWithNone(DefaultModel):
        def update_obs(self, obs): self.obs = obs
    model = ModelWithNone(); server = ModelServer(model); server.running = True
    listener = socket.socket(); listener.bind(('127.0.0.1', 0)); listener.listen()
    thread = threading.Thread(target=lambda: server._handle_client(listener.accept()[0]))
    thread.start(); client = ModelClient(host='127.0.0.1', port=listener.getsockname()[1])
    try:
        if bad:
            with pytest.raises(RuntimeError, match='accepts raw RGB uint8 HxWx3 arrays only'):
                client.call('update_obs', observation(encoded('PNG')))
            assert not hasattr(model, 'obs')
        else:
            assert client.call('update_obs', observation(RGB)) is None
    finally:
        client.close(); listener.close(); thread.join(5)
        assert not thread.is_alive()

def test_g05_nested_history_untouched():
    history = [observation(encoded('PNG'))]
    obs = dict(observation(RGB), __g05_history_observations__=history)
    assert prepare_obs_images(obs) is obs
    assert obs['__g05_history_observations__'] is history
    assert isinstance(history[0]['vision']['front']['color'], bytes)

@pytest.mark.parametrize('value', ['false', 0, None])
def test_setup_invalid_before_model(monkeypatch, value):
    import setup_policy_server as setup
    monkeypatch.setattr(setup, 'eval_function_decorator', lambda *a: pytest.fail('model must not load'))
    with pytest.raises(ValueError, match='raw_rgb_images_only must be a bool'):
        setup.main(dict(policy_name='stub', raw_rgb_images_only=value))

def test_policy_name_and_opt_out_in_error():
    with pytest.raises(ValueError) as error:
        prepare_obs_images(observation(encoded('PNG')), True, 'ExamplePolicy')
    assert 'ExamplePolicy accepts raw RGB uint8 HxWx3 arrays only' in str(error.value)
    assert 'raw_rgb_images_only: false in deploy.yml' in str(error.value)
    assert 'XPolicyLab.utils.process_data.decode_image_bit' in str(error.value)
