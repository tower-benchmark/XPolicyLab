"""Raw-only admission across both server transports, without policy weights."""
from copy import deepcopy
import asyncio
import io
import socket
import threading

import numpy as np
import pytest
from PIL import Image

from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from client_server.ws.protocol.messages import MessageType
from client_server.ws.protocol.schemas import Frame
from client_server.tcp.model_server import ModelServer
from client_server.tcp.utils import numpy_to_json, json_to_numpy
from XPolicyLab.utils.process_data import decode_obs_images, decode_image_bit, encode_image_bit, prepare_obs_images
from XPolicyLab.policy.OpenWAM_TOWER.model import Model as TowerModel

RGB = np.full((8, 9, 3), [201, 35, 9], dtype=np.uint8)
MESSAGE = 'Policy server accepts raw RGB uint8 HxWx3 arrays only; decode on the client with XPolicyLab.utils.process_data.decode_image_bit'


def encoded(kind):
    if kind == 'marked':
        return encode_image_bit(RGB)
    out = io.BytesIO()
    Image.fromarray(RGB).save(out, format=kind)
    return out.getvalue()


class DefaultModel:
    def update_obs(self, obs):
        self.obs = obs
        return obs
    update_obs_batch = update_obs
    custom = update_obs
    infer = update_obs
    def get_action(self):
        return self.obs


class RawModel(DefaultModel):
    pass


def observation(value):
    return {'vision': {'front': {'color': value}}}


def request(model, transport, path, value, **config):
    obs = observation(value)
    if path == 'update_obs_batch':
        obs = [obs]
    if transport == 'ws':
        frame = Frame(message_type=MessageType.INFER if path == 'INFER' else MessageType.CALL,
                      request_id='raw', evaluation_id='test',
                      payload={'observation': obs} if path == 'INFER' else {'func_name': path, 'obs': obs})
        return asyncio.run(PolicyServer(model, PolicyServerConfig(**config)).process_frame(frame)).payload
    server = ModelServer(model, **config)
    server.running = True
    client, peer = socket.socketpair()
    client.settimeout(5)
    thread = threading.Thread(target=server._handle_client, args=(peer,))
    thread.start()
    try:
        raw = numpy_to_json({'cmd': path, 'obs': obs}).encode()
        client.sendall(len(raw).to_bytes(4, 'big') + raw)
        def receive(n):
            result = b''
            while len(result) < n:
                chunk = client.recv(n - len(result))
                assert chunk
                result += chunk
            return result
        return json_to_numpy(receive(int.from_bytes(receive(4), 'big')).decode())
    finally:
        client.close()
        thread.join(5)
        assert not thread.is_alive()


PATHS = [('ws', p) for p in ('INFER', 'custom', 'update_obs', 'update_obs_batch')] + [('tcp', p) for p in ('infer', 'custom', 'update_obs', 'update_obs_batch')]


@pytest.mark.parametrize('transport,path', PATHS)
@pytest.mark.parametrize('kind', ['PNG', 'JPEG', 'marked'])
@pytest.mark.parametrize('container', ['bytes', 'str', 'array'])
def test_encoded_rejected(transport, path, kind, container):
    value = encoded(kind)
    value = bytes(value) if not isinstance(value, bytes) else value
    if container == 'str':
        value = value.decode('latin1')
    elif container == 'array':
        value = np.frombuffer(value, dtype=np.uint8)
    model = RawModel()
    response = request(model, transport, path, value)
    message = response['message'] if transport == 'ws' else response['error']
    assert 'vision.front.color' in message and MESSAGE in message
    if transport == 'ws':
        assert response['code'] == 'invalid_frame'
    assert not hasattr(model, 'obs')


@pytest.mark.parametrize('transport,path', PATHS)
@pytest.mark.parametrize('kind', ['PNG', 'JPEG', 'marked', 'raw'])
def test_default_unchanged_and_raw_accepted(transport, path, kind):
    value = RGB.copy() if kind == 'raw' else encoded(kind)
    expected = decode_obs_images(deepcopy(observation(value)))["vision"]["front"]["color"]
    models = [DefaultModel(), RawModel()] if kind == 'raw' else [DefaultModel()]
    for model in models:
        response = request(model, transport, path, value, **({"raw_rgb_images_only": False} if kind != "raw" else {}))
        assert 'error' not in response and 'code' not in response
        obs = model.obs[0] if path == 'update_obs_batch' else model.obs
        np.testing.assert_array_equal(obs['vision']['front']['color'], expected)
        if transport == 'ws' and kind == 'raw':
            assert obs['vision']['front']['color'] is value


def test_tower_declaration_and_atomic_batch_validation():
    first = observation(RGB)
    with pytest.raises(ValueError, match=MESSAGE):
        prepare_obs_images([first, observation(np.array(encoded('PNG')))])
    assert first['vision']['front']['color'] is RGB


@pytest.mark.parametrize('field', ['color', 'colors', 'rgb', 'image', None])
@pytest.mark.parametrize('container', [bytearray, memoryview, lambda b: np.array(b), lambda b: np.array([b]), lambda b: np.frombuffer(b, dtype=np.uint8)[None, :]])
def test_all_colour_layouts(field, container):
    value = container(bytes(encoded('PNG')))
    obs = {'vision': {'front': {field: value} if field else value}}
    with pytest.raises(ValueError, match='vision.front.' + (field or 'color')):
        prepare_obs_images(obs)


@pytest.mark.parametrize('shape', [(8, 9, 4), (8, 9, 1), (8, 9), (1, 8, 9, 3)])
def test_non_rgb_arrays_rejected(shape):
    with pytest.raises(ValueError, match=MESSAGE):
        prepare_obs_images(observation(np.zeros(shape, np.uint8)))
    with pytest.raises(ValueError, match=MESSAGE):
        prepare_obs_images(observation(np.zeros((8, 9, 3), np.float32)))


def test_raw_infer_result_identical():
    before = request(DefaultModel(), 'ws', 'INFER', RGB.copy(), raw_rgb_images_only=False)
    after = request(RawModel(), 'ws', 'INFER', RGB.copy())
    before.pop('latency_ms', None)
    after.pop('latency_ms', None)
    assert numpy_to_json(before) == numpy_to_json(after)
    np.testing.assert_array_equal(after['actions']['vision']['front']['color'], RGB)


def test_tower_refuses_legacy_tcp():
    with pytest.raises(ValueError, match='ws protocol only'):
        TowerModel(dict(protocol='legacy_tcp', env_cfg_type='nero_dual', seed=42))


@pytest.mark.parametrize('kind', ['PNG', 'JPEG', 'marked', 'raw'])
def test_ws_wire_contract(kind):
    from client_server.ws.model_client import WsModelClient
    from client_server.ws.model_server import PolicyServerConfig
    from client_server.ws.protocol.exceptions import WsError, ErrorCode

    async def exercise():
        server = PolicyServer(RawModel(), PolicyServerConfig(host='127.0.0.1', port=0))
        await server.start()
        def client_request():
            client = WsModelClient(url=server.url, evaluation_id='test', trial_id='test')
            try:
                if kind == 'raw':
                    result = client._run(client._client.infer(observation(RGB))).payload["actions"]
                    np.testing.assert_array_equal(result['vision']['front']['color'], RGB)
                else:
                    with pytest.raises(WsError) as raised:
                        client.call('custom', observation(encoded(kind)))
                    assert raised.value.code == ErrorCode.INVALID_FRAME
                    assert MESSAGE in str(raised.value)
            finally:
                client.close()
        try:
            await asyncio.to_thread(client_request)
        finally:
            await server.stop()
    asyncio.run(exercise())
