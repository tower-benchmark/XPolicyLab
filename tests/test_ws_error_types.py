"""Model exception categories survive the real WebSocket client/server boundary."""
import asyncio
import unittest

from client_server.ws.model_client import WsModelClient
from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from client_server.ws.protocol.exceptions import ErrorCode, WsError


class PolicyOutputError(ValueError):
    pass


class Model:
    exception = ValueError

    def get_action(self):
        raise self.exception('synthetic rejection')

    def update_obs(self, observation):
        pass


class ErrorTypes(unittest.IsolatedAsyncioTestCase):
    async def test_call_details_reach_synchronous_client(self):
        model = Model()
        server = PolicyServer(model, PolicyServerConfig(host='127.0.0.1', port=0))
        await server.start()

        def exercise():
            client = WsModelClient(url=server.url, evaluation_id='test', trial_id='test')
            try:
                for category in (ValueError, PolicyOutputError, RuntimeError):
                    model.exception = category
                    with self.assertRaises(WsError) as raised:
                        client.call('get_action')
                    self.assertEqual(raised.exception.code, ErrorCode.CALL_FAILED)
                    self.assertEqual(str(raised.exception), 'synthetic rejection')
                    self.assertEqual(raised.exception.details, {'exception_type': category.__name__})
            finally:
                client.close()

        try:
            await asyncio.to_thread(exercise)
        finally:
            await server.stop()

    async def test_infer_details(self):
        from client_server.ws.protocol.messages import MessageType
        from client_server.ws.protocol.schemas import Frame

        server = PolicyServer(Model())
        response = await server.process_frame(Frame(
            message_type=MessageType.INFER, request_id='infer', evaluation_id='test',
            payload={'observation': {'synthetic': True}},
        ))
        self.assertEqual(response.payload['code'], 'infer_failed')
        self.assertEqual(response.payload['message'], 'synthetic rejection')
        self.assertEqual(response.payload['details'], {'exception_type': 'ValueError'})
