import logging.config
from aiohttp import web
import aiohttp

log = logging.getLogger(__name__)


class Listener:
    def __init__(self):
        pass

    def onmessage(self, message):
        return None


class WSHandler(Listener):
    def __init__(self, message_converter):
        super().__init__()
        self._message_converter = message_converter
        self._message_listeners = {}
        self._clients = {}

    def add_listener(self, key, listener):
        if not isinstance(listener, Listener):
            raise ValueError('Listener must be a subsclass of Listener')
        self._message_listeners[key] = listener
        self._clients[key] = []

    def path_to_key(self, path):
        return path.split('/')[2]

    def on_client_connect(self, key, ws):
        self._clients[key].append(ws)

    def on_client_disconnect(self, key, ws):
        if key in self._clients:
            self._clients[key].remove(ws)

    async def notify_clients(self, key, message):
        for ws in self._clients[key]:
            await ws.send_json(message)

    async def handle_marketdata(self, request):
        key = self.path_to_key(request.path)
        listener = self._message_listeners[key]
        log.info(f'{key} market data connection opened')

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                log.info(f'received MD {msg.data}')
                response = listener.onmessage(self._message_converter.unmarshall(msg.data))
                # on market data update notify all clients
                await self.notify_clients(key, self._message_converter.marshall(response))
            elif msg.type == aiohttp.WSMsgType.ERROR:
                log.info(f'ws connection closed with exception {ws.exception()}')

        log.info(f'{key} market data connection closed')

        return ws

    async def handle_client(self, request):
        key = self.path_to_key(request.path)
        listener = self._message_listeners[key]

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.on_client_connect(key, ws)
        log.info(f'{key} client connection opened')

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                response = listener.onmessage(self._message_converter.unmarshall(msg.data))
                await ws.send_json(self._message_converter.marshall(response))
            elif msg.type == aiohttp.WSMsgType.ERROR:
                log.info(f'ws connection closed with exception {ws.exception()}')

        self.on_client_disconnect(key, ws)
        log.info(f'{key} client connection closed')
        return ws
