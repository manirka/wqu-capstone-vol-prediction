import logging.config
from aiohttp import web
import aiohttp

log = logging.getLogger(__name__)


class Listener:
    def __init__(self):
        pass

    def onmessage(self, message):
        return None


class WSHandler:
    def __init__(self, message_converter, listener_class):
        self._message_converter = message_converter
        if not issubclass(listener_class, Listener):
            raise ValueError('Second argument should be a subsclass of Listener')
        self._listener_class = listener_class


    async def handle(self, request):
        log.info('websocket connection opened')

        listener = self._listener_class()

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                response = listener.onmessage(self._message_converter.unmarshall(msg.data))
                await ws.send_json(self._message_converter.marshall(response))
            elif msg.type == aiohttp.WSMsgType.ERROR:
                log.info('ws connection closed with exception %s' % ws.exception())

        log.info('websocket connection closed')

        return ws
