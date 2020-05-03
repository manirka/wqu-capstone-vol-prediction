from model.start import *
import logging.config

from aiohttp import web
import aiohttp
import json


logging.config.fileConfig('config/logging.conf')
logger = logging.getLogger(__name__)


def decode(data):
    parsed = json.loads(data)
    try:
        return Bar(float(parsed['high']), float(parsed['low']))
    except:
        return Bar(0, 0)


def encode(data):
    return {'type': 'result', 'value': data}


async def websocket_handler(request):

    logger.info('websocket connection started')

    s = Model('test')
    s.bootstrap()

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            predicted = s.predict(decode(msg.data))
            await ws.send_json(encode(predicted))
        elif msg.type == aiohttp.WSMsgType.ERROR:
            logger('ws connection closed with exception %s' % ws.exception())

    logger.info('websocket connection closed')
    s.stop()

    return ws


def main():
    logger.info('START')

    app = web.Application()
    app.add_routes([web.get('/', websocket_handler)])
    web.run_app(app, host='127.0.0.1', port=5678)

    logger.info('END')


if __name__ == "__main__":
    main()

