import logging.config

logging.config.fileConfig('config/logging.conf')
log = logging.getLogger(__name__)

import argparse

from data.download import IEXDownloader
from model.start import *
from aiohttp import web

from server.marshaller import MessageConverter, BarUnmarshaller, FloatMarshaller
from server.wshandler import WSHandler



def parse_arguments():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("action", help="supported actions: start, download")
    args_parser.add_argument("subaction", help="for start action: server; for download action: ticker")
    args_parser.add_argument("--iex_token", help="IEX TOKEN for downloading data", required=False)

    return args_parser.parse_args()


def main():
    args = parse_arguments()

    if 'start' == args.action:
        # do we want to do start client?
        start_server()
    elif 'download' == args.action:
        if not args.iex_token:
            log.error('iex_token is mandatory for downloading data')
        else:
            download_ticker(args.subaction, args.iex_token)
    else:
        log.error('Unknown action {}'.format(args.action))


def start_server():
    log.info('START')

    message_converter = MessageConverter()
    message_converter.register_unmarshaller('bar', BarUnmarshaller())
    message_converter.register_marshaller('float', FloatMarshaller())
    handler = WSHandler(message_converter, Model)

    app = web.Application()
    app.add_routes([web.get('/', handler.handle)])
    web.run_app(app, host='127.0.0.1', port=5678)

    log.info('END')


def download_ticker(ticker, iex_tocken):
    IEXDownloader(iex_tocken).download(ticker)


if __name__ == "__main__":
    main()


