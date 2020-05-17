import asyncio
import datetime
import logging.config
import os
import re

logging.config.fileConfig('config/logging.conf')
log = logging.getLogger(__name__)

import argparse
import configparser
import webbrowser

from aiohttp import web

from data.download import IEXDownloader
from data.publish import Publisher
from model.start import *
from server.marshaller import *
from server.wshandler import WSHandler


def parse_arguments():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("action", help="supported actions: start, download")
    args_parser.add_argument("subaction", help="for start action: server; for download action: ticker")
    args_parser.add_argument("--iex_token", help="IEX TOKEN for downloading data", required=False)
    args_parser.add_argument("--from", help="Start date for downloading data", required=False)
    args_parser.add_argument("--to", help="End date for downloading data", required=False)
    args_parser.add_argument("--ticker", help="Ticker for client", required=False, default='GOOGL')
    args_parser.add_argument("--delay", help="Delay in milliseconds for publisher", required=False, type=lambda d: int(d), default=1000)
    args_parser.add_argument("--date", help="Date for publisher", required=False, type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d'), default='2019-05-01')

    return args_parser.parse_args()


def main():
    args = parse_arguments()

    settings = configparser.ConfigParser()
    settings.read('config/settings.conf')

    if 'start' == args.action:
        if 'server' == args.subaction:
            start_server(settings)
        elif 'client' == args.subaction:
            start_client(settings, args)
        elif 'publisher' == args.subaction:
            start_publisher(settings, args)
    elif 'download' == args.action:
        if not args.iex_token:
            log.error('iex_token is mandatory for downloading data')
        else:
            download_ticker(args.subaction, args.iex_token)
    else:
        log.error(f'Unknown action {args.action}')


def start_server(settings):
    log.info('START')

    message_converter = MessageConverter()
    message_converter.register_unmarshaller('bar', BarUnmarshaller())
    message_converter.register_unmarshaller('sub', SubUnmarshaller())
    message_converter.register_marshaller('float64', NumberMarshaller())
    message_converter.register_marshaller('int', NumberMarshaller())
    message_converter.register_marshaller('pandas.Series', SeriesMarshaller())

    handler = WSHandler(message_converter)
    app = web.Application()

    for s in settings.sections():
        if s.startswith('model.'):
            model_name = s.split('.')[1]
            vol_curve = eval(settings[s]['vol_curve'])
            log_volume_forecast = float(settings[s]['log_volume_forecast'])
            log_volume_var = float(settings[s]['log_volume_var'])
            handler.add_listener(model_name, Model(model_name, vol_curve, log_volume_forecast, log_volume_var))
            app.add_routes([web.get(f'/client/{model_name}/', handler.handle_client)])
            app.add_routes([web.get(f'/md/{model_name}/', handler.handle_marketdata)])

    web.run_app(app, host=settings['server']['host'], port=settings['server']['port'])

    log.info('END')


def start_client(settings, args):
    # start client --ticker=MSFT
    ticker = args.ticker
    #url = f"file://{os.path.realpath('client/client.html')}?host={settings['server']['host']}&port={settings['server']['port']}&ticker={ticker}"
    if not os.path.exists(f'client/client_{ticker}.html'):
        f = open('client/client.html', 'rt')
        html = f.read()
        f.close()
        html = html.replace('HOST', settings['server']['host'])
        html = html.replace('PORT', settings['server']['port'])
        html = html.replace('TICKER', ticker)
        f = open(f'client/client_{ticker}.html', 'wt')
        f.write(html)
        f.close()
    webbrowser.open(f"file://{os.path.realpath('client/client_'+ticker+'.html')}", new=2)


def start_publisher(settings, args):
    # start publisher --date=2019.05.02 --delay=2000
    models = [s.split('.')[1] for s in settings.sections() if s.startswith('model.')]
    p = Publisher(models, args.date).prepare_tasks(f"ws://{settings['server']['host']}:{settings['server']['port']}/md/", args.delay)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(p)


def download_ticker(ticker, iex_token):
    IEXDownloader(iex_token).download(ticker)


if __name__ == "__main__":
    main()



