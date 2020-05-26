import asyncio
import datetime
import logging.config
import os

logging.config.fileConfig('config/logging.conf')
log = logging.getLogger(__name__)

import argparse
import configparser
import webbrowser

from aiohttp import web

from data.download import IEXDownloader
from data.publish import Publisher
from model.start import *
from model.calibrate import *
from server.marshaller import *
from server.wshandler import WSHandler


def parse_arguments():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("action", help="supported actions: start, download")
    args_parser.add_argument("subaction", help="for start action: server; for download action: ticker")
    args_parser.add_argument("--iex_token", help="IEX TOKEN for downloading data", required=False)
    args_parser.add_argument("--from", help="Start date for downloading data", required=False)
    args_parser.add_argument("--to", help="End date for downloading data", required=False)
    args_parser.add_argument("--ticker", help="Ticker for client", required=False)
    args_parser.add_argument("--delay", help="Delay in milliseconds for publisher", required=False, type=lambda d: int(d), default=2000)
    args_parser.add_argument("--date", help="Date for publisher", required=False, type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d'), default='2019-12-02')

    return args_parser.parse_args()


def main():
    args = parse_arguments()

    settings = configparser.ConfigParser()
    settings.read('config/settings.conf')

    if 'start' == args.action:
        if 'server' == args.subaction:
            start_server(settings, args)
        elif 'client' == args.subaction:
            start_client(settings, args)
        elif 'publisher' == args.subaction:
            start_publisher(settings, args)
        elif 'calibrator' == args.subaction:
            start_calibrator(settings, args)
    elif 'download' == args.action:
        if not args.iex_token:
            log.error('iex_token is mandatory for downloading data')
        else:
            download_ticker(args.subaction, args.iex_token)
    else:
        log.error(f'Unknown action {args.action}')


def start_server(settings, args):
    message_converter = MessageConverter()
    message_converter.register_unmarshaller('bar', BarUnmarshaller())
    message_converter.register_unmarshaller('sub', SubUnmarshaller())
    message_converter.register_marshaller('float64', NumberMarshaller())
    message_converter.register_marshaller('int', NumberMarshaller())
    message_converter.register_marshaller('Series', SeriesMarshaller())

    handler = WSHandler(message_converter)
    app = web.Application()

    for model in get_models(settings, args):
        setting = settings[f'model.{model}']
        vol_curve = eval(setting['vol_curve'])
        log_volume_forecast = float(setting['log_volume_forecast'])
        log_volume_var = float(setting['log_volume_var'])
        handler.add_listener(model, Model(model, vol_curve, log_volume_forecast, log_volume_var))
        app.add_routes([web.get(f'/client/{model}/', handler.handle_client)])
        app.add_routes([web.get(f'/md/{model}/', handler.handle_marketdata)])

    web.run_app(app, host=settings['server']['host'], port=settings['server']['port'])


def start_client(settings, args):
    # start client --ticker=MSFT
    #url = f"file://{os.path.realpath('client/client.html')}?host={settings['server']['host']}&port={settings['server']['port']}&ticker={ticker}"
    for ticker in get_models(settings, args):
        if not os.path.exists(f'client/client_{ticker}1.html'):
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
    # start publisher --date=2019.12.02 --delay=2000
    p = Publisher(get_models(settings, args), args.date).prepare_tasks(f"ws://{settings['server']['host']}:{settings['server']['port']}/md/", args.delay)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(p)


def start_calibrator(settings, args):
    # start calibrator --date=2019-12-02
    for model in get_models(settings, args):
        df = pd.read_pickle(f'./data/intraday/{model}_intraday.2019-01-01_2019-12-31.pkl')
        df = df[df.date < args.date].head(-1)
        vc, lv, lv_var = Calibrator(df).calibrate_ln_model()
        settings.set(f'model.{model}', 'vol_curve', np.array2string(vc, max_line_width=np.inf, separator=","))
        settings.set(f'model.{model}', 'log_volume_forecast', f'{lv}')
        settings.set(f'model.{model}', 'log_volume_var', f'{lv_var}')
    with open('config/settings.conf', 'w') as configfile:
        settings.write(configfile)


def get_models(settings, args):
    return [args.ticker] if args.ticker else [s.split('.')[1] for s in settings.sections() if s.startswith('model.')]


def download_ticker(ticker, iex_token):
    IEXDownloader(iex_token).download(ticker)


if __name__ == "__main__":
    main()



