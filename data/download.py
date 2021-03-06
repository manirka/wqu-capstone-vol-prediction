import logging.config
import os
import pandas as pd

from iexfinance.stocks import get_historical_intraday
from datetime import datetime

log = logging.getLogger(__name__)


class IEXDownloader:
    def __init__(self, iex_token, from_date=datetime(2020, 1, 1), to_date=datetime.now()):
        self._iex_token = iex_token
        self._from = from_date
        self._to = to_date

    def download(self, ticker):
        log.info(f'Loading intraday data for {ticker}')
        path = f'data/intraday/{ticker}_intraday.{self._from.strftime("%Y-%m-%d")}_{self._to.strftime("%Y-%m-%d")}.pkl'
        print(f'Target file path {path}')

        os.environ['IEX_TOKEN'] = self._iex_token
        dfs = []
        for dt in pd.date_range(self._from, self._to, freq='B'):
            print(f"Loading intraday data for {dt}")
            df = get_historical_intraday(ticker, dt, output_format='pandas')
            if len(df) > 0:
                df = df.loc[:, ['date', 'marketOpen', 'marketClose', 'marketVolume']]
                df.rename(columns={'marketOpen': 'open', 'marketClose': 'close', 'marketVolume': 'volume'},
                          inplace=True)
                dfs.append(df)

        intraday = pd.concat(dfs)
        intraday.index = pd.to_datetime(intraday.index)
        intraday.date = pd.to_datetime(intraday.date)
        print(f"Saved intraday data to {path}")
        intraday.to_pickle(path)
