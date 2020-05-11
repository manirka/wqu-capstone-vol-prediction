import logging.config
import os
import pandas as pd

from iexfinance.stocks import get_historical_intraday
from datetime import datetime

log = logging.getLogger(__name__)


class IEXDownloader:
    def __init__(self, iex_token):
        self._iex_token = iex_token
        self._from = datetime(2020, 1, 1)
        self._to = datetime.now()

    def download(self, ticker):
        log.info(f'Loading intraday data for {ticker}')
        path = f'data/intraday/{ticker}_intraday.pkl'

        os.environ['IEX_TOKEN'] = self._iex_token
        dfs = []
        for dt in pd.date_range(self._from, self._to, freq='B'):
            log.info(f'Loading date {dt}')
            df = get_historical_intraday(ticker, dt, output_format='pandas')
            if len(df) > 0:
                df = df.loc[:, ['date', 'open', 'close', 'volume']]
                dfs.append(df)

        pd.concat(dfs).to_pickle(path)
        log.info(f'Saved intraday data to {path}')
