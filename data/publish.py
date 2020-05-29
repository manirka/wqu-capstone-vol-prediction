import logging.config
import websockets
import pandas as pd
import asyncio

log = logging.getLogger(__name__)


class TickerPublisher:
    def __init__(self, ticker, date):
        self._ticker = ticker
        self._date = date
        self._date_bars = self.load_data()
        if self._date_bars.empty:
            log.error(f'No data found for {self._ticker} on {self._date.date()}. Market data won"t be published')

    def load_data(self):
        path = f'./data/intraday/{self._ticker}_intraday.2019-01-01_2019-12-31.pkl'
        all_bars = pd.read_pickle(path)
        return all_bars[all_bars['date'] == self._date].head(-1)

    async def publish(self, url, delay):
        async with websockets.connect(f'{url}{self._ticker}/') as websocket:
            for idx, bar in self._date_bars.iterrows():
                message = '{' + f'"action":"bar","ticker":"{self._ticker}","datetime":"{idx}",{bar.to_json()[1:]}'
                log.info(message)
                await websocket.send(message)
                await asyncio.sleep(delay/1000)


class Publisher:
    def __init__(self, tickers, date):
        self._publishers = [TickerPublisher(t, date) for t in tickers]

    async def prepare_tasks(self, url, delay):
        publish_tasks = [p.publish(url, delay) for p in self._publishers]
        return await asyncio.gather(*publish_tasks)
