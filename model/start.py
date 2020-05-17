import logging.config
from statistics import mean

import pandas as pd

from server.wshandler import Listener

log = logging.getLogger(__name__)


class Model(Listener):

    def __init__(self, ticker):
        super().__init__()
        self._ticker = ticker
        self._bars = pd.DataFrame(columns=['ticker', 'open', 'close', 'volume'])
        log.info(f'Created model {self._ticker}')

    def bootstrap(self):
        log.debug('Bootstrapped')

    def onmessage(self, message):
        if isinstance(message, pd.Series):
            if message.name == 'subscription':
                return self.onsubscription(message)
            else:
                return self.onbar(message)
        else:
            return 0

    def onsubscription(self, message):
        # TODO respond with vol curve series and volume
        return -1

    def onbar(self, message):
        # TODO respond with remaining volume
        log.info('Market data update')
        self._bars.loc[message.name] = message
        return mean(self._bars['volume'])

    def stop(self):
        log.info(f'Stopped model {self._ticker}')
