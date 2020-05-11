import logging.config
from statistics import mean
from server.wshandler import Listener

log = logging.getLogger(__name__)


class Bar:
    def __init__(self, high, low):
        self.high = high
        self.low = low

    def calc(self):
        return self.high + self.low


class Model(Listener):

    def __init__(self):
        super().__init__()
        self.bars = []
        log.info('Created class {}   '.format(__name__))

    def bootstrap(self):
        log.debug('Bootstrapped')

    def onmessage(self, bar):
        log.info('Updated')
        self.bars.append(bar.calc())
        return mean(self.bars)

    def stop(self):
        log.error('Stopped')
