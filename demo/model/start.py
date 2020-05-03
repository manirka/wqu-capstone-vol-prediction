import logging.config
from statistics import mean

logging.config.fileConfig('config/logging.conf')
logger = logging.getLogger(__name__)


class Bar:
    def __init__(self, high, low):
        self.high = high
        self.low = low

    def calc(self):
        return self.high + self.low


class Model:

    def __init__(self, name):
        self.name = name
        self.bars = []
        logger.info('Created class {} {}'.format(self.name, __name__))

    def bootstrap(self):
        logger.debug('Bootstrapped')

    def predict(self, bar):
        logger.info('Updated')
        self.bars.append(bar.calc())
        return mean(self.bars)

    def stop(self):
        logger.error('Stopped')
