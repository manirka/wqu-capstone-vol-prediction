import logging.config
import pandas as pd
import numpy as np
from server.wshandler import Listener

log = logging.getLogger(__name__)


class Model(Listener):

    def __init__(self, ticker, vol_curve, log_volume_forecast, log_volume_var, k0=0.8):
        super().__init__()
        self._log_volume_var = log_volume_var
        self._ticker = ticker
        self._k0 = k0
        self._log_volume_forecast = log_volume_forecast

        self._bars = pd.DataFrame(columns=['v', 'x', 'n', 'xbar', 'mu', 'predV'],
                                  index=pd.date_range("09:30", "15:59", freq="1min").time)
        self._bars['vc'] = vol_curve
        self._bars['n'] = np.arange(1, len(vol_curve) + 1)
        log.info(f'{self._ticker} model created')

    def on_message(self, message):
        if isinstance(message, pd.Series):
            if message.name == 'subscription':
                return self.on_subscription(message)
            else:
                return self.on_bar(message)
        else:
            return 0

    def on_subscription(self, _):
        return self._bars['vc']

    def on_bar(self, message):
        log.info(f'{self._ticker} time: {message.time}, volume: {message.volume}')
        t = message.time
        self._bars.loc[t, 'v'] = v = message.volume
        self._bars.loc[t, 'x'] = np.log(v / self._bars.loc[t, 'vc'])
        self._bars.loc[t, 'xbar'] = xbar = self._bars.x.dropna().mean()

        n = self._bars.loc[t, 'n']
        if n < 100:
            mu = (self._log_volume_forecast * self._k0 + n * xbar) / (self._k0 + n)
        else:
            # switch to 'known variance' once we have 100 observations
            # https://stats.stackexchange.com/questions/7004/calculating-required-sample-size-precision-of-variance-estimate
            var_x = self._bars.x.dropna().var()
            mu = (n * xbar / var_x + self._log_volume_forecast / self._log_volume_var) / \
                 (n / var_x + 1 / self._log_volume_var)
        self._bars.loc[t, 'mu'] = mu
        self._bars.loc[message.time, 'pred_v'] = pred_v = np.ceil(np.exp(mu))

        return pred_v

    def stop(self):
        log.info(f'{self._ticker} model stopped')
