# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from scipy.stats.mstats import gmean
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from model.calibrate import calibrate_ln_model
from model.start import Model

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# %matplotlib inline
plt.rcParams['figure.figsize'] = [10.0, 6.0]
# -

# # Initial params

rollingWindow = 20
ticker = 'FB'
start = datetime(2019, 1, 1)
end = datetime(2019, 12, 31)

# # Loading intraday data

path = f'../data/intraday/{ticker}_intraday.{start.strftime("%Y-%m-%d")}_{end.strftime("%Y-%m-%d")}.pkl'
intraday = pd.read_pickle(path)    
intraday.head()

# Prepare daily aggregate and geometric moving average

daily = intraday.groupby('date').agg({'open': 'first', 'close': 'last', 'volume': 'sum'})
daily.index = pd.to_datetime(daily.index)
daily['gmav'] = daily.volume.rolling(rollingWindow).agg(gmean).transform(np.ceil)
daily.tail()

sns.lineplot(x='date', y='value', hue='variable', data=daily.loc[:,['volume','gmav']].reset_index().melt(id_vars=['date']))
plt.show()

# # Fit lognorm dist
# For some reason scipy fails to accurately fit lognorm to volume, so fitting norm to log volume

sns.distplot(np.log(daily.volume), fit=stats.norm, axlabel='log volume')
plt.show()

_ = stats.probplot(np.log(daily.volume), dist=stats.norm,  plot=plt, rvalue=True)

# # Fitting invgamma

sns.distplot(daily.volume, fit=stats.invgamma)
plt.show()

_ = stats.probplot(daily.volume, dist=stats.invgamma, sparams=stats.invgamma.fit(daily.volume), plot=plt, rvalue=True)

# # ARMA for initial daily volume prediction

daily["lv"] = np.log(daily.volume)
daily["mu_lv"] = daily.lv.rolling(rollingWindow).mean()
daily["sigma_lv"] = daily.lv.rolling(rollingWindow).std()
daily["excess_lv"] = daily.lv - daily.mu_lv
daily.tail()

plot_acf(daily.excess_lv.dropna(), lags=10)
plt.show()

model = ARIMA(daily.excess_lv.dropna(), order=(2,0,1))
model_fit = model.fit(trend='nc')
print(model_fit.summary())

model_fit.plot_predict()
plt.show()

# # Intraday volume curve

intraday['raw_vc'] = intraday.loc[:,['date','volume']].groupby('date').apply(lambda x: x / float(x.sum()))
intraday['time'] = intraday.index.time
intraday.tail()

VC = intraday.loc[:,['time','raw_vc']].groupby('time').mean() # average over many days
VC.plot()
plt.show()

# ## Very spiky VC - need to apply smother

# rolling mean
VC['vc'] = VC.raw_vc.rolling(3, min_periods=1).mean()
VC.vc = VC.vc/VC.vc.sum() # normalize to 1
VC['cumVC'] = VC.vc.cumsum()
VC.vc.plot()
plt.show()

# # Prediction

realized = intraday.where(intraday.date==intraday.date.max()).dropna().reset_index().loc[:,['time','volume']].set_index('time')
realized["cumVolume"] = realized.volume.cumsum()
finalVolume = realized.cumVolume[-1]

predictedLogExcess = model_fit.forecast()[0]
predictedDailyLV = daily.mu_lv.array[-1] + predictedLogExcess # previous day rolling mean + excess
predictedDailyLV

DailyLVvar = daily.lv.rolling(rollingWindow).var()[-1]
DailyLVvar

k0 = 0.8 * rollingWindow

prediction = realized.join(VC)
prediction['x'] = np.log(prediction.volume/prediction.vc)
prediction['n'] = np.arange(1,len(prediction)+1)
prediction['xbar'] = prediction.x.expanding().mean()
prediction['sigma2_x'] = prediction.x.expanding().var()
prediction['mu'] = (predictedDailyLV*k0+prediction.n*prediction.xbar)/(k0+prediction.n) # apply 'unknown mean and variance' prediction
prediction['predictedVolume'] = np.ceil(np.exp(prediction.mu))
prediction

# apply 'unknown mean, known variance' prediction
prediction.loc[prediction.n>100, 'mu'] = (prediction.n*prediction.xbar/prediction.sigma2_x + \
    predictedDailyLV/DailyLVvar)/(prediction.n/prediction.sigma2_x + 1/DailyLVvar)
prediction

prediction.predictedVolume.plot()
plt.axhline(y=finalVolume, c='red', ls='--')
plt.plot()

PERF = prediction.loc[:,['predictedVolume']]
PERF['elpv'] = np.log(PERF.predictedVolume)-np.log(finalVolume)
PERF['SE'] = np.power(PERF.predictedVolume - finalVolume, 2)
PERF['w'] = PERF.elpv.apply(lambda x: 2 if x>0 else 1)
PERF['LE'] = PERF.w*PERF.elpv.abs()
PERF

# RMSE
np.sqrt(PERF.SE.sum()/len(PERF))

# ALE
PERF.LE.sum()

# # Comparison with naive gmean

# RMSE
np.abs(finalVolume - daily.gmav[-1])

# ALE
d = np.log(daily.rolling(rollingWindow).volume.agg(gmean)[-2]) - np.log(finalVolume)
w = 2 if d>0 else 1
len(PERF)*w*np.abs(d)


# # Forecasting performance
#
# For each date:
# - IG 
#  - obtain trailing 21-day avg volume for each minute
#  - obtain trailing 21-day var volume for each minute
#  - obtain daily volume for previous 21 day, compute avg and var
#  - compute {$\alpha_t$}
#  - obtain the current day's realized volume for each minute
#  - for each minute compute forecasted daily volume and forecast error
#  - calculate RMSE for the date
#  
# - LN 
#  - obtain daily volume for previous 21 day, compute avg and var
#  - train ARMA, predict initial daily volume
#  - for each minute compute avg percentage of daily volume, smooth and normalize
#  - obtain the current day's realized volume for each minute
#  - for each minute compute forecasted daily volume and forecast error
#  - calculate RMSE for the date

def ALE(forecast, realized):
    d = forecast - realized
    w = np.ones(len(d))
    w[w>0] = 2
    return np.sum(np.abs(d) * w)


def RMSE(forecast, realized):
    return np.sqrt(np.mean((forecast-realized)**2))


# +
dates = np.flip(intraday.date.unique())
k0 = 0.8 * rollingWindow

errors = pd.DataFrame(columns=['ALE_full', 'ALE_naive', 'RMSE_full', 'RMSE_naive'], index=pd.to_datetime([]))
for i, dt in enumerate(dates[:-6*rollingWindow-1]):
    print(dt)
    df_validate = intraday[intraday.date == dt]
    realized_volume = df_validate.volume.sum()
    realized_log_volume = np.log(realized_volume)
    
    train_start_dt, train_end_dt = dates[[i+6*rollingWindow+1, i+1]]
    df_train = intraday[intraday.date.between(train_start_dt, train_end_dt)]
    
    vc, initial_log_volume_forecast, log_volume_var = calibrate_ln_model(df_train)
    model = Model(ticker, vc, initial_log_volume_forecast, log_volume_var)

    realized = np.repeat(np.log(realized_volume), len(df_validate))
    naivePrediction = np.repeat(np.log(gmean(df_train.groupby('date').volume.sum()[-rollingWindow:-1])), len(df_validate))
    
    fullPrediction = []
    for _, row in df_validate.loc[:,['time','volume']].iterrows():
        fullPrediction.append(np.log(model.on_bar(row)))
    
    fullPrediction = np.array(fullPrediction)
    errors.loc[dt] = (ALE(fullPrediction, realized), ALE(naivePrediction, realized), RMSE(fullPrediction, realized),  RMSE(naivePrediction, realized))
# -
errors

errors.loc[:,['ALE_full','ALE_naive']].plot()
plt.show()

errors.loc[:,['RMSE_full','RMSE_naive']].plot()
plt.show()


