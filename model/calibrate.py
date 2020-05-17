import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima

# Logic
# - read last 21+ days of intraday
# - produce params for requested model
# - produce vol curve
ROLLING_WINDOW = 21


def calibrate_ln_model(minutely_volume: pd.DataFrame):
    nud = minutely_volume.date.nunique()
    assert nud >= ROLLING_WINDOW, f"Number of days should be {ROLLING_WINDOW} or more, got {nud}"

    vc = get_vc(minutely_volume)
    daily = get_daily_volume(minutely_volume)
    log_volume_forecast, log_volume_var = get_volume_stats(daily)

    return vc, log_volume_forecast, log_volume_var


def get_volume_stats(daily_volume: pd.DataFrame):
    daily_volume["lv"] = np.log(daily_volume.volume)
    if len(daily_volume) > ROLLING_WINDOW * 3:
        daily_volume["mu_lv"] = daily_volume.lv.rolling(ROLLING_WINDOW, min_periods=1).mean().transform(np.ceil)
        daily_volume["excess_lv"] = daily_volume.lv - daily_volume.mu_lv
        model = auto_arima(daily_volume.excess_lv.dropna(), start_p=1, start_q=1, max_p=3, max_q=2, seasonal=False, d=0)
        predicted_excess = model.predict(1)[0]
        predicted_daily_lv = daily_volume.mu_lv.array[-1] + predicted_excess
    else:
        # fallback to volume geometric mean if there are too few observations
        predicted_daily_lv = daily_volume.volume[-ROLLING_WINDOW:].mean()

    return predicted_daily_lv, daily_volume.lv[-ROLLING_WINDOW:].var()


def get_vc(minutely_volume):
    """
    Parameters:
    minutely_volume (DataFrame): minutely volume indexed by datetime, expected columns: date, volume

    Returns:
    DataFrame: smoothed volume curve (percentage of daily volume) indexed by time
    """
    minutely_volume = minutely_volume.loc[:, ['date', 'volume']].copy()
    minutely_volume['raw_vc'] = minutely_volume.loc[:, ['date', 'volume']].groupby('date').apply(
        lambda x: x / float(x.sum()))
    minutely_volume['time'] = minutely_volume.index.time

    vc = minutely_volume.loc[:, ['time', 'raw_vc']].groupby('time').mean()  # average over days
    vc['smoothed'] = vc.raw_vc.rolling(window=3, min_periods=1).mean()  # use rolling avg to smooth
    return vc.smoothed.values / vc.smoothed.sum()  # normalize to 1


def get_daily_volume(minutely_volume):
    """
    Parameters:
    minutelyVolume (DataFrame): minutely volume indexed by datetime, expected columns: date, volume

    Returns:
    DataFrame: daily volume indexed by date
    """
    daily = minutely_volume.loc[:, ['date', 'volume']].groupby('date').agg({'volume': 'sum'}).copy()
    daily.index = pd.to_datetime(daily.index)
    return daily


if __name__ == "__main__":
    df = pd.read_pickle("../data/intraday/JNJ_intraday.2019-01-01_2019-12-31.pkl")
    vc, lv, lv_var = calibrate_ln_model(df.head(-1))
    print(f'vol_curve={np.array2string(vc, max_line_width=np.inf, separator=",")}\nlog_volume_forecast={lv}\nlog_volume_var={lv_var}')
