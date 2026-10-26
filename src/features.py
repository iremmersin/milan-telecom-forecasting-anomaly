# Feature engineering for the day ahead forecasting setup.
import numpy as np
import pandas as pd

LAGS = (24, 48, 168)  # all lags >= 24h: only data up to the previous day is available
HOLIDAYS = set(pd.to_datetime(
    ["2013-11-01", "2013-12-07", "2013-12-08", "2013-12-25", "2013-12-26", "2014-01-01"]
).date)


def make_features(s, holiday_as_sunday=True):
    df = pd.DataFrame({"y": s})
    for lag in LAGS:
        df[f"lag_{lag}"] = s.shift(lag)
    df["roll_mean_24"] = s.shift(24).rolling(24).mean()
    df["roll_mean_168"] = s.shift(24).rolling(168).mean()

    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    is_hol = pd.Series(df.index.date, index=df.index).isin(HOLIDAYS)
    dow = pd.Series(df.index.day_of_week, index=df.index)
    # Weekday holidays follow the Sunday schedule (calendar rule, not learned from data)
    df["day_type"] = dow.mask(is_hol & holiday_as_sunday, 6)
    df["is_weekend"] = (df["day_type"] >= 5).astype(int)
    df["is_holiday"] = is_hol.astype(int)
    return df.dropna()


def check_no_leakage(s, pos=600):
    # Features at time t must not change when data after t-23h is corrupted.
    base = make_features(s)
    t = base.index[pos]
    s_bad = s.copy()
    s_bad.loc[t - pd.Timedelta(hours=23):] += 1e6
    bad = make_features(s_bad)
    cols = [c for c in base.columns if c != "y"]
    pd.testing.assert_series_equal(base.loc[t, cols], bad.loc[t, cols])
    print("no leakage detected at", t)