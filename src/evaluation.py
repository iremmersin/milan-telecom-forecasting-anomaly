# Forecast accuracy metrics.
import numpy as np

def smape(a, f):
    return 100 * np.mean(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)))

def scores(a, f):
    err = a - f
    return {"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()), "sMAPE": smape(a, f)} 