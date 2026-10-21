"""Module 6 — time series: decomposition, smoothing-based and regression-based forecasting.

OWID emissions are ANNUAL, so there is no seasonal component to estimate. Classical
decomposition is implemented (and verified against the course's quarterly-sales example)
but on the CO2 data only the trend part (centred moving average) is meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import regression as rg


def centred_moving_average(y, m: int) -> np.ndarray:
    """Odd m: simple centred m-term MA. Even m: 2×m MA, T(t) = [½y(t−m/2) + … + ½y(t+m/2)]/m."""
    y = np.asarray(y, float)
    n = y.size
    out = np.full(n, np.nan)
    if m % 2:
        h = m // 2
        for t in range(h, n - h):
            out[t] = y[t - h:t + h + 1].mean()
    else:
        h = m // 2
        w = np.r_[0.5, np.ones(m - 1), 0.5] / m
        for t in range(h, n - h):
            out[t] = (y[t - h:t + h + 1] * w).sum()
    return out


def classical_decomposition(y, m: int, model: str = "additive") -> dict:
    """Course steps: trend by centred MA → detrend → average by season → adjust indices
    (sum 0 additive / sum m multiplicative) → linear trend per period for forecasting."""
    y = np.asarray(y, float)
    n = y.size
    trend = centred_moving_average(y, m)
    detr = y - trend if model == "additive" else y / trend
    season = np.arange(n) % m
    raw = np.array([np.nanmean(detr[season == s]) for s in range(m)])
    if model == "additive":
        idx = raw - raw.sum() / m
        resid = y - trend - idx[season]
    else:
        idx = raw / (raw.sum() / m)
        resid = y / (trend * idx[season])
    valid = np.flatnonzero(~np.isnan(trend))
    first, last = valid[0], valid[-1]
    slope = (trend[last] - trend[first]) / (last - first)

    def forecast(t_index: int) -> float:
        """t_index is 0-based position in the series (e.g. n + 1 for two steps ahead)."""
        tr = trend[last] + (t_index - last) * slope
        s = idx[t_index % m]
        return float(tr + s if model == "additive" else tr * s)

    return {"trend": trend, "detrended": detr, "raw_indices": raw, "seasonal_indices": idx,
            "residual": resid, "trend_per_period": float(slope), "forecast": forecast}


# ---------------------------------------------------------------- forecasting methods (one-step-ahead)

def average_forecast(y) -> np.ndarray:
    """F(t) = mean(y1 … y(t−1)); F(1) undefined."""
    y = np.asarray(y, float)
    c = np.cumsum(y)
    f = np.full(y.size, np.nan)
    f[1:] = c[:-1] / np.arange(1, y.size)
    return f


def moving_average_forecast(y, k: int = 3) -> np.ndarray:
    """F(t) = mean(y(t−k) … y(t−1)); undefined for the first k points."""
    y = np.asarray(y, float)
    f = np.full(y.size, np.nan)
    for t in range(k, y.size):
        f[t] = y[t - k:t].mean()
    return f


def naive_forecast(y) -> np.ndarray:
    y = np.asarray(y, float)
    f = np.full(y.size, np.nan)
    f[1:] = y[:-1]
    return f


def mase(actual, forecast, scale_series) -> float:
    """MAE / in-sample MAE of the naïve forecast (mean |y_j − y_{j−1}| of scale_series)."""
    scale = np.mean(np.abs(np.diff(np.asarray(scale_series, float))))
    return rg.mae(actual, forecast) / scale


def evaluate(actual, forecast, scale_series) -> dict:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    ok = ~np.isnan(f)
    a, f = a[ok], f[ok]
    return {"n": int(a.size), "MAE": rg.mae(a, f), "RMSE": rg.rmse(a, f), "MAPE": rg.mape(a, f),
            "MASE": mase(a, f, scale_series)}


def holdout_comparison(series: pd.Series, train_end: int, k: int = 3, trend_start: int | None = None) -> dict:
    """Compare methods on the years after `train_end`.

    Average / moving average / naïve: one-step-ahead (each forecast uses actual values up to t−1).
    Linear trend: least-squares line fitted on [trend_start, train_end] and extrapolated (multi-step).
    MASE is scaled by the naïve error over the training period.
    """
    s = series.sort_index()
    years = s.index.to_numpy()
    y = s.to_numpy(float)
    test = years > train_end
    train_y = s.loc[:train_end]
    trend_start = trend_start or int(years[0])
    tr = s.loc[trend_start:train_end]
    fit = rg.simple_regression(tr.index.to_numpy(float), tr.to_numpy())
    trend_fc = fit["slope"] * years + fit["intercept"]
    forecasts = {
        "Average method": average_forecast(y),
        f"Moving average (k={k})": moving_average_forecast(y, k),
        "Naïve (last value)": naive_forecast(y),
        f"Linear trend ({trend_start}–{train_end})": trend_fc,
    }
    rows = [{"method": name, **evaluate(y[test], f[test], train_y)} for name, f in forecasts.items()]
    table = pd.DataFrame(rows).set_index("method")
    detail = pd.DataFrame({"year": years[test], "actual": y[test], **{n: f[test] for n, f in forecasts.items()}})
    return {"table": table, "detail": detail, "trend_fit": {"slope": fit["slope"], "intercept": fit["intercept"],
            "r2": fit["metrics"]["R2"], "n": fit["n"]}, "forecasts": forecasts, "years": years}
