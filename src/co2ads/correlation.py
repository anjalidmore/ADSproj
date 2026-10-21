"""Module 2.4 — multivariate exploration: central data point, covariance, Karl Pearson r."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        raise ValueError("need at least 3 complete pairs")
    return x[ok], y[ok]


def covariance(x, y) -> float:
    x, y = _pair(x, y)
    return float(((x - x.mean()) * (y - y.mean())).sum() / x.size)


def pearson_actual_mean(x, y) -> float:
    """Method 1: r = Σxy / √(Σx² · Σy²), deviations from actual means."""
    x, y = _pair(x, y)
    dx, dy = x - x.mean(), y - y.mean()
    return float((dx * dy).sum() / math.sqrt((dx ** 2).sum() * (dy ** 2).sum()))


def pearson_direct(x, y) -> float:
    """Method 2: r = [NΣXY − ΣXΣY] / √[NΣX² − (ΣX)²]·√[NΣY² − (ΣY)²]."""
    x, y = _pair(x, y)
    n = x.size
    num = n * (x * y).sum() - x.sum() * y.sum()
    den = math.sqrt(n * (x ** 2).sum() - x.sum() ** 2) * math.sqrt(n * (y ** 2).sum() - y.sum() ** 2)
    return float(num / den)


def pearson_assumed_mean(x, y, ax: float | None = None, ay: float | None = None) -> float:
    """Method 3: deviations from assumed means A_x, A_y (result is origin-independent)."""
    x, y = _pair(x, y)
    ax = float(np.round(np.median(x))) if ax is None else ax
    ay = float(np.round(np.median(y))) if ay is None else ay
    dx, dy = x - ax, y - ay
    n = x.size
    num = n * (dx * dy).sum() - dx.sum() * dy.sum()
    den = math.sqrt(n * (dx ** 2).sum() - dx.sum() ** 2) * math.sqrt(n * (dy ** 2).sum() - dy.sum() ** 2)
    return float(num / den)


def probable_error(r: float, n: int) -> float:
    """P.E. = 0.6745 (1 − r²)/√N; r is 'significant' if |r| > 6·P.E. (classical rule)."""
    return 0.6745 * (1 - r ** 2) / math.sqrt(n)


def degree(r: float) -> str:
    """Degree of correlation per the course table (0.5–0.75 labelled 'moderate' to fill the gap)."""
    a = abs(r)
    sign = "positive" if r > 0 else "negative"
    if a == 1:
        return f"Perfect {sign}"
    if a >= 0.75:
        return f"High degree {sign}"
    if a >= 0.5:
        return f"Moderate degree {sign}"
    if a > 0.05:
        return f"Low degree {sign}"
    return "No correlation"


def central_data_point(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """The 'average point' in multi-dimensional space: (mean of each variable)."""
    return df[cols].mean()


def distance_from_centre(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Euclidean distance of each observation from the central point, in standard-deviation units."""
    z = (df[cols] - df[cols].mean()) / df[cols].std(ddof=0)
    return np.sqrt((z ** 2).sum(axis=1))


def correlation_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pairwise Pearson r using complete pairs (n differs per pair; see pair_counts)."""
    return df[cols].corr(method="pearson")


def pair_counts(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    m = df[cols].notna().astype(int)
    return m.T @ m


def analyse(x, y) -> dict:
    x, y = _pair(x, y)
    r = pearson_actual_mean(x, y)
    pe = probable_error(r, x.size)
    return {
        "n": int(x.size), "r": r, "r_direct": pearson_direct(x, y), "r_assumed": pearson_assumed_mean(x, y),
        "covariance": covariance(x, y), "r_squared": r * r, "probable_error": pe,
        "significant_6pe": abs(r) > 6 * pe, "degree": degree(r),
    }
