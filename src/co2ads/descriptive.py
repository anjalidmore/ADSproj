"""Module 2 — univariate exploration, implemented exactly as on the course formula sheet.

Individual series: quantile position = k(N+1)/parts (the course rule; equals numpy's
"weibull" method). Grouped (continuous) series: L + (kN/parts - cf)/f * i.
Standard deviation on the formula sheet divides by N (population SD).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        raise ValueError("empty series")
    return a


# ---------------------------------------------------------------- individual series

def mean_direct(x) -> float:
    a = _arr(x)
    return a.sum() / a.size                      # X̄ = ΣX / N


def mean_shortcut(x, assumed: float) -> float:
    a = _arr(x)
    return assumed + (a - assumed).sum() / a.size  # X̄ = A + Σd / N


def weighted_mean(x, w) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = ~(np.isnan(x) | np.isnan(w))
    return (w[ok] * x[ok]).sum() / w[ok].sum()     # X̄w = ΣWX / ΣW


def combined_mean(means, sizes) -> float:
    means, sizes = np.asarray(means, float), np.asarray(sizes, float)
    return (sizes * means).sum() / sizes.sum()     # X̄12 = (N1X̄1 + N2X̄2)/(N1+N2)


def position_value(x, position: float) -> float:
    """Value at a 1-based (possibly fractional) rank position in the ordered series."""
    a = np.sort(_arr(x))
    n = a.size
    if position <= 1:
        return float(a[0])
    if position >= n:
        return float(a[-1])
    lo = math.floor(position)
    frac = position - lo
    return float(a[lo - 1] + frac * (a[lo] - a[lo - 1]))


def median(x) -> float:
    return position_value(x, (_arr(x).size + 1) / 2)   # (N+1)/2 th item


def quantile(x, k: int, parts: int) -> float:
    """Quartile (parts=4), decile (10) or percentile (100): k(N+1)/parts th item."""
    return position_value(x, k * (_arr(x).size + 1) / parts)


def quartiles(x) -> tuple[float, float, float]:
    return quantile(x, 1, 4), quantile(x, 2, 4), quantile(x, 3, 4)


def value_range(x) -> float:
    a = _arr(x)
    return float(a.max() - a.min())                # L − S


def coefficient_of_range(x) -> float:
    a = _arr(x)
    return float((a.max() - a.min()) / (a.max() + a.min()))


def iqr(x) -> float:
    q1, _, q3 = quartiles(x)
    return q3 - q1


def quartile_deviation(x) -> float:
    return iqr(x) / 2                              # (Q3 − Q1)/2


def coefficient_of_qd(x) -> float:
    q1, _, q3 = quartiles(x)
    return (q3 - q1) / (q3 + q1)


def variance(x, ddof: int = 0) -> float:
    a = _arr(x)
    return float(((a - a.mean()) ** 2).sum() / (a.size - ddof))


def std(x, ddof: int = 0) -> float:
    """σ = sqrt(Σx²/N) with x = X − X̄ (ddof=0, formula sheet); ddof=1 gives sample s."""
    return math.sqrt(variance(x, ddof))


def std_assumed_mean(x, assumed: float) -> float:
    a = _arr(x)
    d = a - assumed
    return math.sqrt((d ** 2).sum() / a.size - (d.sum() / a.size) ** 2)


def central_moment(x, r: int) -> float:
    a = _arr(x)
    return float(((a - a.mean()) ** r).sum() / a.size)


def kurtosis_beta2(x) -> float:
    return central_moment(x, 4) / central_moment(x, 2) ** 2   # β2 = μ4 / μ2²


def kurtosis_label(beta2: float, tol: float = 0.05) -> str:
    # Note: the course formula sheet's shape descriptions are swapped; this follows the standard meaning.
    if abs(beta2 - 3) <= tol:
        return "Mesokurtic (same peakedness as normal)"
    if beta2 > 3:
        return "Leptokurtic (more peaked, heavier tails than normal)"
    return "Platykurtic (flatter than normal)"


def bowley_skewness(x) -> float:
    q1, med, q3 = quartiles(x)
    return (q3 + q1 - 2 * med) / (q3 - q1)


def pearson_skewness_median(x) -> float:
    """Sk = 3(X̄ − Median)/σ, used when the mode is ill-defined."""
    return 3 * (mean_direct(x) - median(x)) / std(x)


def pearson_skewness_mode(x, mode_value: float) -> float:
    """Sk = (X̄ − Mode)/σ."""
    return (mean_direct(x) - mode_value) / std(x)


def skew_direction(sk: float, tol: float = 0.05) -> str:
    if abs(sk) <= tol:
        return "approximately symmetric"
    return "positively skewed (Mode < Median < Mean)" if sk > 0 else "negatively skewed (Mean < Median < Mode)"


def box_plot(x) -> dict:
    a = _arr(x)
    q1, med, q3 = quartiles(a)
    i = q3 - q1
    lo, hi = q1 - 1.5 * i, q3 + 1.5 * i
    return {
        "min": float(a.min()), "q1": q1, "median": med, "q3": q3, "max": float(a.max()),
        "iqr": i, "lower_limit": lo, "upper_limit": hi,
        "n_low_outliers": int((a < lo).sum()), "n_high_outliers": int((a > hi).sum()),
        "whisker_low": float(a[a >= lo].min()), "whisker_high": float(a[a <= hi].max()),
    }


def iqr_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    b = box_plot(df[col])
    mask = (df[col] < b["lower_limit"]) | (df[col] > b["upper_limit"])
    return df.loc[mask].sort_values(col, ascending=False)


# ---------------------------------------------------------------- grouped (continuous) series

def _nice_width(raw: float) -> float:
    p = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 3, 4, 5, 10):
        if m * p >= raw:
            return m * p
    return 10 * p


def frequency_table(x, width: float | None = None, start: float | None = None) -> pd.DataFrame:
    """Exclusive class intervals [L, U). Default number of classes ≈ √n (course rule of thumb)."""
    a = _arr(x)
    if width is None:
        width = _nice_width((a.max() - a.min()) / math.ceil(math.sqrt(a.size)))
    if start is None:
        start = math.floor(a.min() / width) * width
    k = int(math.floor((a.max() - start) / width)) + 1
    idx = np.floor((a - start) / width).astype(int)
    f = np.bincount(idx, minlength=k)
    lower = start + width * np.arange(k)
    t = pd.DataFrame({"lower": lower, "upper": lower + width, "mid": lower + width / 2, "f": f})
    t["cf"] = t["f"].cumsum()
    t["rel_f_pct"] = t["f"] / t["f"].sum() * 100
    t.attrs["width"] = width
    return t


def _width(t: pd.DataFrame) -> float:
    return float(t["upper"].iloc[0] - t["lower"].iloc[0])


def grouped_mean(t: pd.DataFrame, assumed: float | None = None) -> dict:
    """Direct Σfm/N and step-deviation A + (Σfd'/N)·i — both must agree."""
    i, n = _width(t), t["f"].sum()
    direct = (t["f"] * t["mid"]).sum() / n
    if assumed is None:
        assumed = float(t.loc[t["f"].idxmax(), "mid"])
    d = (t["mid"] - assumed) / i
    step = assumed + (t["f"] * d).sum() / n * i
    return {"direct": float(direct), "step_deviation": float(step), "assumed_mean": assumed}


def grouped_std(t: pd.DataFrame, assumed: float | None = None) -> float:
    """σ = sqrt(Σfd'²/N − (Σfd'/N)²) × i (step-deviation method)."""
    i, n = _width(t), t["f"].sum()
    if assumed is None:
        assumed = float(t.loc[t["f"].idxmax(), "mid"])
    d = (t["mid"] - assumed) / i
    return float(math.sqrt((t["f"] * d ** 2).sum() / n - ((t["f"] * d).sum() / n) ** 2) * i)


def grouped_quantile(t: pd.DataFrame, k: int, parts: int) -> float:
    """Qk = L + (kN/parts − cf)/f × i."""
    i, n = _width(t), t["f"].sum()
    target = k * n / parts
    row = int(np.argmax(t["cf"].to_numpy() >= target))
    cf_prev = t["cf"].iloc[row - 1] if row > 0 else 0
    return float(t["lower"].iloc[row] + (target - cf_prev) / t["f"].iloc[row] * i)


def grouped_median(t: pd.DataFrame) -> float:
    return grouped_quantile(t, 1, 2)                # L + [(N/2 − c.f.)/f] × i


def grouped_mode(t: pd.DataFrame) -> dict:
    """Mode = L + Δ1/(Δ1+Δ2) × i, Δ1=|f1−f0|, Δ2=|f1−f2|."""
    i = _width(t)
    f = t["f"].to_numpy()
    m = int(np.argmax(f))
    f0 = f[m - 1] if m > 0 else 0
    f2 = f[m + 1] if m + 1 < len(f) else 0
    d1, d2 = abs(f[m] - f0), abs(f[m] - f2)
    mode = t["lower"].iloc[m] + (d1 / (d1 + d2) if d1 + d2 else 0.5) * i
    return {"mode": float(mode), "modal_class": (float(t["lower"].iloc[m]), float(t["upper"].iloc[m])),
            "f1": int(f[m]), "f0": int(f0), "f2": int(f2), "delta1": int(d1), "delta2": int(d2)}


# ---------------------------------------------------------------- one-call summary

def summary(x) -> dict:
    a = _arr(x)
    q1, med, q3 = quartiles(a)
    t = frequency_table(a)
    gm = grouped_mode(t)
    sigma = std(a)
    b2 = kurtosis_beta2(a)
    return {
        "n": int(a.size),
        "mean": mean_direct(a),
        "median": med,
        "grouped_mode": gm["mode"],
        "min": float(a.min()), "max": float(a.max()),
        "range": value_range(a), "coef_range": coefficient_of_range(a),
        "q1": q1, "q3": q3, "iqr": q3 - q1,
        "quartile_deviation": (q3 - q1) / 2, "coef_qd": (q3 - q1) / (q3 + q1),
        "d1": quantile(a, 1, 10), "d9": quantile(a, 9, 10),
        "p10": quantile(a, 10, 100), "p90": quantile(a, 90, 100),
        "variance": variance(a), "std": sigma, "std_sample": std(a, 1),
        "coef_variation_pct": sigma / a.mean() * 100,
        "pearson_sk_mode": pearson_skewness_mode(a, gm["mode"]),
        "pearson_sk_median": pearson_skewness_median(a),
        "bowley_sk": bowley_skewness(a),
        "beta2": b2, "kurtosis_label": kurtosis_label(b2),
        "classes": len(t), "class_width": _width(t),
    }
