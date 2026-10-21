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


def class_table(lower, upper, f) -> pd.DataFrame:
    """Frequency table from given class limits (exclusive form) and frequencies."""
    lower, upper, f = (np.asarray(v, float) for v in (lower, upper, f))
    t = pd.DataFrame({"lower": lower, "upper": upper, "mid": (lower + upper) / 2, "f": f.astype(int)})
    t["cf"] = t["f"].cumsum()
    t["rel_f_pct"] = t["f"] / t["f"].sum() * 100
    t.attrs["width"] = float(upper[0] - lower[0])
    return t


def inclusive_to_exclusive(lower, upper) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive classes (e.g. 0–4, 5–9) → exclusive by subtracting half the gap from lower
    limits and adding it to upper limits (gap = next lower − this upper, usually 1 → ±0.5)."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    half = (lower[1] - upper[0]) / 2
    return lower - half, upper + half


def classes_from_midpoints(mid) -> tuple[np.ndarray, np.ndarray]:
    """Mid-values → class limits: i = difference of mid-values, limits = m ∓ i/2."""
    mid = np.asarray(mid, float)
    i = mid[1] - mid[0]
    return mid - i / 2, mid + i / 2


def cumulative_frequencies(t: pd.DataFrame) -> pd.DataFrame:
    """'Less than' (cf up to each upper limit) and 'more than' (cf from each lower limit) tables."""
    n = t["f"].sum()
    return pd.DataFrame({
        "upper": t["upper"], "less_than_cf": t["f"].cumsum(),
        "lower": t["lower"], "more_than_cf": n - t["f"].cumsum() + t["f"],
    })


def ogive_value(t: pd.DataFrame, target: float) -> float:
    """Read the less-than ogive at cumulative frequency `target` (linear between class limits).

    At target = N/2 this is the graphical median; it equals the interpolation formula
    L + (N/2 − cf)/f × i, which is why both methods give the same answer.
    """
    xs = np.r_[t["lower"].iloc[0], t["upper"].to_numpy()]
    ys = np.r_[0, t["f"].cumsum().to_numpy()]
    return float(np.interp(target, ys, xs))


def ogive_intersection(t: pd.DataFrame) -> tuple[float, float]:
    """Point where less-than and more-than ogives cross: x = median, y = N/2."""
    n = t["f"].sum()
    return ogive_value(t, n / 2), n / 2


def open_end_mean(t: pd.DataFrame, open_first: bool = False, open_last: bool = True) -> float:
    """Mean when the first/last class is open-ended ('below L', 'L and above').

    Course rule: assume the open class has the same width as its neighbour, then Σfm/N.
    Unreliable when class widths vary; median and quartiles do not need the open limit.
    """
    t = t.copy()
    if open_first:
        w = t["upper"].iloc[1] - t["lower"].iloc[1]
        t.loc[t.index[0], "lower"] = t["upper"].iloc[0] - w
    if open_last:
        w = t["upper"].iloc[-2] - t["lower"].iloc[-2]
        t.loc[t.index[-1], "upper"] = t["lower"].iloc[-1] + w
    mid = (t["lower"] + t["upper"]) / 2
    return float((t["f"] * mid).sum() / t["f"].sum())


def close_top_classes(t: pd.DataFrame, open_from: float) -> pd.DataFrame:
    """Merge all classes with lower limit ≥ open_from into one open-ended class 'open_from and above'."""
    keep = t[t["lower"] < open_from]
    top = t[t["lower"] >= open_from]
    row = pd.DataFrame({"lower": [open_from], "upper": [np.inf], "mid": [np.nan], "f": [int(top["f"].sum())]})
    out = pd.concat([keep[["lower", "upper", "mid", "f"]], row], ignore_index=True)
    out["cf"] = out["f"].cumsum()
    out["rel_f_pct"] = out["f"] / out["f"].sum() * 100
    out.attrs["width"] = t.attrs.get("width", _width(t))
    return out


def corrected_mean(n: int, mean: float, wrong=(), right=()) -> dict:
    """Correct a mean computed with wrong items: ΣX = N·X̄ − Σwrong + Σright.

    Replacing items keeps N; removing items (right empty) or adding items changes N.
    """
    wrong, right = list(wrong), list(right)
    total = n * mean - sum(wrong) + sum(right)
    n_new = n - len(wrong) + len(right)
    return {"incorrect_sum": n * mean, "correct_sum": total, "n": n_new, "correct_mean": total / n_new}


def empirical_mode(x) -> float:
    """Karl Pearson's empirical relation for moderately skewed data: Mode ≈ 3·Median − 2·Mean."""
    return 3 * median(x) - 2 * mean_direct(x)


# ---------------------------------------------------------------- discrete series

def discrete_table(x) -> pd.DataFrame:
    """Discrete frequency distribution: each distinct value X with f and cf (ascending X)."""
    a = _arr(x)
    vals, f = np.unique(a, return_counts=True)
    t = pd.DataFrame({"X": vals, "f": f})
    t["cf"] = t["f"].cumsum()
    t["rel_f_pct"] = t["f"] / t["f"].sum() * 100
    return t


def discrete_from_counts(values, freqs) -> pd.DataFrame:
    t = pd.DataFrame({"X": np.asarray(values, float), "f": np.asarray(freqs, int)}).sort_values("X", ignore_index=True)
    t["cf"] = t["f"].cumsum()
    t["rel_f_pct"] = t["f"] / t["f"].sum() * 100
    return t


def discrete_mean(t: pd.DataFrame, assumed: float | None = None) -> dict:
    """Direct ΣfX/N and short-cut A + Σfd/N (d = X − A)."""
    n = t["f"].sum()
    direct = (t["f"] * t["X"]).sum() / n
    if assumed is None:
        assumed = float(t.loc[t["f"].idxmax(), "X"])
    shortcut = assumed + (t["f"] * (t["X"] - assumed)).sum() / n
    return {"direct": float(direct), "shortcut": float(shortcut), "assumed_mean": assumed}


def discrete_quantile(t: pd.DataFrame, k: int, parts: int) -> float:
    """Size of the k(N+1)/parts-th item: the X whose cf first equals or exceeds that position."""
    n = t["f"].sum()
    pos = k * (n + 1) / parts
    row = int(np.argmax(t["cf"].to_numpy() >= min(pos, n)))
    return float(t["X"].iloc[row])


def discrete_median(t: pd.DataFrame) -> float:
    return discrete_quantile(t, 1, 2)


def discrete_mode(t: pd.DataFrame) -> list[float]:
    """Value(s) with the highest frequency (more than one = multimodal)."""
    return t.loc[t["f"] == t["f"].max(), "X"].astype(float).tolist()


def discrete_std(t: pd.DataFrame, assumed: float | None = None) -> dict:
    """Actual-mean σ = √(Σfx²/N) and assumed-mean σ = √(Σfd²/N − (Σfd/N)²); both must agree."""
    n = t["f"].sum()
    m = (t["f"] * t["X"]).sum() / n
    actual = math.sqrt((t["f"] * (t["X"] - m) ** 2).sum() / n)
    if assumed is None:
        assumed = float(t.loc[t["f"].idxmax(), "X"])
    d = t["X"] - assumed
    am = math.sqrt((t["f"] * d ** 2).sum() / n - ((t["f"] * d).sum() / n) ** 2)
    return {"actual_mean_method": actual, "assumed_mean_method": am}


def discrete_summary(t: pd.DataFrame) -> dict:
    m = discrete_mean(t)
    q1, med, q3 = discrete_quantile(t, 1, 4), discrete_median(t), discrete_quantile(t, 3, 4)
    sd = discrete_std(t)["actual_mean_method"]
    return {"n": int(t["f"].sum()), "mean": m["direct"], "mean_shortcut": m["shortcut"], "median": med,
            "mode": discrete_mode(t), "q1": q1, "q3": q3, "d1": discrete_quantile(t, 1, 10),
            "d9": discrete_quantile(t, 9, 10), "range": float(t["X"].max() - t["X"].min()),
            "iqr": q3 - q1, "quartile_deviation": (q3 - q1) / 2, "std": sd,
            "coef_variation_pct": sd / m["direct"] * 100 if m["direct"] else float("nan")}


# ---------------------------------------------------------------- moments

def raw_moment(x, r: int, about: float = 0.0) -> float:
    """μ'_r = Σ(X − A)^r / N, moments about an arbitrary origin A (A = 0: about the origin)."""
    a = _arr(x)
    return float(((a - about) ** r).sum() / a.size)


def moments(x) -> dict:
    """First four moments about the origin and about the mean, β1, β2 and γ1, γ2.

    Central moments are also derived from raw moments with the standard conversion
    (μ2 = μ'2 − μ'1², μ3 = μ'3 − 3μ'2μ'1 + 2μ'1³, μ4 = μ'4 − 4μ'3μ'1 + 6μ'2μ'1² − 3μ'1⁴)
    so the two routes can be checked against each other.
    """
    r1, r2, r3, r4 = (raw_moment(x, r) for r in (1, 2, 3, 4))
    m2, m3, m4 = (central_moment(x, r) for r in (2, 3, 4))
    conv = {"mu2": r2 - r1 ** 2, "mu3": r3 - 3 * r2 * r1 + 2 * r1 ** 3,
            "mu4": r4 - 4 * r3 * r1 + 6 * r2 * r1 ** 2 - 3 * r1 ** 4}
    b1 = m3 ** 2 / m2 ** 3
    b2 = m4 / m2 ** 2
    return {"raw": {"mu1'": r1, "mu2'": r2, "mu3'": r3, "mu4'": r4},
            "central": {"mu1": central_moment(x, 1), "mu2": m2, "mu3": m3, "mu4": m4},
            "central_from_raw": conv,
            "beta1": b1, "beta2": b2, "gamma1": math.copysign(math.sqrt(b1), m3), "gamma2": b2 - 3}


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
