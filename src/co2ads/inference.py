"""Module 3 — inferential statistics (normal & Poisson, CLT, CI, z/t tests, ANOVA).

Test statistics are computed with the course formulas; scipy is used only for critical
values / p-values (the "statistical tables") and, in tests, as an independent cross-check.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from . import descriptive as ds


def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


# ---------------------------------------------------------------- distributions

def normal_pdf(x, mu: float, sigma: float):
    """f(x) = 1/√(2πσ²) · e^(−(x−μ)²/2σ²)."""
    x = np.asarray(x, float)
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / math.sqrt(2 * math.pi * sigma ** 2)


def z_scores(x) -> np.ndarray:
    a = _arr(x)
    return (a - a.mean()) / a.std()


def empirical_rule(x) -> pd.DataFrame:
    """Observed share within μ ± kσ vs the normal 68.27 / 95.45 / 99.73 %."""
    a = _arr(x)
    mu, s = a.mean(), a.std()
    rows = []
    for k in (1, 2, 3):
        obs = ((a >= mu - k * s) & (a <= mu + k * s)).mean() * 100
        exp = (stats.norm.cdf(k) - stats.norm.cdf(-k)) * 100
        rows.append({"k_sigma": k, "observed_pct": obs, "normal_pct": exp, "difference": obs - exp})
    return pd.DataFrame(rows)


def poisson_pmf(k, lam: float):
    """P(X = x) = e^(−μ) μ^x / x!"""
    k = np.asarray(k)
    return np.exp(-lam) * lam ** k / np.vectorize(math.factorial)(k)


def poisson_check(counts) -> dict:
    """Poisson requires mean = variance. Dispersion index = variance/mean (≈1 if Poisson)."""
    c = _arr(counts)
    m, v = c.mean(), c.var(ddof=1)
    n = c.size
    chi2 = (n - 1) * v / m                      # index-of-dispersion test
    p = 2 * min(stats.chi2.cdf(chi2, n - 1), stats.chi2.sf(chi2, n - 1))
    return {"n": int(n), "mean": float(m), "variance": float(v), "dispersion_index": float(v / m),
            "chi2": float(chi2), "p_value": float(p), "poisson_plausible": bool(p >= 0.05)}


# ---------------------------------------------------------------- CLT

def clt_simulation(population, n: int, reps: int, seed: int) -> dict:
    """Draw `reps` random samples of size n (with replacement) and record their means."""
    pop = _arr(population)
    rng = np.random.default_rng(seed)
    means = rng.choice(pop, size=(reps, n), replace=True).mean(axis=1)
    return {
        "means": means,
        "population_mean": float(pop.mean()),
        "population_sd": float(pop.std()),
        "population_skew": float(ds.pearson_skewness_median(pop)),
        "mean_of_means": float(means.mean()),
        "sd_of_means": float(means.std()),
        "theoretical_se": float(pop.std() / math.sqrt(n)),
        "skew_of_means": float(ds.pearson_skewness_median(means)),
    }


# ---------------------------------------------------------------- confidence intervals

def ci_mean(x, conf: float = 0.95, sigma: float | None = None) -> dict:
    """z interval (n ≥ 30 or σ known), otherwise t interval with n − 1 d.f. (course rule)."""
    a = _arr(x)
    n, xbar = a.size, a.mean()
    s = a.std(ddof=1) if sigma is None else sigma
    if sigma is not None or n >= 30:
        crit, method = stats.norm.ppf(1 - (1 - conf) / 2), "z"
    else:
        crit, method = stats.t.ppf(1 - (1 - conf) / 2, n - 1), "t"
    se = s / math.sqrt(n)
    return {"n": int(n), "mean": float(xbar), "sd": float(s), "se": float(se), "critical": float(crit),
            "method": method, "margin": float(crit * se), "lower": float(xbar - crit * se),
            "upper": float(xbar + crit * se), "confidence": conf}


# ---------------------------------------------------------------- hypothesis tests

def _decision(stat: float, crit: float, tail: str) -> str:
    if tail == "two":
        reject = abs(stat) > crit
    elif tail == "right":
        reject = stat > crit
    else:
        reject = stat < -crit
    return "Reject H0" if reject else "Fail to reject H0"


def _p(stat: float, dist, tail: str) -> float:
    if tail == "two":
        return float(2 * dist.sf(abs(stat)))
    return float(dist.sf(stat) if tail == "right" else dist.cdf(stat))


def z_test_mean(x, mu0: float, sigma: float | None = None, alpha: float = 0.05, tail: str = "two") -> dict:
    """Z = (x̄ − μ0)/(σ/√n); s replaces σ when unknown (large sample)."""
    a = _arr(x)
    s = a.std(ddof=1) if sigma is None else sigma
    z = (a.mean() - mu0) / (s / math.sqrt(a.size))
    crit = stats.norm.ppf(1 - alpha / 2) if tail == "two" else stats.norm.ppf(1 - alpha)
    return {"test": "z, single mean", "n": int(a.size), "mean": float(a.mean()), "mu0": mu0, "sd": float(s),
            "statistic": float(z), "critical": float(crit), "p_value": _p(z, stats.norm, tail),
            "alpha": alpha, "tail": tail, "decision": _decision(z, crit, tail)}


def t_test_one_sample(x, mu0: float, alpha: float = 0.05, tail: str = "two") -> dict:
    """t = (x̄ − μ)/(s/√n), d.f. = n − 1."""
    a = _arr(x)
    n, s = a.size, a.std(ddof=1)
    t = (a.mean() - mu0) / (s / math.sqrt(n))
    df = n - 1
    crit = stats.t.ppf(1 - alpha / 2, df) if tail == "two" else stats.t.ppf(1 - alpha, df)
    return {"test": "one-sample t", "n": int(n), "df": int(df), "mean": float(a.mean()), "mu0": mu0,
            "sd": float(s), "statistic": float(t), "critical": float(crit),
            "p_value": _p(t, stats.t(df), tail), "alpha": alpha, "tail": tail, "decision": _decision(t, crit, tail)}


def t_test_independent(x, y, alpha: float = 0.05, tail: str = "two") -> dict:
    """Pooled-variance t (course slides): s² = [Σ(x−x̄)² + Σ(y−ȳ)²]/(n1+n2−2)."""
    a, b = _arr(x), _arr(y)
    n1, n2 = a.size, b.size
    s2 = (((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()) / (n1 + n2 - 2)
    t = (a.mean() - b.mean()) / math.sqrt(s2 / n1 + s2 / n2)
    df = n1 + n2 - 2
    crit = stats.t.ppf(1 - alpha / 2, df) if tail == "two" else stats.t.ppf(1 - alpha, df)
    var_ratio = max(a.var(ddof=1), b.var(ddof=1)) / min(a.var(ddof=1), b.var(ddof=1))
    return {"test": "independent t (pooled)", "n1": int(n1), "n2": int(n2), "df": int(df),
            "mean1": float(a.mean()), "mean2": float(b.mean()), "pooled_var": float(s2),
            "statistic": float(t), "critical": float(crit), "p_value": _p(t, stats.t(df), tail),
            "variance_ratio": float(var_ratio), "alpha": alpha, "tail": tail, "decision": _decision(t, crit, tail)}


def t_test_paired(before, after, alpha: float = 0.05, tail: str = "two") -> dict:
    """t = (d̄ − 0)/(s_d/√n) with d = after − before, d.f. = n − 1."""
    b, a = np.asarray(before, float), np.asarray(after, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    n = d.size
    sd = d.std(ddof=1)
    t = d.mean() / (sd / math.sqrt(n))
    df = n - 1
    crit = stats.t.ppf(1 - alpha / 2, df) if tail == "two" else stats.t.ppf(1 - alpha, df)
    return {"test": "paired t", "n": int(n), "df": int(df), "mean_diff": float(d.mean()), "sd_diff": float(sd),
            "statistic": float(t), "critical": float(crit), "p_value": _p(t, stats.t(df), tail),
            "n_decreased": int((d < 0).sum()), "n_increased": int((d > 0).sum()),
            "alpha": alpha, "tail": tail, "decision": _decision(t, crit, tail)}


def error_rate_simulation(population, n: int, shift: float, reps: int, seed: int, alpha: float = 0.05) -> dict:
    """Type I / Type II errors, estimated by repeatedly sampling from the real data.

    Type I: test H0: μ = true mean (H0 is true)   → rejection rate ≈ α.
    Type II: test H0: μ = true mean + shift (H0 false) → acceptance rate = β.
    """
    pop = _arr(population)
    mu = pop.mean()
    rng = np.random.default_rng(seed)
    samples = rng.choice(pop, size=(reps, n), replace=True)
    xbar, s = samples.mean(axis=1), samples.std(axis=1, ddof=1)
    crit = stats.norm.ppf(1 - alpha / 2)
    z_true = (xbar - mu) / (s / math.sqrt(n))
    z_false = (xbar - (mu + shift)) / (s / math.sqrt(n))
    type1 = float((np.abs(z_true) > crit).mean())
    beta = float((np.abs(z_false) <= crit).mean())
    return {"n": n, "reps": reps, "alpha": alpha, "shift": shift, "true_mean": float(mu),
            "type1_rate": type1, "type2_rate": beta, "power": 1 - beta}


# ---------------------------------------------------------------- ANOVA

def one_way_anova(groups: dict[str, np.ndarray], alpha: float = 0.05) -> dict:
    """One-way ANOVA following the course steps.

    The grand mean is the mean of ALL observations (Σ n_j X̄_j / n). The slides' "mean of
    sample means" equals this only when every group has the same size, which is not the
    case for continents, so the weighted form is used (both are reported).
    """
    g = {k: _arr(v) for k, v in groups.items() if len(_arr(v)) > 0}
    allv = np.concatenate(list(g.values()))
    n, k = allv.size, len(g)
    grand = allv.mean()
    unweighted = float(np.mean([v.mean() for v in g.values()]))
    ss_between = sum(v.size * (v.mean() - grand) ** 2 for v in g.values())
    ss_within = sum(((v - v.mean()) ** 2).sum() for v in g.values())
    ss_total = ((allv - grand) ** 2).sum()
    df_b, df_w = k - 1, n - k
    ms_b, ms_w = ss_between / df_b, ss_within / df_w
    f = ms_b / ms_w
    crit = stats.f.ppf(1 - alpha, df_b, df_w)
    table = pd.DataFrame({
        "SS": [ss_between, ss_within, ss_total], "df": [df_b, df_w, n - 1],
        "MS": [ms_b, ms_w, np.nan], "F": [f, np.nan, np.nan],
    }, index=["Between groups", "Within groups", "Total"])
    group_stats = pd.DataFrame({k_: {"n": v.size, "mean": v.mean(), "sd": v.std(ddof=1)} for k_, v in g.items()}).T
    return {"table": table, "groups": group_stats, "grand_mean": float(grand), "mean_of_means": unweighted,
            "F": float(f), "critical": float(crit), "p_value": float(stats.f.sf(f, df_b, df_w)),
            "df_between": df_b, "df_within": df_w, "ss_check": float(ss_total - ss_between - ss_within),
            "eta_squared": float(ss_between / ss_total), "decision": "Reject H0" if f >= crit else "Fail to reject H0"}


def two_way_anova(matrix: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Two-way ANOVA without replication (one value per cell), as on the course slides.

    Rows = factor A, columns = factor B. Correction factor = T²/n.
    """
    x = matrix.to_numpy(dtype=float)
    if np.isnan(x).any():
        raise ValueError("two-way ANOVA without replication needs a complete table")
    r, c = x.shape
    n = r * c
    t = x.sum()
    cf = t ** 2 / n
    ss_total = (x ** 2).sum() - cf
    ss_cols = (x.sum(axis=0) ** 2 / r).sum() - cf
    ss_rows = (x.sum(axis=1) ** 2 / c).sum() - cf
    ss_res = ss_total - ss_cols - ss_rows
    df_c, df_r, df_e = c - 1, r - 1, (c - 1) * (r - 1)
    ms_c, ms_r, ms_e = ss_cols / df_c, ss_rows / df_r, ss_res / df_e
    f_c, f_r = ms_c / ms_e, ms_r / ms_e
    crit_c, crit_r = stats.f.ppf(1 - alpha, df_c, df_e), stats.f.ppf(1 - alpha, df_r, df_e)
    table = pd.DataFrame({
        "SS": [ss_cols, ss_rows, ss_res, ss_total], "df": [df_c, df_r, df_e, n - 1],
        "MS": [ms_c, ms_r, ms_e, np.nan], "F": [f_c, f_r, np.nan, np.nan],
        "F_crit_5pct": [crit_c, crit_r, np.nan, np.nan],
    }, index=[f"Between columns ({matrix.columns.name or 'B'})", f"Between rows ({matrix.index.name or 'A'})",
              "Residual", "Total"])
    return {"table": table, "F_columns": float(f_c), "F_rows": float(f_r),
            "p_columns": float(stats.f.sf(f_c, df_c, df_e)), "p_rows": float(stats.f.sf(f_r, df_r, df_e)),
            "crit_columns": float(crit_c), "crit_rows": float(crit_r),
            "decision_columns": "Reject H0" if f_c >= crit_c else "Fail to reject H0",
            "decision_rows": "Reject H0" if f_r >= crit_r else "Fail to reject H0"}
