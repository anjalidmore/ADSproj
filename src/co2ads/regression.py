"""Module 4 — regression (plus Module 1 model evaluation: hold-out, k-fold CV, bootstrapping).

Coefficients are computed with the course's summation formulas, not a library solver;
tests cross-check them against numpy.linalg.lstsq and statsmodels.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy import stats


def _clean(*cols):
    arrs = [np.asarray(c, float) for c in cols]
    ok = np.all([~np.isnan(a) for a in arrs], axis=0)
    return [a[ok] for a in arrs]


# ---------------------------------------------------------------- error metrics

def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def mse(y, yhat) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))


def rmse(y, yhat) -> float:
    return math.sqrt(mse(y, yhat))


def mape(y, yhat) -> float:
    """100/n Σ|y − ŷ|/|y|; rows with y = 0 are excluded (MAPE is undefined there)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    nz = y != 0
    return float(np.mean(np.abs((y[nz] - yhat[nz]) / y[nz])) * 100)


def r2(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def adjusted_r2(r2_value: float, n: int, k: int) -> float:
    return 1 - (1 - r2_value) * (n - 1) / (n - k - 1)


def all_metrics(y, yhat) -> dict:
    return {"MAE": mae(y, yhat), "MSE": mse(y, yhat), "RMSE": rmse(y, yhat), "MAPE": mape(y, yhat), "R2": r2(y, yhat)}


# ---------------------------------------------------------------- simple linear regression

def simple_regression(x, y, alpha: float = 0.05) -> dict:
    """y′ = a·x + b with a = (nΣxy − ΣxΣy)/(nΣx² − (Σx)²), b = (Σy − aΣx)/n.

    Also returns the slide 18–24 fit statistics with k = 1, so the simple and the two-predictor
    model can be compared metric for metric.
    """
    x, y = _clean(x, y)
    n, k = x.size, 1
    a = (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x ** 2).sum() - x.sum() ** 2)
    b = (y.sum() - a * x.sum()) / n
    yhat = a * x + b
    sxx = (x ** 2).sum() - x.sum() ** 2 / n
    sxy = (x * y).sum() - x.sum() * y.sum() / n
    syy = (y ** 2).sum() - y.sum() ** 2 / n
    ss_reg = a * sxy                                     # R²Σy² = aΣxy
    ss_err = syy - ss_reg
    df_err = n - k - 1
    r2_value = ss_reg / syy
    f = (df_err * r2_value) / (k * (1 - r2_value))
    f_crit = stats.f.ppf(1 - alpha, k, df_err)
    se_a = math.sqrt(ss_err / df_err / sxx)
    t = a / se_a
    crit = stats.t.ppf(1 - alpha / 2, df_err)
    return {"n": int(n), "slope": float(a), "intercept": float(b), "se_slope": float(se_a), "t_slope": float(t),
            "t_critical": float(crit), "p_slope": float(2 * stats.t.sf(abs(t), df_err)),
            "slope_significant": bool(abs(t) > crit),
            "sums": {"Σx²": sxx, "Σxy": sxy, "Σy²": syy}, "r": float(sxy / math.sqrt(sxx * syy)),
            "r2": float(r2_value), "adj_r2": float(adjusted_r2(r2_value, n, k)),
            "anova": {"ss_regression": float(ss_reg), "ss_error": float(ss_err), "ss_total": float(syy),
                      "df_regression": k, "df_error": df_err, "ms_regression": float(ss_reg / k),
                      "ms_error": float(ss_err / df_err)},
            "F": float(f), "F_critical": float(f_crit), "p_F": float(stats.f.sf(f, k, df_err)),
            "F_decision": "Reject H0 (model significant)" if f >= f_crit else "Fail to reject H0",
            "metrics": all_metrics(y, yhat), "fitted": yhat, "residuals": y - yhat}


# ---------------------------------------------------------------- multiple regression (2 predictors)

def multiple_regression_2(x1, x2, y, alpha: float = 0.05) -> dict:
    """Y′ = b0 + b1X1 + b2X2 via the deviation-score formulas on the course slides."""
    x1, x2, y = _clean(x1, x2, y)
    n, k = y.size, 2
    sx1x1 = (x1 ** 2).sum() - x1.sum() ** 2 / n
    sx2x2 = (x2 ** 2).sum() - x2.sum() ** 2 / n
    sx1y = (x1 * y).sum() - x1.sum() * y.sum() / n
    sx2y = (x2 * y).sum() - x2.sum() * y.sum() / n
    sx1x2 = (x1 * x2).sum() - x1.sum() * x2.sum() / n
    syy = (y ** 2).sum() - y.sum() ** 2 / n
    den = sx1x1 * sx2x2 - sx1x2 ** 2
    b1 = (sx2x2 * sx1y - sx1x2 * sx2y) / den
    b2 = (sx1x1 * sx2y - sx1x2 * sx1y) / den
    b0 = y.mean() - b1 * x1.mean() - b2 * x2.mean()

    ss_reg = b1 * sx1y + b2 * sx2y                       # R²Σy² = b1Σx1y + b2Σx2y
    r2_value = ss_reg / syy
    ss_err = syy - ss_reg
    df_err = n - k - 1
    f = (df_err * r2_value) / (k * (1 - r2_value))
    f_crit = stats.f.ppf(1 - alpha, k, df_err)

    r12 = sx1x2 / math.sqrt(sx1x1 * sx2x2)
    mse_err = ss_err / df_err
    se_b1 = math.sqrt(mse_err / (sx1x1 * (1 - r12 ** 2)))
    se_b2 = math.sqrt(mse_err / (sx2x2 * (1 - r12 ** 2)))
    t_crit = stats.t.ppf(1 - alpha / 2, df_err)
    yhat = b0 + b1 * x1 + b2 * x2
    coefs = []
    for name, b, se in (("b1", b1, se_b1), ("b2", b2, se_b2)):
        t = b / se
        coefs.append({"term": name, "coef": float(b), "se": float(se), "t": float(t),
                      "p_value": float(2 * stats.t.sf(abs(t), df_err)),
                      "significant": bool(abs(t) > t_crit)})
    return {
        "n": int(n), "b0": float(b0), "b1": float(b1), "b2": float(b2),
        "sums": {"Σx1²": sx1x1, "Σx2²": sx2x2, "Σx1y": sx1y, "Σx2y": sx2y, "Σx1x2": sx1x2, "Σy²": syy},
        "r2": float(r2_value), "adj_r2": float(adjusted_r2(r2_value, n, k)),
        "anova": {"ss_regression": float(ss_reg), "ss_error": float(ss_err), "ss_total": float(syy),
                  "df_regression": k, "df_error": df_err, "ms_regression": float(ss_reg / k),
                  "ms_error": float(mse_err)},
        "F": float(f), "F_critical": float(f_crit), "p_F": float(stats.f.sf(f, k, df_err)),
        "F_decision": "Reject H0 (model significant)" if f >= f_crit else "Fail to reject H0",
        "R": float(math.sqrt(r2_value)), "r12": float(r12), "collinearity": collinearity(r12),
        "t_critical": float(t_crit), "coefficients": coefs,
        "metrics": all_metrics(y, yhat), "fitted": yhat, "residuals": y - yhat,
    }


def collinearity(r12: float, threshold: float = 0.8) -> dict:
    """Overlap between X1 and X2, read off the (1 − r12²) term in the slide 39 SE(b_i) formula.

    Tolerance = 1 − r12² is the share of a predictor NOT explained by the other one;
    VIF = 1/tolerance is the factor by which that term inflates Var(b_i) (√VIF inflates SE).
    With two predictors both predictors share the same VIF. Rules of thumb: |r12| ≥ 0.8 flags
    multicollinearity; VIF ≥ 5 is high, VIF ≥ 10 severe.
    """
    tol = 1 - r12 ** 2
    vif = 1 / tol
    level = "severe" if vif >= 10 else "high" if vif >= 5 or abs(r12) >= threshold else \
        "moderate" if abs(r12) >= 0.5 else "low"
    return {"r12": float(r12), "r12_squared": float(r12 ** 2), "tolerance": float(tol), "vif": float(vif),
            "se_inflation": float(math.sqrt(vif)), "threshold": threshold, "level": level,
            "flagged": bool(abs(r12) >= threshold or vif >= 5)}


def compare_models(simple: dict, multi: dict) -> dict:
    """Simple (X1 only) vs two-predictor model: every slide metric side by side, plus how much the
    X1 coefficient moves once X2 is added — a sign flip or a lost t-test is the symptom of overlap."""
    rows = {
        "n": (simple["n"], multi["n"]),
        "R (multiple correlation)": (abs(simple["r"]), multi["R"]),
        "R²": (simple["r2"], multi["r2"]),
        "adjusted R²": (simple["adj_r2"], multi["adj_r2"]),
        "F": (simple["F"], multi["F"]),
        "F critical (α = 0.05)": (simple["F_critical"], multi["F_critical"]),
        "df (regression, error)": (f"1, {simple['anova']['df_error']}", f"2, {multi['anova']['df_error']}"),
        "SS regression": (simple["anova"]["ss_regression"], multi["anova"]["ss_regression"]),
        "SS error": (simple["anova"]["ss_error"], multi["anova"]["ss_error"]),
        "MS error": (simple["anova"]["ms_error"], multi["anova"]["ms_error"]),
        **{m: (simple["metrics"][m], multi["metrics"][m]) for m in ("MAE", "MSE", "RMSE", "MAPE")},
    }
    b1_alone, b1_joint = simple["slope"], multi["b1"]
    c1 = multi["coefficients"][0]
    return {
        "table": [{"metric": k, "simple": v[0], "multiple": v[1]} for k, v in rows.items()],
        "delta_r2": multi["r2"] - simple["r2"], "delta_adj_r2": multi["adj_r2"] - simple["adj_r2"],
        "b1_simple": b1_alone, "b1_multiple": b1_joint,
        "b1_change_pct": (b1_joint - b1_alone) / abs(b1_alone) * 100,
        "b1_sign_flip": bool(np.sign(b1_alone) != np.sign(b1_joint)),
        "se_b1_simple": simple["se_slope"], "se_b1_multiple": c1["se"],
        "b1_lost_significance": bool(simple["slope_significant"] and not c1["significant"]),
    }


def predict(model: dict, x1: float, x2: float | None = None) -> float:
    """Slide 32 'estimate for unknown values': plug new X values into the fitted equation."""
    if x2 is None:
        return float(model["slope"] * x1 + model["intercept"])
    return float(model["b0"] + model["b1"] * x1 + model["b2"] * x2)


# ---------------------------------------------------------------- Module 1: model evaluation

def min_max(x) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def standardise(x) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


Fitter = Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]]


def linear_fitter(X: np.ndarray, y: np.ndarray):
    """Fit y on 1 or 2 predictor columns with the course formulas; return a predict function."""
    if X.shape[1] == 1:
        m = simple_regression(X[:, 0], y)
        return lambda Z: m["slope"] * Z[:, 0] + m["intercept"]
    m = multiple_regression_2(X[:, 0], X[:, 1], y)
    return lambda Z: m["b0"] + m["b1"] * Z[:, 0] + m["b2"] * Z[:, 1]


def holdout(X, y, train_frac: float = 2 / 3, seed: int = 42, fitter: Fitter = linear_fitter) -> dict:
    """Shuffle once, train on ~two-thirds, score once on the rest (course rule of thumb)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    idx = np.random.default_rng(seed).permutation(y.size)
    cut = int(round(train_frac * y.size))
    tr, te = idx[:cut], idx[cut:]
    predict = fitter(X[tr], y[tr])
    return {"n_train": int(tr.size), "n_test": int(te.size), "test": all_metrics(y[te], predict(X[te])),
            "train": all_metrics(y[tr], predict(X[tr]))}


def kfold_cv(X, y, k: int = 5, seed: int = 42, fitter: Fitter = linear_fitter) -> dict:
    """k-fold CV: every record is validated exactly once; report per-fold and mean scores."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    idx = np.random.default_rng(seed).permutation(y.size)
    folds = np.array_split(idx, k)
    rows = []
    for i, val in enumerate(folds):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != i])
        predict = fitter(X[tr], y[tr])
        rows.append({"fold": i + 1, "n_val": int(val.size), **all_metrics(y[val], predict(X[val]))})
    mean = {m: float(np.mean([r[m] for r in rows])) for m in ("MAE", "RMSE", "MAPE", "R2")}
    sd = {m: float(np.std([r[m] for r in rows], ddof=1)) for m in ("MAE", "RMSE", "MAPE", "R2")}
    return {"k": k, "folds": rows, "mean": mean, "sd": sd}


def bootstrap_ci(x, stat: Callable[[np.ndarray], float], reps: int = 5000, conf: float = 0.95, seed: int = 42) -> dict:
    """Percentile bootstrap: resample with replacement, same size as the original data."""
    a = np.asarray(x, float)
    a = a[~np.isnan(a)]
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(a, size=a.size, replace=True)) for _ in range(reps)])
    lo, hi = np.quantile(boots, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return {"estimate": float(stat(a)), "lower": float(lo), "upper": float(hi), "reps": reps,
            "boot_se": float(boots.std(ddof=1)), "distribution": boots}
