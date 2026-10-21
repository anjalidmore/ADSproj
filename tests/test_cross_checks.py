"""Course-formula implementations agree with independent library implementations."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.formula.api import ols

from co2ads import correlation as cr
from co2ads import descriptive as ds
from co2ads import inference as inf
from co2ads import lof as lofm
from co2ads import regression as rg

rng = np.random.default_rng(7)
X = rng.lognormal(1, 0.9, 214)
Y = 0.8 * X + rng.normal(0, 1, 214)


def test_quantiles_follow_n_plus_1_rule():
    for k, parts in [(1, 4), (2, 4), (3, 4), (1, 10), (90, 100)]:
        assert ds.quantile(X, k, parts) == pytest.approx(np.quantile(X, k / parts, method="weibull"))
    assert ds.median(X) == pytest.approx(np.median(X))
    assert ds.median([1, 3, 5, 7]) == 4


def test_moments_match_scipy():
    assert ds.std(X) == pytest.approx(np.std(X))
    assert ds.std(X, 1) == pytest.approx(np.std(X, ddof=1))
    assert ds.kurtosis_beta2(X) == pytest.approx(stats.kurtosis(X, fisher=False, bias=True))
    assert ds.mean_shortcut(X, 3.0) == pytest.approx(X.mean())
    assert ds.std_assumed_mean(X, 3.0) == pytest.approx(np.std(X))


def test_grouped_formulas_consistent():
    t = ds.frequency_table(X)
    assert t["f"].sum() == X.size
    g = ds.grouped_mean(t)
    assert g["direct"] == pytest.approx(g["step_deviation"])
    assert ds.grouped_std(t) == pytest.approx(
        np.sqrt((t["f"] * (t["mid"] - g["direct"]) ** 2).sum() / X.size))
    # grouped median lies inside the median class
    med = ds.grouped_median(t)
    assert t["lower"].min() <= med <= t["upper"].max()
    assert abs(med - np.median(X)) < t.attrs["width"]


def test_weighted_and_combined_means():
    w = rng.uniform(1, 10, X.size)
    assert ds.weighted_mean(X, w) == pytest.approx(np.average(X, weights=w))
    a, b = X[:100], X[100:]
    assert ds.combined_mean([a.mean(), b.mean()], [a.size, b.size]) == pytest.approx(X.mean())


def test_pearson_three_methods_agree():
    r = np.corrcoef(X, Y)[0, 1]
    assert cr.pearson_actual_mean(X, Y) == pytest.approx(r)
    assert cr.pearson_direct(X, Y) == pytest.approx(r)
    assert cr.pearson_assumed_mean(X, Y, 5, 4) == pytest.approx(r)
    assert cr.covariance(X, Y) == pytest.approx(np.cov(X, Y, ddof=0)[0, 1])


def test_t_tests_match_scipy():
    a, b = X[:90], X[90:]
    assert inf.t_test_independent(a, b)["p_value"] == pytest.approx(stats.ttest_ind(a, b).pvalue)
    assert inf.t_test_one_sample(a, 3)["p_value"] == pytest.approx(stats.ttest_1samp(a, 3).pvalue)
    assert inf.t_test_paired(a, b[:90])["p_value"] == pytest.approx(stats.ttest_rel(b[:90], a).pvalue)


def test_one_way_anova_matches_scipy_unequal_groups():
    groups = {"g1": X[:30], "g2": X[30:110], "g3": X[110:]}
    r = inf.one_way_anova(groups)
    s = stats.f_oneway(*groups.values())
    assert r["F"] == pytest.approx(s.statistic)
    assert r["p_value"] == pytest.approx(s.pvalue)
    assert r["ss_check"] == pytest.approx(0, abs=1e-8)


def test_two_way_anova_matches_statsmodels():
    m = pd.DataFrame(rng.normal(5, 2, (6, 4)), index=list("ABCDEF"), columns=[1990, 2000, 2010, 2020])
    r = inf.two_way_anova(m)
    long = m.stack().rename("y").reset_index()
    long.columns = ["row", "col", "y"]
    a = sm.stats.anova_lm(ols("y ~ C(row) + C(col)", long).fit(), typ=2)
    assert r["F_rows"] == pytest.approx(a.loc["C(row)", "F"])
    assert r["F_columns"] == pytest.approx(a.loc["C(col)", "F"])


def test_regression_matches_lstsq_and_statsmodels():
    x2 = rng.normal(size=X.size)
    r = rg.multiple_regression_2(X, x2, Y)
    fit = sm.OLS(Y, sm.add_constant(np.column_stack([X, x2]))).fit()
    assert [r["b0"], r["b1"], r["b2"]] == pytest.approx(fit.params)
    assert r["r2"] == pytest.approx(fit.rsquared)
    assert r["adj_r2"] == pytest.approx(fit.rsquared_adj)
    assert r["F"] == pytest.approx(fit.fvalue)
    assert [c["se"] for c in r["coefficients"]] == pytest.approx(fit.bse[1:])
    s = rg.simple_regression(X, Y)
    f1 = sm.OLS(Y, sm.add_constant(X)).fit()
    assert [s["intercept"], s["slope"]] == pytest.approx(f1.params)
    assert s["se_slope"] == pytest.approx(f1.bse[1])


def test_lof_matches_sklearn_on_continuous_data():
    pts = rng.normal(size=(60, 2))
    mine = lofm.lof(pts, k=5)["lof"].to_numpy()
    sk = LocalOutlierFactor(n_neighbors=5, metric="manhattan").fit(pts)
    assert mine == pytest.approx(-sk.negative_outlier_factor_)


def test_kfold_uses_every_record_once():
    cv = rg.kfold_cv(X.reshape(-1, 1), Y, k=5)
    assert sum(f["n_val"] for f in cv["folds"]) == X.size


def test_bootstrap_reproducible():
    a = rg.bootstrap_ci(X, np.median, reps=500, seed=1)
    b = rg.bootstrap_ci(X, np.median, reps=500, seed=1)
    assert a["lower"] == b["lower"] and a["lower"] <= a["estimate"] <= a["upper"]


def test_error_simulation_type1_near_alpha():
    r = inf.error_rate_simulation(X, n=50, shift=1.0, reps=4000, seed=3)
    assert 0.03 < r["type1_rate"] < 0.09
