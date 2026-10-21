"""Each formula reproduces the worked example from the course slides (page refs in comments)."""

import numpy as np
import pytest

from co2ads import inference as inf
from co2ads import lof as lofm
from co2ads import regression as rg
from co2ads import timeseries as ts


def test_one_sample_t_iq_example():  # Module 3 slides 33–36
    iq = [70, 120, 110, 101, 88, 83, 95, 98, 107, 100]
    r = inf.t_test_one_sample(iq, 100)
    assert r["mean"] == pytest.approx(97.2)
    assert r["sd"] == pytest.approx(14.27, abs=0.01)
    assert abs(r["statistic"]) == pytest.approx(0.62, abs=0.01)
    assert r["critical"] == pytest.approx(2.262, abs=0.001)
    assert r["decision"] == "Fail to reject H0"
    ci = inf.ci_mean(iq)  # n < 30 -> t interval
    assert ci["method"] == "t"
    assert ci["lower"] == pytest.approx(87.0, abs=0.1) and ci["upper"] == pytest.approx(107.4, abs=0.1)


def test_independent_t_diet_example():  # Module 3 slides 37–39
    a = [25, 32, 30, 34, 24, 14, 32, 24, 30, 31, 35, 25]
    b = [44, 34, 22, 10, 47, 31, 40, 30, 32, 35, 18, 21, 35, 29, 22]
    r = inf.t_test_independent(a, b)
    assert r["pooled_var"] == pytest.approx(71.6)
    # slide rounds √10.74 before dividing (0.609); unrounded value is 0.6103
    assert abs(r["statistic"]) == pytest.approx(0.609, abs=0.002)
    assert r["df"] == 25
    assert r["decision"] == "Fail to reject H0"


def test_paired_t_soldiers_example():  # Module 3 slides 40–41
    before = [67, 24, 57, 55, 63, 54, 56, 68, 33, 43]
    after = [70, 38, 58, 58, 56, 67, 68, 75, 42, 38]
    r = inf.t_test_paired(before, after)
    assert r["mean_diff"] == pytest.approx(5)          # after − before (slide uses before − after = −5)
    assert abs(r["statistic"]) == pytest.approx(2.16, abs=0.01)


def test_z_ci_wait_time_example():  # Module 3 slide 10 (σ known)
    x = np.full(50, 45.0)
    ci = inf.ci_mean(x, sigma=10)
    assert ci["lower"] == pytest.approx(42.23, abs=0.01) and ci["upper"] == pytest.approx(47.77, abs=0.01)


def test_z_test_ocean_example():  # Module 3 slide 26
    z = np.random.default_rng(0).normal(size=40)
    z = (z - z.mean()) / z.std(ddof=1)
    depths = 59.1 + 5.2 * z                      # exact mean 59.1, sample SD 5.2
    r = inf.z_test_mean(depths, 57.4)
    assert r["statistic"] == pytest.approx(2.067, abs=0.001)
    assert r["decision"] == "Reject H0"


def test_one_way_anova_wheat_example():  # Module 3 slides 47–52
    r = inf.one_way_anova({"A": [6, 7, 3, 8], "B": [5, 5, 3, 7], "C": [5, 4, 3, 4]})
    t = r["table"]
    assert t.loc["Between groups", "SS"] == pytest.approx(8)
    assert t.loc["Within groups", "SS"] == pytest.approx(24)
    assert r["F"] == pytest.approx(1.5)
    assert r["critical"] == pytest.approx(4.26, abs=0.01)
    assert r["decision"] == "Fail to reject H0"


def test_multiple_regression_example():  # Module 4 slides 26–39
    y = [140, 155, 159, 179, 192, 200, 212, 215]
    x1 = [60, 62, 67, 70, 71, 72, 75, 78]
    x2 = [22, 25, 24, 20, 15, 14, 14, 11]
    r = rg.multiple_regression_2(x1, x2, y)
    assert r["b1"] == pytest.approx(3.1479, abs=1e-4)
    assert r["b2"] == pytest.approx(-1.6561, abs=1e-4)
    assert r["b0"] == pytest.approx(-6.8675, abs=1e-3)
    assert r["r2"] == pytest.approx(0.9626, abs=1e-4)
    assert r["F"] == pytest.approx(64.35, abs=0.1)
    assert r["r12"] == pytest.approx(-0.88, abs=0.005)
    assert r["coefficients"][0]["t"] == pytest.approx(3.756, abs=0.01)
    assert r["coefficients"][0]["significant"]
    assert r["b0"] + r["b1"] * 74 + r["b2"] * 14 == pytest.approx(202.89, abs=0.01)


def test_simple_regression_expenditure_example():  # Module 6 slides 54–59
    r = rg.simple_regression([1, 2, 3, 4, 5], [12, 19, 29, 37, 45])
    assert r["slope"] == pytest.approx(8.4)
    assert r["intercept"] == pytest.approx(3.2)
    assert np.abs(r["residuals"]).sum() == pytest.approx(2.4)
    assert (r["residuals"] ** 2).sum() == pytest.approx(1.6)


def test_lof_manhattan_example():  # Module 5, all 8 slides
    X = [[0, 0], [0, 1], [1, 1], [3, 0]]
    out = lofm.lof(X, k=2)
    assert out["k_distance"].tolist() == [2, 1, 2, 3]
    assert out["lrd"].to_numpy() == pytest.approx([2 / 3, 0.5, 2 / 3, 1 / 3])
    assert out["lof"].to_numpy() == pytest.approx([0.875, 4 / 3, 0.875, 2.0])


SALES = [18, 20, 34, 44, 22, 20, 38, 48, 18, 20, 42, 56]


def test_additive_decomposition_example():  # Module 6 slides 13–27
    r = ts.classical_decomposition(SALES, 4, "additive")
    assert r["trend"][2:10] == pytest.approx([29.5, 30, 30.5, 31.5, 31.5, 31, 31.5, 33])
    assert r["seasonal_indices"] == pytest.approx([-10.4375, -11.6875, 6.0625, 16.0625])
    assert r["trend_per_period"] == pytest.approx(0.5)
    assert r["forecast"](13) == pytest.approx(23.3125)   # Year 4 Q2
    assert r["forecast"](18) == pytest.approx(43.5625)   # Year 5 Q3


def test_multiplicative_decomposition_example():  # Module 6 slides 28–43
    r = ts.classical_decomposition(SALES, 4, "multiplicative")
    assert r["seasonal_indices"] == pytest.approx([0.654, 0.628, 1.193, 1.525], abs=0.002)
    assert r["seasonal_indices"].sum() == pytest.approx(4)
    assert r["forecast"](13) == pytest.approx(21.98, abs=0.05)


def test_average_and_ma_forecast_metrics():  # Module 6 slides 47–50
    y = np.array([20, 22, 21, 25, 24, 26], float)
    fa = ts.average_forecast(y)
    assert fa[1:] == pytest.approx([20, 21, 21, 22, 22.4])
    m = ts.evaluate(y, fa, y)
    assert m["MAE"] == pytest.approx(2.32)
    assert m["RMSE"] == pytest.approx(2.72, abs=0.01)
    assert m["MAPE"] == pytest.approx(9.45, abs=0.01)
    assert m["MASE"] == pytest.approx(1.16)
    fm = ts.moving_average_forecast(y, 3)
    m3 = ts.evaluate(y, fm, y[2:])
    assert m3["MAE"] == pytest.approx(2.67, abs=0.01)
    assert m3["MAPE"] == pytest.approx(10.60, abs=0.01)
    assert m3["MASE"] == pytest.approx(1.14, abs=0.01)


def test_poisson_pmf_hospital_example():  # Module 3 handwritten notes: μ = 4 calls
    assert inf.poisson_pmf([0, 1, 2], 4).sum() == pytest.approx(0.238, abs=0.001)
    assert inf.poisson_pmf(3, 4) == pytest.approx(0.195, abs=0.001)
    assert np.isfinite(inf.poisson_pmf(60, 12.0))
