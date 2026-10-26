"""Module 2 additions: discrete series, cumulative frequencies, class conversions, open-end classes,
corrections, moments, partial/multiple correlation — checked against the slide examples and libraries."""

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from co2ads import correlation as cr
from co2ads import descriptive as ds


def test_discrete_mean_marks_example():  # Module 2: marks of 60 students
    t = ds.discrete_from_counts([20, 30, 40, 50, 60, 70], [8, 12, 20, 10, 6, 4])
    m = ds.discrete_mean(t, assumed=40)
    assert m["direct"] == pytest.approx(41)
    assert m["shortcut"] == pytest.approx(41)


def test_discrete_median_income_example():  # Module 2: median income ₹16,800
    t = ds.discrete_from_counts([15000, 15500, 16800, 18000, 18500, 17800], [24, 26, 20, 16, 6, 30])
    assert t["cf"].iloc[-1] == 122
    assert ds.discrete_median(t) == 16800


def test_discrete_quantiles_example():  # Module 2: P20 = Q1 = D3 = 20, median 30, Q3 40
    t = ds.discrete_from_counts([10, 20, 30, 40, 50], [4, 6, 10, 8, 2])
    assert ds.discrete_quantile(t, 20, 100) == 20
    assert ds.discrete_quantile(t, 1, 4) == 20
    assert ds.discrete_quantile(t, 3, 10) == 20
    assert ds.discrete_median(t) == 30
    assert ds.discrete_quantile(t, 3, 4) == 40


def test_discrete_std_methods_agree():  # Module 2: σ for a discrete series, N = 217, mean 7.09
    X = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
    f = [3, 7, 22, 60, 85, 32, 8]
    t = ds.discrete_from_counts(X, f)
    assert ds.discrete_mean(t)["direct"] == pytest.approx(7.09, abs=0.005)
    s = ds.discrete_std(t, assumed=6.5)
    assert s["actual_mean_method"] == pytest.approx(s["assumed_mean_method"])
    assert s["actual_mean_method"] == pytest.approx(np.sqrt(np.cov(X, fweights=f, ddof=0)))


def test_discrete_table_children_example():  # Module 2: children per family, 35 families
    raw = [1, 0, 2, 3, 4, 5, 6, 7, 2, 3, 4, 0, 2, 5, 8, 4, 5, 12, 6, 3, 2, 7, 6, 5, 3, 3, 7, 8, 9, 7, 9, 4, 5, 4, 3]
    t = ds.discrete_table(raw).set_index("X")
    assert t["f"].sum() == 35
    assert (t.loc[0, "f"], t.loc[1, "f"], t.loc[2, "f"]) == (2, 1, 4)


def test_grouped_quartile_example():  # Module 2: continuous series Q1 = 10
    t = ds.class_table([0, 5, 10, 15, 20], [5, 10, 15, 20, 25], [7, 18, 25, 30, 20])
    assert ds.grouped_quantile(t, 1, 4) == pytest.approx(10)
    assert ds.grouped_quantile(t, 3, 4) == pytest.approx(15 + 25 / 30 * 5)


def test_median_from_cumulative_example_and_slide_error():
    # Module 2 'Median from cumulative data': N = 655, median class 10–15, L = 10, cf = 224, f = 241.
    # The classes are 5 wide; the slide uses i = 10 and prints 14.29. The correct value is 12.147.
    lo = np.arange(0, 45, 5)
    t = ds.class_table(lo, lo + 5, [29, 195, 241, 117, 52, 10, 6, 3, 2])
    assert t["f"].sum() == 655
    assert ds.grouped_median(t) == pytest.approx(10 + (327.5 - 224) / 241 * 5)
    assert ds.ogive_value(t, 327.5) == pytest.approx(ds.grouped_median(t))
    x, y = ds.ogive_intersection(t)
    assert y == pytest.approx(327.5) and x == pytest.approx(ds.grouped_median(t))


def test_cumulative_tables():
    t = ds.class_table([0, 5, 10], [5, 10, 15], [3, 5, 2])
    c = ds.cumulative_frequencies(t)
    assert c["less_than_cf"].tolist() == [3, 8, 10]
    assert c["more_than_cf"].tolist() == [10, 7, 2]


def test_classes_from_midpoints_example():  # Module 2: mid-value 115, i = 10 → 110–120
    lo, hi = ds.classes_from_midpoints([115, 125, 135, 145])
    assert (lo[0], hi[0]) == (110, 120)


def test_inclusive_to_exclusive():  # Module 2: subtract 0.5 from lower limits, add to upper
    lo, hi = ds.inclusive_to_exclusive([0, 5, 10], [4, 9, 14])
    assert lo.tolist() == [-0.5, 4.5, 9.5] and hi.tolist() == [4.5, 9.5, 14.5]


def test_open_end_mean_example():  # Module 2: 'Below 10' and 'Above 50', uniform width 10
    t = ds.class_table([np.nan, 10, 20, 30, 40, 50], [10, 20, 30, 40, 50, np.inf], [4, 6, 10, 15, 8, 7])
    expected = (5 * 4 + 15 * 6 + 25 * 10 + 35 * 15 + 45 * 8 + 55 * 7) / 50
    assert ds.open_end_mean(t, open_first=True, open_last=True) == pytest.approx(expected)


def test_corrected_mean_examples():  # Module 2 Illustrations 6 and 7
    assert ds.corrected_mean(100, 40, wrong=[83], right=[53])["correct_mean"] == pytest.approx(39.7)
    assert ds.corrected_mean(100, 40, wrong=[30, 27], right=[3, 72])["correct_mean"] == pytest.approx(40.18)
    r = ds.corrected_mean(5, 3.0, wrong=[5])                     # removing an item changes N
    assert r["n"] == 4 and r["correct_mean"] == pytest.approx(10 / 4)


def test_moments_match_scipy():
    x = np.random.default_rng(3).lognormal(1, 0.8, 300)
    m = ds.moments(x)
    for r in (2, 3, 4):
        assert m["central"][f"mu{r}"] == pytest.approx(stats.moment(x, r))
        assert m["central_from_raw"][f"mu{r}"] == pytest.approx(m["central"][f"mu{r}"])
    assert m["gamma1"] == pytest.approx(stats.skew(x))
    assert m["beta2"] == pytest.approx(stats.kurtosis(x, fisher=False))


def test_partial_and_multiple_correlation():
    rng = np.random.default_rng(5)
    x2 = rng.normal(size=200)
    x1 = 0.8 * x2 + rng.normal(size=200)
    y = x1 + 2 * x2 + rng.normal(size=200)
    # partial r = correlation of residuals after removing x2 from both
    ry = sm.OLS(y, sm.add_constant(x2)).fit().resid
    r1 = sm.OLS(x1, sm.add_constant(x2)).fit().resid
    assert cr.partial_correlation(y, x1, x2) == pytest.approx(np.corrcoef(ry, r1)[0, 1])
    R2 = sm.OLS(y, sm.add_constant(np.column_stack([x1, x2]))).fit().rsquared
    assert cr.multiple_correlation(y, x1, x2) == pytest.approx(np.sqrt(R2))


def test_grouped_vs_raw_matches_individual_formulas():
    x = np.random.default_rng(7).lognormal(1, 0.9, 213)
    t = ds.frequency_table(x)
    g = ds.grouped_vs_raw(x, t).set_index("measure")
    s = ds.summary(x)
    assert g.loc["Mean", "raw"] == pytest.approx(s["mean"]) and g.loc["Mean", "grouped"] == pytest.approx(ds.grouped_mean(t)["direct"])
    assert g.loc["Median", "raw"] == pytest.approx(s["median"]) and g.loc["Median", "grouped"] == pytest.approx(ds.grouped_median(t))
    assert g.loc["Q3", "grouped"] == pytest.approx(ds.grouped_quantile(t, 3, 4))
    assert g.loc["Standard deviation σ", "raw"] == pytest.approx(s["std"])
    assert g.loc["Standard deviation σ", "grouped"] == pytest.approx(ds.grouped_std(t))
    assert np.isnan(g.loc["Mode", "raw"]) and g.loc["Mode", "grouped"] == pytest.approx(s["grouped_mode"])


def test_grouped_mean_exact_when_values_sit_at_midpoints():
    x = np.repeat([0.5, 1.5, 2.5, 3.5], [3, 9, 5, 2])
    g = ds.grouped_vs_raw(x, ds.frequency_table(x, width=1, start=0)).set_index("measure")
    assert g.loc["Mean", "difference"] == pytest.approx(0)
    assert g.loc["Standard deviation σ", "difference"] == pytest.approx(0)


def test_per_capita_means_reconcile_with_world_row():
    from co2ads import data
    from co2ads import results as R
    df = data.load_clean()
    m = R.per_capita_means(df)
    cs = data.cross_section(df, m["year"], ["co2", "co2_per_capita"])
    # ΣWX/ΣW with W = population is total CO2 over total population (per-person values are rounded in the file)
    assert m["weighted_mean"] == pytest.approx(cs["co2"].sum() * 1e6 / cs["population"].sum(), rel=1e-4)
    world = df[(df["country"] == "World") & (df["year"] == m["year"])].iloc[0]
    assert m["world_per_capita"] == pytest.approx(world["co2_per_capita"], abs=1e-4)
    assert m["steps"]["t_per_person"].tolist() == pytest.approx(
        [m["simple_mean"], m["weighted_mean"], m["with_international"], m["world_per_capita"]])
    assert m["international_mt"] > 0 and abs(m["with_international"] - m["world_per_capita"]) < 0.01
