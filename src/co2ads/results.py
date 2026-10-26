"""Every analysis applied to the real data. The app, report, slides and viva guide all read
their numbers from here (via scripts/export_results.py -> docs/results.json), so no
finding is ever typed in by hand.

Each function takes the clean DataFrame and returns a dict (DataFrames allowed).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from . import correlation as cr
from . import data
from . import descriptive as ds
from . import inference as inf
from . import lof as lofm
from . import regression as rg
from . import timeseries as ts

Y, G, B = config.LATEST_YEAR, config.GDP_YEAR, config.BASE_YEAR


# ---------------------------------------------------------------- research questions 1–4

def global_trend(df: pd.DataFrame) -> dict:
    w = data.world_series(df)
    w_pop = data.world_series(df, "population")
    peak_year = int(w.idxmax())
    return {
        "series": w,
        "first_year": int(w.index.min()),
        "co2_1950": float(w.loc[1950]), "co2_1990": float(w.loc[1990]), "co2_base": float(w.loc[B]),
        "co2_latest": float(w.loc[Y]),
        "change_1990_latest_pct": float((w.loc[Y] / w.loc[1990] - 1) * 100),
        "multiple_1950_latest": float(w.loc[Y] / w.loc[1950]),
        "peak_year": peak_year, "peak_value": float(w.max()),
        "covid_drop_pct": float((w.loc[2020] / w.loc[2019] - 1) * 100),
        "rebound_2021_pct": float((w.loc[2021] / w.loc[2020] - 1) * 100),
        "per_capita_latest": float(w.loc[Y] * 1e6 / w_pop.loc[Y]),
        "cumulative_latest": float(data.world_series(df, "cumulative_co2").loc[Y]),
    }


def top_emitters(df: pd.DataFrame, n: int = 10) -> dict:
    cs = data.cross_section(df, Y, ["co2"])
    world = data.world_series(df).loc[Y]
    cs["share_pct"] = cs["co2"] / world * 100
    cs["rank_total"] = cs["co2"].rank(ascending=False, method="min").astype(int)
    pc = data.cross_section(df, Y, ["co2_per_capita"])
    pc["rank_per_capita"] = pc["co2_per_capita"].rank(ascending=False, method="min").astype(int)
    top = cs.nlargest(n, "co2").merge(pc[["country", "co2_per_capita", "rank_per_capita"]], on="country", how="left")
    cum = data.cross_section(df, Y, ["cumulative_co2"])
    cum["share_pct"] = cum["cumulative_co2"] / data.world_series(df, "cumulative_co2").loc[Y] * 100
    return {
        "table": top[["country", "continent", "co2", "share_pct", "rank_total", "co2_per_capita", "rank_per_capita", "population"]],
        "top_n_share_pct": float(top["share_pct"].sum()),
        "top3_share_pct": float(top["share_pct"].head(3).sum()),
        "cumulative_top": cum.nlargest(n, "cumulative_co2")[["country", "cumulative_co2", "share_pct"]],
        "per_capita_top": pc.nlargest(n, "co2_per_capita")[["country", "continent", "co2_per_capita", "population"]],
        "n_countries": int(len(cs)),
    }


def changes(df: pd.DataFrame, min_base_mt: float = 10.0) -> dict:
    p = data.paired_years(df, "co2", B, Y)
    p["abs_change"] = p[f"co2_{Y}"] - p[f"co2_{B}"]
    big = p[p[f"co2_{B}"] >= min_base_mt].copy()
    big["pct_change"] = (big[f"co2_{Y}"] / big[f"co2_{B}"] - 1) * 100
    cols = ["country", "continent", f"co2_{B}", f"co2_{Y}", "abs_change"]
    return {
        "n_pairs": int(len(p)), "n_pct_eligible": int(len(big)), "min_base_mt": min_base_mt,
        "abs_increase": p.nlargest(10, "abs_change")[cols],
        "abs_decrease": p.nsmallest(10, "abs_change")[cols],
        "pct_increase": big.nlargest(10, "pct_change")[cols + ["pct_change"]],
        "pct_decrease": big.nsmallest(10, "pct_change")[cols + ["pct_change"]],
        "n_decreased": int((p["abs_change"] < 0).sum()), "n_increased": int((p["abs_change"] > 0).sum()),
    }


# ---------------------------------------------------------------- Module 2: distribution of per-capita CO2

def per_capita_means(df: pd.DataFrame, year: int = Y) -> dict:
    """Simple mean (one vote per country) → population-weighted mean (one vote per person) → World figure.

    ΣWX/ΣW with W = population equals Σ country CO2 / Σ country population. The World row adds
    international aviation and shipping (no country owns them) and divides by the World population,
    which also counts territories without a per-person value.
    """
    cs = data.cross_section(df, year, ["co2_per_capita"])
    x, w = cs["co2_per_capita"], cs["population"]
    country_mt = float((x * w).sum() / 1e6)
    country_pop = float(w.sum())
    intl_mt = float(df.loc[(df["entity_type"] == "international") & (df["year"] == year), "co2"].sum())
    world = df.loc[(df["country"] == "World") & (df["year"] == year)].iloc[0]
    simple, weighted = ds.mean_direct(x), ds.weighted_mean(x, w)
    with_intl = (country_mt + intl_mt) * 1e6 / country_pop
    world_pc = float(world["co2"] * 1e6 / world["population"])
    steps = pd.DataFrame([
        ("Simple mean of countries", "ΣX / N", f"{len(cs)} countries, each counted once", simple),
        ("Population-weighted mean", "ΣWX / ΣW = Σ CO₂ / Σ population",
         f"{country_mt:,.0f} Mt / {country_pop / 1e9:.3f} bn people", weighted),
        ("+ international aviation & shipping", "(Σ CO₂ + international) / Σ population",
         f"({country_mt:,.0f} + {intl_mt:,.0f}) Mt / {country_pop / 1e9:.3f} bn", with_intl),
        ("World row", "World CO₂ / World population",
         f"{world['co2']:,.0f} Mt / {world['population'] / 1e9:.3f} bn", world_pc),
    ], columns=["step", "formula", "inputs", "t_per_person"])
    return {
        "year": year, "n": int(len(cs)), "simple_mean": simple, "weighted_mean": weighted,
        "with_international": with_intl, "world_per_capita": world_pc,
        "country_co2_mt": country_mt, "country_population": country_pop, "international_mt": intl_mt,
        "world_co2_mt": float(world["co2"]), "world_population": float(world["population"]),
        "steps": steps,
    }


def per_capita_distribution(df: pd.DataFrame) -> dict:
    cs = data.cross_section(df, Y, ["co2_per_capita"])
    x = cs["co2_per_capita"]
    t = ds.frequency_table(x)
    gm = ds.grouped_mean(t)
    world_pc = global_trend(df)["per_capita_latest"]
    by_cont = cs.dropna(subset=["continent"]).groupby("continent")["co2_per_capita"].agg(["count", "mean"])
    pop_weighted = ds.weighted_mean(cs["co2_per_capita"], cs["population"])
    return {
        "data": cs, "summary": ds.summary(x), "freq_table": t, "box": ds.box_plot(x),
        "outliers": ds.iqr_outliers(cs, "co2_per_capita")[["country", "continent", "co2_per_capita", "population"]],
        "grouped": {"mean_direct": gm["direct"], "mean_step": gm["step_deviation"], "assumed_mean": gm["assumed_mean"],
                    "median": ds.grouped_median(t), "mode": ds.grouped_mode(t),
                    "q1": ds.grouped_quantile(t, 1, 4), "q3": ds.grouped_quantile(t, 3, 4),
                    "std": ds.grouped_std(t)},
        "grouped_vs_raw": ds.grouped_vs_raw(x, t),
        "population_weighted_mean": pop_weighted,
        "world_per_capita": world_pc,
        "mean_comparison": per_capita_means(df),
        "by_continent": by_cont,
        "combined_mean_check": ds.combined_mean(by_cont["mean"], by_cont["count"]),
        "mean_excl_antarctica": float(cs.dropna(subset=["continent"])["co2_per_capita"].mean()),
        "log_summary": ds.summary(np.log10(x)),
    }


# ---------------------------------------------------------------- Module 2.2–2.3: remaining syllabus topics

PC_BANDS = [0, 1, 3, 6, 10, np.inf]
PC_LABELS = ["< 1 t", "1–3 t", "3–6 t", "6–10 t", "10 t and above"]


def years_of_decline(df: pd.DataFrame, start: int = B, end: int = Y) -> pd.Series:
    """Discrete variable: in how many of the years start+1 … end did a country's CO2 fall?

    Only countries with positive CO2 in every year of the window are counted.
    """
    c = data.countries(df)
    w = c[c["year"].between(start, end)].pivot(index="country", columns="year", values="co2").dropna()
    w = w[(w > 0).all(axis=1)]
    return (w.diff(axis=1).iloc[:, 1:] < 0).sum(axis=1).rename("years_of_decline")


def dominant_source(df: pd.DataFrame, year: int = Y) -> pd.Series:
    """Qualitative attribute: each country's largest reported fossil/industry CO2 source."""
    c = data.countries(df)
    f = c[c["year"] == year].set_index("country")[config.FUEL_COLUMNS].dropna(how="all")
    return f.idxmax(axis=1).str.replace("_co2", "").str.replace("other_industry", "other industry").rename("dominant_source")


def classification_and_tabulation(df: pd.DataFrame) -> dict:
    """Module 2.2: the four types of classification and simple / two-way / higher-order tables."""
    cs = data.cross_section(df, Y, ["co2", "co2_per_capita"]).dropna(subset=["continent"])
    cs["band"] = pd.cut(cs["co2_per_capita"], PC_BANDS, right=False, labels=PC_LABELS)
    fell = data.paired_years(df, "co2", B, Y)
    fell["fell_since_base"] = np.where(fell[f"co2_{Y}"] < fell[f"co2_{B}"], "Fell", "Rose or same")
    cs = cs.merge(fell[["country", "fell_since_base"]], on="country", how="left")
    cs = cs.merge(dominant_source(df).reset_index(), on="country", how="left")

    world_total = data.world_series(df).loc[Y]
    geo = cs.groupby("continent").agg(countries=("country", "size"), co2_mt=("co2", "sum"), population=("population", "sum"))
    geo["share_of_world_pct"] = geo["co2_mt"] / world_total * 100
    geo["co2_per_person_t"] = geo["co2_mt"] * 1e6 / geo["population"]
    geo = geo.sort_values("co2_mt", ascending=False)

    w = data.world_series(df).loc[1950:Y]
    decade = (w.index // 10 * 10)
    chrono = w.groupby(decade).agg(years="size", total_mt="sum", mean_annual_mt="mean")
    chrono.index = [f"{d}s" if d + 9 <= Y else f"{d}–{Y}" for d in chrono.index]
    chrono.index.name = "period"
    chrono["change_in_mean_pct"] = chrono["mean_annual_mt"].pct_change() * 100

    two_fold = cs["fell_since_base"].value_counts().rename_axis("attribute").rename("countries").to_frame()
    manifold = pd.crosstab([cs["continent"], cs["fell_since_base"]], cs["dominant_source"], margins=True, margins_name="Total")

    simple = cs["band"].value_counts().reindex(PC_LABELS).rename_axis("CO₂ per person").rename("countries").to_frame()
    simple.loc["Total"] = simple["countries"].sum()
    two_way = pd.crosstab(cs["continent"], cs["band"], margins=True, margins_name="Total")
    higher = pd.crosstab([cs["continent"], cs["fell_since_base"]], cs["band"], margins=True, margins_name="Total")
    return {"data": cs, "geographical": geo, "chronological": chrono, "two_fold": two_fold, "manifold": manifold,
            "quantitative": simple, "simple_table": simple, "two_way_table": two_way, "higher_order_table": higher,
            "n": int(len(cs))}


def univariate_extras(df: pd.DataFrame) -> dict:
    """Module 2.3 pieces beyond the summary table: discrete series, cumulative frequencies & ogive,
    open-end classes, moments, empirical mode, correcting incorrect items, inclusive classes."""
    x = data.cross_section(df, Y, ["co2_per_capita"])["co2_per_capita"]
    t = ds.frequency_table(x)

    yod = years_of_decline(df)
    dt = ds.discrete_table(yod)
    dstd = ds.discrete_std(dt)
    # Inclusive classes 0–3, 4–7, … of the discrete count, converted to exclusive (−0.5 / +0.5).
    lo_inc = np.arange(0, int(yod.max()) + 1, 4)
    hi_inc = lo_inc + 3
    f_inc = [int(((yod >= lo) & (yod <= hi)).sum()) for lo, hi in zip(lo_inc, hi_inc)]
    lo_ex, hi_ex = ds.inclusive_to_exclusive(lo_inc, hi_inc)
    t_inc = ds.class_table(lo_ex, hi_ex, f_inc)

    open_from = 15.0
    t_open = ds.close_top_classes(t, open_from)
    # Incorrect items: aggregates (regions, income groups, World…) wrongly averaged in with countries.
    all_rows = df[(df["year"] == Y) & df["co2_per_capita"].notna()]
    wrong = all_rows.loc[all_rows["entity_type"] != "country", "co2_per_capita"]
    incorrect_mean = float(all_rows["co2_per_capita"].mean())
    corr = ds.corrected_mean(len(all_rows), incorrect_mean, wrong=wrong.tolist())

    gm = ds.grouped_mode(t)
    return {
        "discrete_table": dt, "discrete_summary": ds.discrete_summary(dt),
        "discrete_std_methods": dstd, "discrete_variable": f"years with falling CO₂, {B + 1}–{Y}",
        "inclusive_table": pd.DataFrame({"inclusive_class": [f"{a}–{b}" for a, b in zip(lo_inc, hi_inc)],
                                         "exclusive_class": [f"{a:g}–{b:g}" for a, b in zip(lo_ex, hi_ex)], "f": f_inc}),
        "inclusive_grouped_median": ds.grouped_median(t_inc),
        "cumulative": ds.cumulative_frequencies(t), "freq_table": t,
        "ogive_median": ds.ogive_intersection(t)[0], "formula_median": ds.grouped_median(t),
        "ogive_q1": ds.ogive_value(t, t["f"].sum() / 4), "ogive_q3": ds.ogive_value(t, 3 * t["f"].sum() / 4),
        "open_table": t_open, "open_from": open_from,
        "open_mean": ds.open_end_mean(t_open), "closed_grouped_mean": ds.grouped_mean(t)["direct"],
        "open_median": ds.grouped_median(t_open), "actual_mean": float(x.mean()),
        "moments": ds.moments(x),
        "empirical_mode": ds.empirical_mode(x), "grouped_mode": gm["mode"],
        "correction": {**corr, "n_all": int(len(all_rows)), "n_wrong": int(len(wrong)), "incorrect_mean": incorrect_mean},
    }


def correlation_forms(df: pd.DataFrame) -> dict:
    """Module 2.4 'types of correlation': positive, negative, none, non-linear; simple, partial, multiple."""
    cs = data.cross_section(df, G, ["co2_per_capita", "gdp_per_capita", "energy_per_capita", "co2_per_unit_energy"])
    cs = cs[(cs[["co2_per_capita", "gdp_per_capita", "energy_per_capita", "co2_per_unit_energy"]] > 0).all(axis=1)]
    lg = {c: np.log10(cs[c]) for c in ["co2_per_capita", "gdp_per_capita", "energy_per_capita", "co2_per_unit_energy"]}
    pop = data.cross_section(df, Y, ["co2_per_capita"])
    uk = data.countries(df)
    uk = uk[(uk["country"] == "United Kingdom") & uk["year"].between(1990, Y)].dropna(subset=["co2"])
    y, x1, x2 = lg["co2_per_capita"], lg["gdp_per_capita"], lg["energy_per_capita"]
    return {
        "n": int(len(cs)), "data": cs.assign(**{f"log_{k}": v for k, v in lg.items()}), "uk": uk[["year", "co2"]],
        "pop_data": pop,
        "positive": cr.analyse(x1, y),
        "negative_cross_section": cr.analyse(x1, lg["co2_per_unit_energy"]),
        "negative_time": cr.analyse(uk["year"], uk["co2"]),
        "none": cr.analyse(pop["population"], pop["co2_per_capita"]),
        "nonlinear_raw": cr.analyse(cs["gdp_per_capita"], cs["co2_per_capita"]),
        "partial_gdp_given_energy": cr.partial_correlation(y, x1, x2),
        "partial_energy_given_gdp": cr.partial_correlation(y, x2, x1),
        "multiple_R": cr.multiple_correlation(y, x1, x2),
        "simple_r_gdp": cr.pearson_actual_mean(y, x1), "simple_r_energy": cr.pearson_actual_mean(y, x2),
    }


def diagram_data(df: pd.DataFrame) -> dict:
    """Tables behind the Module 2 one- and two-dimensional diagrams."""
    c = data.countries(df).dropna(subset=["continent"])
    yrs = [1990, B, Y]
    multi = c[c["year"].isin(yrs)].groupby(["continent", "year"])["co2"].sum().unstack("year")
    fuel = c[c["year"] == Y].groupby("continent")[config.FUEL_COLUMNS].sum()
    fuel.columns = [k.replace("_co2", "").replace("other_industry", "other industry") for k in fuel.columns]
    fuel_pct = fuel.div(fuel.sum(axis=1), axis=0) * 100
    top = data.cross_section(df, Y, ["co2", "co2_per_capita"]).nlargest(10, "co2")
    pc_top = data.cross_section(df, Y, ["co2_per_capita"]).nlargest(10, "co2_per_capita")
    w = data.world_series(df).loc[2000:]
    return {"multiple_bar": multi, "fuel_mt": fuel, "fuel_pct": fuel_pct,
            "rectangles": top[["country", "population", "co2_per_capita", "co2"]],
            "squares": top.head(5)[["country", "co2"]].assign(side=lambda d: np.sqrt(d["co2"]),
                                                                radius=lambda d: np.sqrt(d["co2"] / np.pi)),
            "per_capita_top": pc_top[["country", "co2_per_capita"]], "world_recent": w}


# ---------------------------------------------------------------- Module 2.4: correlation

CORR_COLS = ["co2_per_capita", "gdp_per_capita", "energy_per_capita", "population", "co2"]


def relationships(df: pd.DataFrame) -> dict:
    cs = data.cross_section(df, G, ["co2_per_capita", "gdp_per_capita"])
    cs = cs.merge(data.cross_section(df, G, ["energy_per_capita"])[["country", "energy_per_capita"]], on="country", how="left")
    cs = cs.merge(data.cross_section(df, G, ["co2"])[["country", "co2"]], on="country", how="left")
    raw = cr.analyse(cs["gdp_per_capita"], cs["co2_per_capita"])
    log = cr.analyse(np.log10(cs["gdp_per_capita"]), np.log10(cs["co2_per_capita"]))
    e = cs.dropna(subset=["energy_per_capita"])
    energy_log = cr.analyse(np.log10(e["energy_per_capita"]), np.log10(e["co2_per_capita"]))
    logged = cs[CORR_COLS].apply(np.log10)
    return {
        "data": cs, "gdp_raw": raw, "gdp_log": log, "energy_log": energy_log,
        "matrix_raw": cr.correlation_matrix(cs, CORR_COLS), "matrix_log": cr.correlation_matrix(logged, CORR_COLS),
        "pair_counts": cr.pair_counts(cs, CORR_COLS),
        "central_point": cr.central_data_point(cs, ["gdp_per_capita", "co2_per_capita"]),
        "farthest_from_centre": cs.assign(dist=cr.distance_from_centre(cs, ["gdp_per_capita", "co2_per_capita"]))
                                  .nlargest(5, "dist")[["country", "gdp_per_capita", "co2_per_capita", "dist"]],
    }


# ---------------------------------------------------------------- Module 3: inference

def inference(df: pd.DataFrame) -> dict:
    cs = data.cross_section(df, Y, ["co2_per_capita"])
    x = cs["co2_per_capita"].to_numpy()
    world_pc = global_trend(df)["per_capita_latest"]
    pc = data.paired_years(df, "co2_per_capita", B, Y)
    with_cont = cs.dropna(subset=["continent"]).assign(log_pc=lambda d: np.log10(d["co2_per_capita"]))
    groups = {k: v["log_pc"].to_numpy() for k, v in with_cont.groupby("continent")}

    # Two-way ANOVA without replication: continent (rows) × year (columns),
    # cell = population-weighted continental CO2 per person (Σ CO2 / Σ population of its countries).
    years = [1990, 2000, 2010, 2020, Y]
    c = data.countries(df)
    c = c[c["year"].isin(years) & c["continent"].notna()].dropna(subset=["co2", "population"])
    agg = c.groupby(["continent", "year"])[["co2", "population"]].sum()
    mat = (agg["co2"] * 1e6 / agg["population"]).unstack("year")
    mat.index.name, mat.columns.name = "continent", "year"

    # Poisson suitability: yearly count of countries (≥1 Mt) whose emissions fell by more than 10%.
    cc = data.countries(df)
    cc = cc[(cc["year"].between(1991, Y)) & (cc["co2"] >= 1)]
    counts = cc.assign(drop=cc["co2_growth_prct"] < -10).groupby("year")["drop"].sum()

    return {
        "ci_mean": inf.ci_mean(x),
        "bootstrap_median": {k: v for k, v in rg.bootstrap_ci(x, np.median, seed=config.SEED).items() if k != "distribution"},
        "empirical_raw": inf.empirical_rule(x), "empirical_log": inf.empirical_rule(np.log10(x)),
        "clt": inf.clt_simulation(x, n=30, reps=10000, seed=config.SEED),
        "z_test": inf.z_test_mean(x, world_pc),
        "paired": {**inf.t_test_paired(pc[f"co2_per_capita_{B}"], pc[f"co2_per_capita_{Y}"]),
                   "mean_before": float(pc[f"co2_per_capita_{B}"].mean()), "mean_after": float(pc[f"co2_per_capita_{Y}"].mean())},
        "paired_data": pc,
        "independent": inf.t_test_independent(groups["Europe"], groups["Asia"]),
        "independent_back": {"europe_geo_mean": float(10 ** groups["Europe"].mean()), "asia_geo_mean": float(10 ** groups["Asia"].mean())},
        # Sensitivity check: variances differ a lot, so also report Welch's unequal-variance t.
        "independent_welch_p": float(stats.ttest_ind(groups["Europe"], groups["Asia"], equal_var=False).pvalue),
        "anova": inf.one_way_anova(groups),
        "anova_data": with_cont,
        "two_way": inf.two_way_anova(mat), "two_way_matrix": mat,
        "errors": inf.error_rate_simulation(x, n=40, shift=1.5, reps=10000, seed=config.SEED),
        "poisson": inf.poisson_check(counts.to_numpy()), "poisson_counts": counts,
    }


# ---------------------------------------------------------------- Module 4 (+ Module 1 evaluation)

def regression(df: pd.DataFrame) -> dict:
    cs = data.cross_section(df, G, ["co2_per_capita", "gdp_per_capita", "energy_per_capita"])
    lx1, lx2, ly = (np.log10(cs[c]).to_numpy() for c in ("gdp_per_capita", "energy_per_capita", "co2_per_capita"))
    simple = rg.simple_regression(lx1, ly)
    multi = rg.multiple_regression_2(lx1, lx2, ly)
    back = {
        "simple_mape_original": rg.mape(cs["co2_per_capita"], 10 ** simple["fitted"]),
        "multi_mape_original": rg.mape(cs["co2_per_capita"], 10 ** multi["fitted"]),
    }
    # slide 32 "estimate for unknown values": a typical country at the sample-median GDP and energy per person
    gdp0, en0 = float(cs["gdp_per_capita"].median()), float(cs["energy_per_capita"].median())
    p1, p2 = rg.predict(simple, math.log10(gdp0)), rg.predict(multi, math.log10(gdp0), math.log10(en0))
    prediction = {"gdp_per_capita": gdp0, "energy_per_capita": en0,
                  "log_pred_simple": p1, "log_pred_multi": p2,
                  "co2_pc_simple": 10 ** p1, "co2_pc_multi": 10 ** p2}
    X1, X2 = lx1.reshape(-1, 1), np.column_stack([lx1, lx2])
    return {
        "data": cs.assign(fitted_simple=10 ** simple["fitted"], fitted_multi=10 ** multi["fitted"]),
        "simple": simple, "multiple": multi, "back_transformed": back,
        "comparison": rg.compare_models(simple, multi), "collinearity": multi["collinearity"],
        "prediction": prediction,
        "cv_simple": rg.kfold_cv(X1, ly, k=5, seed=config.SEED),
        "cv_multi": rg.kfold_cv(X2, ly, k=5, seed=config.SEED),
        "holdout_multi": rg.holdout(X2, ly, seed=config.SEED),
        "normalised_preview": pd.DataFrame({"country": cs["country"], "gdp_pc": cs["gdp_per_capita"],
                                            "gdp_pc_minmax": rg.min_max(cs["gdp_per_capita"])}).head(8),
    }


# ---------------------------------------------------------------- Module 5

def outliers_lof(df: pd.DataFrame, k: int = 10) -> dict:
    cs = data.cross_section(df, G, ["co2_per_capita", "gdp_per_capita"])
    cs = cs.assign(log_co2_pc=np.log10(cs["co2_per_capita"]), log_gdp_pc=np.log10(cs["gdp_per_capita"]))
    table = lofm.lof_table(cs, ["log_gdp_pc", "log_co2_pc"], k=k)
    table = table.merge(cs[["country", "co2_per_capita", "gdp_per_capita", "continent"]], on="country")
    iqr_set = set(ds.iqr_outliers(cs, "co2_per_capita")["country"])
    top = table.head(10)
    return {"k": k, "n": int(len(table)), "table": table, "threshold": 1.5,
            "n_flagged": int((table["lof"] > 1.5).sum()),
            "iqr_outliers": sorted(iqr_set), "lof_top10_also_iqr": sorted(set(top["country"]) & iqr_set)}


# ---------------------------------------------------------------- Module 6

def time_series(df: pd.DataFrame) -> dict:
    w = data.world_series(df, start=1950)
    comp = ts.holdout_comparison(w, train_end=2019, k=3, trend_start=2000)
    full = w.loc[2000:]
    fit = rg.simple_regression(full.index.to_numpy(float), full.to_numpy())
    future = pd.Series({yr: fit["slope"] * yr + fit["intercept"] for yr in range(Y + 1, 2031)})
    yearly = w.diff()
    return {
        "series": w, "cma5": pd.Series(ts.centred_moving_average(w.to_numpy(), 5), index=w.index),
        "comparison": comp["table"], "detail": comp["detail"], "trend_fit_train": comp["trend_fit"],
        "trend_full": {"slope": fit["slope"], "intercept": fit["intercept"], "r2": fit["metrics"]["R2"]},
        "projection": future,
        "mean_annual_change_2000_latest": float(yearly.loc[2001:].mean()),
        "decomposition_note": "Annual data: no seasonal component exists; only the trend (centred MA) is estimated.",
    }


def quality_summary(df: pd.DataFrame) -> dict:
    from . import quality
    return quality.report(data.load_raw(), df)


def compute_all(df: pd.DataFrame) -> dict:
    return {
        "quality": quality_summary(df),
        "global_trend": global_trend(df),
        "top": top_emitters(df),
        "changes": changes(df),
        "distribution": per_capita_distribution(df),
        "classification": classification_and_tabulation(df),
        "univariate_extras": univariate_extras(df),
        "correlation_forms": correlation_forms(df),
        "diagrams": diagram_data(df),
        "relationships": relationships(df),
        "inference": inference(df),
        "regression": regression(df),
        "lof": outliers_lof(df),
        "time_series": time_series(df),
    }
