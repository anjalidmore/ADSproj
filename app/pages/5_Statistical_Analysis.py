import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/common.py importable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (CONTINENT_COLORS, GREEN, LIME, SEQ, all_results, clean_df, config, countries_df, data, fmt, label,
                    note, page, year_options)
from co2ads import correlation as cr
from co2ads import inference as inf

page("Statistical Analysis", "Module 2.4 multivariate exploration & Module 3 inferential statistics")

df = clean_df()
res = all_results()
t_corr, t_dist, t_ci, t_tests, t_anova, t_err = st.tabs(
    ["Correlation", "Normal & CLT", "Confidence intervals", "z / t tests", "ANOVA", "Type I/II errors & Poisson"])

# ------------------------------------------------------------------ correlation
with t_corr:
    c1, c2, c3, c4 = st.columns([1, 1.4, 1.4, 1])
    year = c1.selectbox("Year", list(range(config.GDP_YEAR, 1989, -1)), key="cy")
    xv = c2.selectbox("X", ["gdp_per_capita", "energy_per_capita", "population", "co2_per_unit_energy"], format_func=label)
    yv = c3.selectbox("Y", ["co2_per_capita", "co2", "co2_per_gdp"], format_func=label)
    logs = c4.toggle("log₁₀ both", value=True)
    cs = data.cross_section(df, year, [xv, yv])
    if logs:
        cs = cs[(cs[xv] > 0) & (cs[yv] > 0)]
    if len(cs) < 10:
        st.warning("Too few countries with both variables.")
    else:
        X = np.log10(cs[xv]) if logs else cs[xv]
        Yv = np.log10(cs[yv]) if logs else cs[yv]
        a = cr.analyse(X, Yv)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("n (complete pairs)", a["n"])
        k2.metric("Pearson r", fmt(a["r"], 3), a["degree"], delta_color="off")
        k3.metric("r²", fmt(a["r_squared"], 3))
        k4.metric("Covariance", f"{a['covariance']:,.4g}")
        k5.metric("Probable error", fmt(a["probable_error"], 4), "significant (|r| > 6·PE)" if a["significant_6pe"] else "not significant", delta_color="off")
        st.caption(f"Three course methods give the same r: actual mean {a['r']:.6f} · direct {a['r_direct']:.6f} · assumed mean {a['r_assumed']:.6f}")
        fig = px.scatter(cs.assign(_x=X, _y=Yv), x="_x", y="_y", color="continent", size="population", size_max=45,
                         hover_name="country", color_discrete_map=CONTINENT_COLORS,
                         labels={"_x": ("log₁₀ " if logs else "") + label(xv), "_y": ("log₁₀ " if logs else "") + label(yv)})
        cp = (X.mean(), Yv.mean())
        fig.add_scatter(x=[cp[0]], y=[cp[1]], mode="markers", marker={"symbol": "x", "size": 16, "color": "#1d211e"}, name="central data point")
        fig.update_layout(height=520, title=f"Bubble chart (size = population), {year}: r = {a['r']:.3f}")
        st.plotly_chart(fig, width="stretch")
        note("Correlation measures association only. GDP, energy use and emissions move together because richer "
             "economies use more energy, mostly from fossil fuels. That does not show GDP causes emissions.")

    st.markdown("#### Correlation matrix & scatter matrix")
    cols = ["co2_per_capita", "gdp_per_capita", "energy_per_capita", "co2_per_unit_energy", "population", "co2"]
    base = data.cross_section(df, year, ["co2_per_capita"])
    full = data.countries(df)
    full = full[full["year"] == year].set_index("country")[cols]
    m = full.loc[base["country"]].dropna()
    mm = np.log10(m.clip(lower=1e-9)) if logs else m
    corr = cr.correlation_matrix(mm, cols)
    fig = px.imshow(corr.round(2), text_auto=True, color_continuous_scale=["#3d7ea6", "#f7f5ef", GREEN], zmin=-1, zmax=1,
                    labels={"color": "r"})
    fig.update_layout(height=460, title=f"Pearson correlation matrix ({'log₁₀ ' if logs else ''}values), n = {len(m)} countries with all six variables")
    st.plotly_chart(fig, width="stretch")
    fig = px.scatter_matrix(mm.reset_index(), dimensions=cols[:4], hover_name="country", color_discrete_sequence=[GREEN])
    fig.update_traces(diagonal_visible=False, marker={"size": 4, "opacity": 0.6})
    fig.update_layout(height=650, title="Scatter matrix (pairs plot)")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ distributions & CLT
with t_dist:
    cs = data.cross_section(df, config.LATEST_YEAR, ["co2_per_capita"])
    x = cs["co2_per_capita"].to_numpy()
    st.markdown(f"**Population:** CO₂ per person of {len(x)} countries in {config.LATEST_YEAR} (strongly right-skewed).")
    c1, c2 = st.columns(2)
    n = c1.slider("Sample size n", 2, 100, 30)
    reps = c2.select_slider("Number of samples", [500, 1000, 5000, 10000], 5000)
    sim = inf.clt_simulation(x, n, reps, seed=config.SEED)
    l, r = st.columns(2)
    fig = px.histogram(x=x, nbins=30, color_discrete_sequence=["#9aa39c"], labels={"x": "t per person"})
    fig.update_layout(height=360, title="Population distribution", showlegend=False)
    l.plotly_chart(fig, width="stretch")
    xs = np.linspace(sim["means"].min(), sim["means"].max(), 200)
    fig = go.Figure()
    fig.add_histogram(x=sim["means"], histnorm="probability density", marker_color=GREEN, name="sample means", nbinsx=50)
    fig.add_scatter(x=xs, y=inf.normal_pdf(xs, sim["population_mean"], sim["theoretical_se"]), line={"color": LIME, "width": 3}, name="N(μ, σ/√n)")
    fig.update_layout(height=360, title=f"Distribution of {reps:,} sample means (n = {n})")
    r.plotly_chart(fig, width="stretch")
    t = pd.DataFrame({"": ["mean", "standard deviation / error", "skewness (3(mean−median)/σ)"],
                      "population": [sim["population_mean"], sim["population_sd"], sim["population_skew"]],
                      "sample means": [sim["mean_of_means"], sim["sd_of_means"], sim["skew_of_means"]],
                      "CLT prediction": [sim["population_mean"], sim["theoretical_se"], 0.0]})
    st.dataframe(t.round(4), hide_index=True, width="stretch")
    note("Increase n: the histogram of sample means becomes narrower (σ/√n) and more symmetric. That is the central limit theorem, "
         "and it is why z-tests on means can work even though the data themselves are skewed.")
    st.markdown("#### z-scores")
    cs = cs.assign(z=inf.z_scores(x))
    st.dataframe(cs.reindex(cs["z"].abs().sort_values(ascending=False).index)[["country", "co2_per_capita", "z"]].head(10).round(3),
                 hide_index=True, width="stretch")
    st.caption("z = (x − μ)/σ: how many standard deviations a country lies from the mean.")

# ------------------------------------------------------------------ CIs
with t_ci:
    c1, c2, c3 = st.columns(3)
    var = c1.selectbox("Variable", ["co2_per_capita", "co2_per_gdp", "energy_per_capita"], format_func=label, key="ci_v")
    yr = c2.selectbox("Year", year_options(var)[::-1], key="ci_y")
    conf = c3.select_slider("Confidence", [0.90, 0.95, 0.99], 0.95)
    sub = data.cross_section(df, yr, [var])
    cont = st.multiselect("Restrict to continents (smaller n → t-interval if n < 30)", sorted(CONTINENT_COLORS), key="ci_c")
    if cont:
        sub = sub[sub["continent"].isin(cont)]
    if len(sub) < 3:
        st.warning("Need at least 3 countries.")
    else:
        ci = inf.ci_mean(sub[var], conf)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("n", ci["n"])
        k2.metric("Sample mean x̄", fmt(ci["mean"], 3))
        k3.metric(f"{int(conf * 100)}% CI", f"{fmt(ci['lower'], 3)} – {fmt(ci['upper'], 3)}")
        k4.metric("Method", f"{ci['method']} (critical {ci['critical']:.3f})")
        st.latex(rf"\bar x \pm {ci['method']}\cdot\frac{{s}}{{\sqrt n}} = {ci['mean']:.3f} \pm {ci['critical']:.3f}\times\frac{{{ci['sd']:.3f}}}{{\sqrt{{{ci['n']}}}}} = {ci['mean']:.3f} \pm {ci['margin']:.3f}")
        b = res["inference"]["bootstrap_median"]
        st.markdown(f"**Bootstrap (Module 1)** 95% CI for the *median* CO₂ per person in {config.LATEST_YEAR}: "
                    f"{fmt(b['lower'], 3)} – {fmt(b['upper'], 3)} t (median {fmt(b['estimate'], 3)}, {b['reps']:,} resamples).")
        note("Caveat for the viva: the ~214 countries are almost the whole population, not a random sample. The interval is "
             "read as uncertainty about the process that generates national emissions (a super-population view), not about "
             "unknown countries.")

# ------------------------------------------------------------------ z / t tests
with t_tests:
    alpha = st.select_slider("Level of significance α", [0.01, 0.05, 0.10], 0.05)
    st.markdown("#### Paired t-test: has the average country changed?")
    c1, c2 = st.columns(2)
    y0 = c1.number_input("Before year", 1990, config.LATEST_YEAR - 1, config.BASE_YEAR, key="p0")
    y1 = c2.number_input("After year", 1991, config.LATEST_YEAR, config.LATEST_YEAR, key="p1")
    pc = data.paired_years(df, "co2_per_capita", int(y0), int(y1))
    if y1 > y0 and len(pc) > 2:
        p = inf.t_test_paired(pc[f"co2_per_capita_{y0}"], pc[f"co2_per_capita_{y1}"], alpha=alpha)
        st.latex(rf"H_0: \mu_d = 0,\quad t = \frac{{\bar d}}{{s_d/\sqrt n}} = \frac{{{p['mean_diff']:.4f}}}{{{p['sd_diff']:.4f}/\sqrt{{{p['n']}}}}} = {p['statistic']:.3f}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("n pairs", p["n"])
        k2.metric("t critical (two-tailed)", fmt(p["critical"], 3))
        k3.metric("p-value", f"{p['p_value']:.4f}")
        k4.metric("Decision", p["decision"])
        st.caption(f"{p['n_decreased']} countries decreased, {p['n_increased']} increased. Mean change d̄ = {p['mean_diff']:+.3f} t per person.")
    st.markdown("#### Independent t-test: do two continents differ? (log₁₀ CO₂ per person)")
    c1, c2, c3 = st.columns(3)
    ga = c1.selectbox("Group 1", sorted(CONTINENT_COLORS), index=2)
    gb = c2.selectbox("Group 2", sorted(CONTINENT_COLORS), index=1)
    yr = c3.selectbox("Year", list(range(config.LATEST_YEAR, 1989, -1)), key="it_y")
    cs = data.cross_section(df, yr, ["co2_per_capita"])
    cs = cs[cs["co2_per_capita"] > 0]
    xa = np.log10(cs.loc[cs["continent"] == ga, "co2_per_capita"])
    xb = np.log10(cs.loc[cs["continent"] == gb, "co2_per_capita"])
    if ga == gb:
        st.info("Pick two different continents.")
    else:
        it = inf.t_test_independent(xa, xb, alpha=alpha)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"{ga} vs {gb} (n)", f"{it['n1']} vs {it['n2']}")
        k2.metric("t (pooled)", fmt(it["statistic"], 3), f"df = {it['df']}", delta_color="off")
        k3.metric("p-value", f"{it['p_value']:.4f}")
        k4.metric("Decision", it["decision"])
        from scipy import stats
        st.caption(f"Geometric means: {ga} {10 ** xa.mean():.2f} t, {gb} {10 ** xb.mean():.2f} t. Variance ratio {it['variance_ratio']:.1f}: "
                   f"if far above ~2–3, the pooled assumption is doubtful; Welch's t gives p = {stats.ttest_ind(xa, xb, equal_var=False).pvalue:.4f}.")
    st.markdown("#### z-test for a single mean (large sample)")
    z = res["inference"]["z_test"]
    st.latex(rf"H_0: \mu = {z['mu0']:.3f}\ (\text{{world per-person value}}),\quad Z = \frac{{\bar x-\mu_0}}{{s/\sqrt n}} = \frac{{{z['mean']:.3f}-{z['mu0']:.3f}}}{{{z['sd']:.3f}/\sqrt{{{z['n']}}}}} = {z['statistic']:.3f}")
    st.write(f"|Z| = {abs(z['statistic']):.3f} vs critical {z['critical']:.2f} → **{z['decision']}** (p = {z['p_value']:.3f}).")

# ------------------------------------------------------------------ ANOVA
with t_anova:
    st.markdown("#### One-way ANOVA: log₁₀ CO₂ per person by continent")
    yr = st.selectbox("Year", list(range(config.LATEST_YEAR, 1989, -1)), key="an_y")
    cs = data.cross_section(df, yr, ["co2_per_capita"]).dropna(subset=["continent"])
    cs = cs[cs["co2_per_capita"] > 0].assign(log_pc=lambda d: np.log10(d["co2_per_capita"]))
    an = inf.one_way_anova({k: v["log_pc"].to_numpy() for k, v in cs.groupby("continent")})
    st.dataframe(an["table"].round(4), width="stretch")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("F", fmt(an["F"], 2))
    k2.metric("F critical (5%)", fmt(an["critical"], 3), f"df = ({an['df_between']}, {an['df_within']})", delta_color="off")
    k3.metric("p-value", f"{an['p_value']:.2e}")
    k4.metric("η² = SS_B / SS_T", fmt(an["eta_squared"], 3))
    st.write(f"**{an['decision']}**: mean log emissions per person differ between continents. Check: SS_T − SS_B − SS_W = {an['ss_check']:.1e}.")
    st.dataframe(an["groups"].round(4), width="stretch")
    fig = px.box(cs, x="continent", y="co2_per_capita", color="continent", color_discrete_map=CONTINENT_COLORS, log_y=True,
                 hover_name="country", points="all", labels={"co2_per_capita": "t per person (log scale)"})
    fig.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    note(f"Groups have different sizes, so the grand mean is the mean of all {int(an['df_between'] + an['df_within'] + 1)} observations "
         f"({an['grand_mean']:.4f}), not the slides' mean of group means ({an['mean_of_means']:.4f}); the two only agree when groups are equal. "
         "Assumptions: normal residuals (reasonable on the log scale) and similar variances (Europe's spread is smaller, so treat p as approximate).")

    st.markdown("#### Two-way ANOVA (without replication): continent × year")
    tw, mat = res["inference"]["two_way"], res["inference"]["two_way_matrix"]
    st.dataframe(mat.round(3), width="stretch")
    st.caption("Cell = population-weighted CO₂ per person of the continent's countries (Σ CO₂ / Σ population).")
    st.dataframe(tw["table"].round(3), width="stretch")
    st.write(f"Continent: F = {tw['F_rows']:.2f} → **{tw['decision_rows']}**; Year: F = {tw['F_columns']:.2f} → **{tw['decision_columns']}**.")
    fig = px.line(mat.T.reset_index().melt("year", var_name="continent", value_name="t per person"), x="year", y="t per person",
                  color="continent", color_discrete_map=CONTINENT_COLORS, markers=True)
    fig.update_layout(height=380, title="Why 'year' is not significant: continents moved in opposite directions")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ errors & Poisson
with t_err:
    st.markdown("#### Type I and Type II errors, simulated on the real data")
    cs = data.cross_section(df, config.LATEST_YEAR, ["co2_per_capita"])
    x = cs["co2_per_capita"].to_numpy()
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Sample size n", 30, 200, 40, step=10)
    shift = c2.slider("True difference for Type II (t)", 0.25, 3.0, 1.5, step=0.25)
    reps = c3.select_slider("Simulations", [1000, 5000, 10000], 5000)
    e = inf.error_rate_simulation(x, n, shift, reps, seed=config.SEED)
    k1, k2, k3 = st.columns(3)
    k1.metric("Type I rate (α = 0.05 nominal)", fmt(e["type1_rate"], 3))
    k2.metric("Type II rate β", fmt(e["type2_rate"], 3))
    k3.metric("Power 1 − β", fmt(e["power"], 3))
    note("Type I: H₀ is true (we test the real mean) but is rejected. Type II: the real mean differs by the chosen amount but "
         "H₀ is accepted. With skewed data and small n the Type I rate exceeds α; it approaches α as n grows (CLT). "
         "Larger n is the only way to reduce both errors together.")
    st.markdown("#### Poisson distribution: are yearly counts of sharp emission drops Poisson?")
    pc_ = res["inference"]["poisson_counts"]
    ps = res["inference"]["poisson"]
    fig = px.bar(x=pc_.index, y=pc_.values, labels={"x": "year", "y": "countries with CO₂ falling > 10%"}, color_discrete_sequence=[GREEN])
    fig.update_layout(height=320)
    st.plotly_chart(fig, width="stretch")
    k1, k2, k3 = st.columns(3)
    k1.metric("Mean count", fmt(ps["mean"]))
    k2.metric("Variance", fmt(ps["variance"]))
    k3.metric("Variance / mean", fmt(ps["dispersion_index"]))
    lam = ps["mean"]
    ks = np.arange(0, int(pc_.max()) + 5)
    obs = np.bincount(pc_.astype(int), minlength=len(ks))[: len(ks)] / len(pc_)
    fig = go.Figure()
    fig.add_bar(x=ks, y=obs, name="observed share of years", marker_color="#cfe1a8")
    fig.add_scatter(x=ks, y=inf.poisson_pmf(ks, lam), name=f"Poisson(μ = {lam:.1f})", line={"color": GREEN, "width": 3})
    fig.update_layout(height=320, title="Observed counts vs Poisson pmf  P(x) = e^(−μ) μ^x / x!")
    st.plotly_chart(fig, width="stretch")
    st.write(f"For a Poisson variable mean = variance. Here variance is {ps['dispersion_index']:.1f}× the mean "
             f"(dispersion test p = {ps['p_value']:.1e}), so the Poisson model is **rejected**: drops come in clusters during global crises.")
