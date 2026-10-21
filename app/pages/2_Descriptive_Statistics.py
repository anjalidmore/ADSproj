import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/common.py importable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (CONTINENT_COLORS, GREEN, LIME, NUMERIC_VARS, clean_df, config, csv_button, data, fmt, label,
                    note, page, year_options)
from co2ads import descriptive as ds
from co2ads import inference as inf
from co2ads import results as R

page("Descriptive Statistics", "Module 2 · univariate exploration with the course formula-sheet methods, recomputed live for your selection")

df = clean_df()
c1, c2, c3, c4 = st.columns([1.6, 1, 1.6, 1])
var = c1.selectbox("Variable", list(NUMERIC_VARS), format_func=label)
years = year_options(var)
if not years:
    st.error("This variable has fewer than 30 countries in every year.")
    st.stop()
year = c2.selectbox("Year", years[::-1], index=0)
conts = c3.multiselect("Continents (empty = all)", sorted(CONTINENT_COLORS))
use_log = c4.toggle("log₁₀ scale", value=False, help="Analyse log₁₀ of the variable (only positive values).")

cs = data.cross_section(df, year, [var])
if conts:
    cs = cs[cs["continent"].isin(conts)]
if use_log:
    cs = cs[cs[var] > 0].assign(**{var: lambda d: np.log10(d[var])})
name = ("log₁₀ " if use_log else "") + label(var)
if len(cs) < 5:
    st.warning("Fewer than 5 countries match this selection; pick more continents or another year.")
    st.stop()

x = cs[var].to_numpy()
s = ds.summary(x)
st.caption(f"{name}, {year}: n = {s['n']} countries (countries only; regional aggregates excluded; missing values dropped for this variable only).")

a, b, c, d, e = st.columns(5)
a.metric("Mean", fmt(s["mean"], 3))
b.metric("Median", fmt(s["median"], 3))
c.metric("Std deviation σ", fmt(s["std"], 3))
d.metric("IQR", fmt(s["iqr"], 3))
e.metric("Skewness (Bowley)", fmt(s["bowley_sk"], 3))

tab_sum, tab_freq, tab_cum, tab_box, tab_norm, tab_mom, tab_disc, tab_open = st.tabs(
    ["All measures", "Frequency distribution & histogram", "Cumulative frequency & ogive", "Box plot & outliers",
     "Distribution chart", "Moments", "Discrete series", "Open-end classes & corrections"])

with tab_sum:
    rows = [
        ("Central tendency", "Arithmetic mean", "X̄ = ΣX / N", s["mean"]),
        ("Central tendency", "Median", "(N+1)/2 th item", s["median"]),
        ("Central tendency", "Mode (grouped)", "L + Δ1/(Δ1+Δ2)·i", s["grouped_mode"]),
        ("Dispersion", "Minimum", "S", s["min"]),
        ("Dispersion", "Maximum", "L", s["max"]),
        ("Dispersion", "Range", "L − S", s["range"]),
        ("Dispersion", "Coefficient of range", "(L − S)/(L + S)", s["coef_range"]),
        ("Dispersion", "Q1", "(N+1)/4 th item", s["q1"]),
        ("Dispersion", "Q3", "3(N+1)/4 th item", s["q3"]),
        ("Dispersion", "Interquartile range", "Q3 − Q1", s["iqr"]),
        ("Dispersion", "Quartile deviation", "(Q3 − Q1)/2", s["quartile_deviation"]),
        ("Dispersion", "Coefficient of QD", "(Q3 − Q1)/(Q3 + Q1)", s["coef_qd"]),
        ("Dispersion", "D1 / P10", "(N+1)/10 th item", s["d1"]),
        ("Dispersion", "D9 / P90", "9(N+1)/10 th item", s["d9"]),
        ("Dispersion", "Variance σ²", "Σ(X − X̄)²/N", s["variance"]),
        ("Dispersion", "Standard deviation σ", "√[Σ(X − X̄)²/N]", s["std"]),
        ("Dispersion", "Sample SD s", "√[Σ(X − X̄)²/(n−1)]", s["std_sample"]),
        ("Dispersion", "Coefficient of variation %", "σ / X̄ × 100", s["coef_variation_pct"]),
        ("Shape", "Karl Pearson Sk (mode)", "(X̄ − Mode)/σ", s["pearson_sk_mode"]),
        ("Shape", "Karl Pearson Sk (median)", "3(X̄ − Median)/σ", s["pearson_sk_median"]),
        ("Shape", "Bowley Sk", "(Q3 + Q1 − 2·Med)/(Q3 − Q1)", s["bowley_sk"]),
        ("Shape", "Kurtosis β₂", "μ4 / μ2²", s["beta2"]),
    ]
    table = pd.DataFrame(rows, columns=["group", "measure", "formula (course sheet)", "value"])
    st.dataframe(table.assign(value=table["value"].map(lambda v: f"{v:,.4f}")), width="stretch", hide_index=True, height=600)
    csv_button(table, f"descriptive_{var}_{year}.csv")
    note(f"<b>Interpretation:</b> the distribution is <b>{ds.skew_direction(s['pearson_sk_median'])}</b> "
         f"(mean {fmt(s['mean'])} vs median {fmt(s['median'])}); kurtosis: {s['kurtosis_label']}.")
    if not use_log and var.endswith("per_capita") and "population" in cs:
        wm = ds.weighted_mean(cs[var], cs["population"])
        st.markdown(f"**Weighted mean** (weights = population, ΣWX/ΣW): **{fmt(wm, 3)}** vs simple mean {fmt(s['mean'], 3)}. "
                    "The simple mean describes the typical *country*; the weighted mean describes the typical *person*.")

with tab_freq:
    default_w = float(s["class_width"])
    w = st.number_input("Class width (default ≈ range/√n, rounded)", min_value=default_w / 20, value=default_w, step=default_w / 4, format="%.4g")
    ft = ds.frequency_table(x, width=w)
    g = ds.grouped_mean(ft)
    gm = ds.grouped_mode(ft)
    st.dataframe(ft.round(4), width="stretch", hide_index=True, height=300)
    csv_button(ft, f"frequency_table_{var}_{year}.csv")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Grouped mean (step deviation)", fmt(g["step_deviation"], 3), help=f"A = {g['assumed_mean']:g}")
    k2.metric("Grouped median", fmt(ds.grouped_median(ft), 3))
    k3.metric("Grouped mode", fmt(gm["mode"], 3), help=f"Δ1 = {gm['delta1']}, Δ2 = {gm['delta2']}")
    k4.metric("Grouped σ (step deviation)", fmt(ds.grouped_std(ft), 3))
    fig = go.Figure()
    fig.add_bar(x=ft["mid"], y=ft["f"], width=w * 0.96, marker_color=GREEN, name="frequency",
                customdata=np.c_[ft["lower"], ft["upper"]], hovertemplate="%{customdata[0]:.3g}–%{customdata[1]:.3g}: %{y}<extra></extra>")
    fig.add_scatter(x=ft["mid"], y=ft["f"], mode="lines+markers", line={"color": LIME, "width": 2}, name="frequency polygon")
    fig.add_vline(s["mean"], line_dash="dash", line_color="#c0392b", annotation_text="mean")
    fig.add_vline(s["median"], line_dash="dot", line_color="#1d211e", annotation_text="median")
    fig.update_layout(height=420, title=f"Histogram & frequency polygon: {name}, {year}", xaxis_title=name, yaxis_title="number of countries")
    st.plotly_chart(fig, width="stretch")
    st.caption("Grouped values differ slightly from the exact (individual-series) values because grouping assumes every value sits at its class mid-point.")

with tab_box:
    b = ds.box_plot(x)
    by_cont = st.toggle("Split by continent", value=False)
    fig = px.box(cs.dropna(subset=["continent"]) if by_cont else cs, x="continent" if by_cont else None, y=var, points="outliers",
                 hover_name="country", color="continent" if by_cont else None,
                 color_discrete_map=CONTINENT_COLORS, labels={var: name})
    fig.update_traces(boxmean=True)
    if not by_cont:
        fig.update_traces(marker_color=GREEN)
    fig.update_layout(height=460, title=f"Box plot (whiskers at 1.5 × IQR), {year}")
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"Lower limit = Q1 − 1.5·IQR = **{fmt(b['lower_limit'], 3)}**, upper limit = Q3 + 1.5·IQR = **{fmt(b['upper_limit'], 3)}**. "
                f"Flagged: **{b['n_high_outliers']}** high, **{b['n_low_outliers']}** low.")
    out = ds.iqr_outliers(cs, var)[list(dict.fromkeys(["country", "continent", var, "population"]))]
    st.dataframe(out, width="stretch", hide_index=True)
    note("These are treated as <b>genuine extreme values</b>, not errors: they are plausible (small populations with large "
         "oil, gas or industrial sectors), so they are kept and reported rather than deleted. "
         "Their influence is reduced by using medians, IQR and log scales.")

with tab_norm:
    mu, sg = x.mean(), x.std()
    xs = np.linspace(x.min(), x.max(), 300)
    fig = go.Figure()
    fig.add_histogram(x=x, histnorm="probability density", marker_color="#cfe1a8", name="data", nbinsx=int(np.sqrt(len(x))) + 1)
    fig.add_scatter(x=xs, y=inf.normal_pdf(xs, mu, sg), line={"color": GREEN, "width": 3}, name="Normal(μ, σ) curve")
    for k in (1, 2, 3):
        fig.add_vrect(x0=mu - k * sg, x1=mu + k * sg, fillcolor=GREEN, opacity=0.04, line_width=0)
    fig.update_layout(height=420, title=f"Distribution chart: data vs fitted normal curve ({name})", xaxis_title=name)
    st.plotly_chart(fig, width="stretch")
    er = inf.empirical_rule(x)
    st.dataframe(er.round(2), width="stretch", hide_index=True)
    st.caption("Empirical rule check: a normal distribution has 68.27 / 95.45 / 99.73 % of values within ±1/2/3σ. "
               "Turn on the log₁₀ scale above to see per-person emissions become close to normal.")

with tab_cum:
    ft = ds.frequency_table(x)
    cum = ds.cumulative_frequencies(ft)
    n = int(ft["f"].sum())
    med_x, med_y = ds.ogive_intersection(ft)
    c_tab, c_fig = st.columns([1, 2])
    c_tab.dataframe(pd.DataFrame({"less than": cum["upper"], "cf (less than)": cum["less_than_cf"],
                                  "more than": cum["lower"], "cf (more than)": cum["more_than_cf"]}).round(4),
                    hide_index=True, width="stretch", height=420)
    xs_lt = np.r_[ft["lower"].iloc[0], ft["upper"].to_numpy()]
    fig = go.Figure()
    fig.add_scatter(x=xs_lt, y=np.r_[0, cum["less_than_cf"].to_numpy()], mode="lines+markers", name="'less than' ogive",
                    line={"color": GREEN, "width": 3})
    fig.add_scatter(x=np.r_[ft["lower"].to_numpy(), ft["upper"].iloc[-1]], y=np.r_[cum["more_than_cf"].to_numpy(), 0],
                    mode="lines+markers", name="'more than' ogive", line={"color": LIME, "width": 3})
    fig.add_scatter(x=[med_x], y=[med_y], mode="markers", marker={"size": 13, "color": "#c0392b", "symbol": "x"},
                    name=f"intersection: median = {med_x:.3f}")
    for k, lab in [(1, "Q1"), (3, "Q3")]:
        q = ds.ogive_value(ft, k * n / 4)
        fig.add_annotation(x=q, y=k * n / 4, text=f"{lab} = {q:.2f}", showarrow=True, arrowhead=0, ax=40, ay=-25)
    fig.update_layout(height=440, title=f"Ogives (cumulative frequency curves): {name}, {year}", xaxis_title=name,
                      yaxis_title="cumulative number of countries")
    c_fig.plotly_chart(fig, width="stretch")
    st.markdown(f"The two ogives cross at **N/2 = {med_y:g}**, giving the median **{med_x:.4f}**. The formula "
                f"L + (N/2 − cf)/f × i gives **{ds.grouped_median(ft):.4f}**: the same value, because the ogive interpolates "
                "linearly inside the median class, exactly as the formula does. Quartiles are read the same way at N/4 and 3N/4.")

with tab_mom:
    m = ds.moments(x)
    l, r = st.columns(2)
    l.markdown("**Moments about the origin** μ′ᵣ = ΣXʳ/N")
    l.dataframe(pd.DataFrame({"moment": list(m["raw"]), "value": list(m["raw"].values())}), hide_index=True, width="stretch")
    r.markdown("**Moments about the mean** μᵣ = Σ(X − X̄)ʳ/N")
    r.dataframe(pd.DataFrame({"moment": list(m["central"]), "direct": list(m["central"].values()),
                              "from raw moments": [0.0, *m["central_from_raw"].values()]}), hide_index=True, width="stretch")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("β₁ = μ₃²/μ₂³", fmt(m["beta1"], 4))
    k2.metric("γ₁ = ±√β₁ (skewness)", fmt(m["gamma1"], 4))
    k3.metric("β₂ = μ₄/μ₂² (kurtosis)", fmt(m["beta2"], 4))
    k4.metric("γ₂ = β₂ − 3 (excess)", fmt(m["gamma2"], 4))
    note("μ₁ (about the mean) is always 0; μ₂ is the variance; μ₃ measures asymmetry (its sign gives the direction of skew); "
         "μ₄ measures peakedness and tail weight. Central moments are computed directly and again from the raw moments "
         "(μ₂ = μ′₂ − μ′₁², μ₃ = μ′₃ − 3μ′₂μ′₁ + 2μ′₁³, μ₄ = μ′₄ − 4μ′₃μ′₁ + 6μ′₂μ′₁² − 3μ′₁⁴): the two columns agree.")
    em = ds.empirical_mode(x)
    st.markdown(f"**Empirical relation** Mode ≈ 3·Median − 2·Mean = **{em:.3f}** vs grouped-formula mode **{s['grouped_mode']:.3f}**. "
                "The relation is only meant for moderately skewed data; a large gap is a sign of strong skewness.")

with tab_disc:
    st.markdown("A **discrete series** takes only whole-number values. From the data we build one: for each country, the "
                "number of years in a window in which its CO₂ emissions fell compared with the year before.")
    c1, c2 = st.columns(2)
    w0 = c1.number_input("Window start", 1950, config.LATEST_YEAR - 5, config.BASE_YEAR)
    w1 = c2.number_input("Window end", int(w0) + 5, config.LATEST_YEAR, config.LATEST_YEAR)
    yod = R.years_of_decline(df, int(w0), int(w1))
    if len(yod) < 5:
        st.warning("Too few countries have complete data for this window.")
    else:
        dt = ds.discrete_table(yod)
        dsum = ds.discrete_summary(dt)
        dsd = ds.discrete_std(dt)
        l, r = st.columns([1, 2])
        l.dataframe(dt.round(2), hide_index=True, width="stretch", height=430)
        fig = go.Figure(go.Bar(x=dt["X"], y=dt["f"], marker_color=GREEN, width=0.35))
        fig.update_layout(height=430, title=f"Discrete frequency distribution (line-style bars), n = {dsum['n']} countries",
                          xaxis_title=f"years with falling CO₂, {int(w0) + 1}–{int(w1)}", yaxis_title="countries (f)")
        r.plotly_chart(fig, width="stretch")
        rows = [("Mean (direct)", "ΣfX / N", dsum["mean"]), ("Mean (short-cut)", "A + Σfd / N", dsum["mean_shortcut"]),
                ("Median", "X at cf ≥ (N+1)/2", dsum["median"]), ("Mode", "X with highest f", ", ".join(f"{v:g}" for v in dsum["mode"])),
                ("Q1 / Q3", "X at cf ≥ (N+1)/4, 3(N+1)/4", f"{dsum['q1']:g} / {dsum['q3']:g}"),
                ("D1 / D9", "X at cf ≥ k(N+1)/10", f"{dsum['d1']:g} / {dsum['d9']:g}"),
                ("Range / IQR / QD", "L − S, Q3 − Q1, (Q3 − Q1)/2", f"{dsum['range']:g} / {dsum['iqr']:g} / {dsum['quartile_deviation']:g}"),
                ("σ (actual mean method)", "√(Σfx²/N)", dsd["actual_mean_method"]),
                ("σ (assumed mean method)", "√(Σfd²/N − (Σfd/N)²)", dsd["assumed_mean_method"]),
                ("Coefficient of variation", "σ / X̄ × 100", f"{dsum['coef_variation_pct']:.1f}%")]
        st.dataframe(pd.DataFrame(rows, columns=["measure", "discrete-series formula", "value"]).astype(str), hide_index=True, width="stretch")
        st.caption("For a discrete series the course reads the median and quartiles directly from the cf column (no interpolation): "
                   "the value whose cumulative frequency first reaches the position.")
        st.markdown("**Inclusive → exclusive classes.** Grouped into inclusive classes of 4 values (0–3, 4–7, …), the "
                    "classes are made continuous by subtracting 0.5 from each lower limit and adding 0.5 to each upper limit.")
        mx = int(yod.max())
        lo = np.arange(0, mx + 1, 4)
        hi = lo + 3
        f_inc = [int(((yod >= a) & (yod <= b)).sum()) for a, b in zip(lo, hi)]
        lo_ex, hi_ex = ds.inclusive_to_exclusive(lo, hi)
        t_inc = ds.class_table(lo_ex, hi_ex, f_inc)
        st.dataframe(pd.DataFrame({"inclusive class": [f"{a}–{b}" for a, b in zip(lo, hi)],
                                   "exclusive class": [f"{a:g}–{b:g}" for a, b in zip(lo_ex, hi_ex)], "f": f_inc}),
                     hide_index=True, width="stretch")
        st.write(f"Grouped median from the exclusive classes: **{ds.grouped_median(t_inc):.3f}** (discrete median {dsum['median']:g}).")

with tab_open:
    st.markdown("**Open-end classes.** Published tables often end with a class such as “15 t and above”. The mean then needs "
                "an assumption about the missing limit; the median and quartiles do not.")
    ft = ds.frequency_table(x)
    top_default = 15.0 if (var == "co2_per_capita" and not use_log) else float(np.quantile(x, 0.9))
    # keep at least two closed classes before the open one
    top_default = float(min(max(top_default, ft["lower"].iloc[min(2, len(ft) - 1)]), x.max()))
    open_from = st.number_input("Open the last class from", float(np.round(x.min(), 2)), float(x.max()), top_default)
    if open_from <= ft["lower"].iloc[1]:
        st.warning("Choose a value above the second class's lower limit.")
    else:
        ft_open = ds.close_top_classes(ft, ft["lower"][ft["lower"] >= open_from].min() if (ft["lower"] >= open_from).any() else ft["lower"].iloc[-1])
        st.dataframe(ft_open.assign(upper=ft_open["upper"].replace(np.inf, "and above")).astype({"upper": str}).round(3),
                     hide_index=True, width="stretch")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Mean, open class assumed same width", fmt(ds.open_end_mean(ft_open), 3))
        k2.metric("Grouped mean, all classes closed", fmt(ds.grouped_mean(ft)["direct"], 3))
        k3.metric("Exact mean", fmt(s["mean"], 3))
        k4.metric("Median (open table)", fmt(ds.grouped_median(ft_open), 3))
        st.caption("The mean moves with the assumed width of the open class; the median is unchanged as long as it does not fall in the "
                   "open class. With skewed data and an open top class, the course advises reporting the median.")
    st.markdown("**Correcting incorrect items.** Suppose the regional and income-group rows (and World) had been averaged in "
                "with the countries by mistake. The correction formula removes them from ΣX and N.")
    if use_log:
        st.info("Turn off the log₁₀ scale to see this example.")
    else:
        allr = df[(df["year"] == year) & df[var].notna()]
        wrong = allr.loc[allr["entity_type"] != "country", var]
        if wrong.empty:
            st.info("No aggregate rows report this variable in this year.")
        else:
            inc = float(allr[var].mean())
            cm = ds.corrected_mean(len(allr), inc, wrong=wrong.tolist())
            st.latex(rf"\Sigma X_{{correct}} = N\bar X - \Sigma X_{{wrong}} = {len(allr)}\times{inc:.4f} - {wrong.sum():,.4f} = {cm['correct_sum']:,.4f}")
            st.write(f"Incorrect mean with {len(wrong)} aggregate rows: **{inc:.4f}**. Corrected mean over {cm['n']} countries: "
                     f"**{cm['correct_mean']:.4f}** (countries-only mean computed directly: {data.cross_section(df, year, [var])[var].mean():.4f}).")
