import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/common.py importable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import CHAR, CONTINENT_COLORS, GREEN, LIME, all_results, config, csv_button, fmt, note, page

page("Classification & Diagrams", "Module 2.2 · classification, tabulation, and diagrammatic & graphic representation of the CO₂ data")

res = all_results()
cl, dg, cf = res["classification"], res["diagrams"], res["correlation_forms"]
Y, B = config.LATEST_YEAR, config.BASE_YEAR
FUEL_COLORS = ["#1d211e", GREEN, LIME, "#d98c3f", "#3d7ea6", "#9aa39c"]

t_class, t_tab, t_1d, t_2d, t_graph, t_corr = st.tabs(
    ["Classification", "Tabulation", "One-dimensional diagrams", "Two-dimensional & pictorial", "Graphs", "Types of correlation"])

# ------------------------------------------------------------------ classification
with t_class:
    st.markdown(f"Classification groups related facts into classes that share a characteristic. The course names four types; "
                f"each is applied to the {Y} country data below ({cl['n']} countries with a continent).")
    st.subheader("Geographical (area-wise): by continent")
    geo = cl["geographical"].reset_index()
    st.dataframe(geo.round(2), hide_index=True, width="stretch")
    st.caption("Summary table ranked by size. Continental CO₂ per person = Σ CO₂ / Σ population of the continent's countries.")
    st.subheader("Chronological (time-wise): world CO₂ by decade")
    st.dataframe(cl["chronological"].round(1).reset_index(), hide_index=True, width="stretch")
    st.caption("Listed earliest → latest. The last period is incomplete, so compare the mean per year, not the total.")
    st.subheader("Qualitative (by attribute)")
    l, r = st.columns([1, 2])
    l.markdown(f"**Simple (two-fold):** did the country's CO₂ fall between {B} and {Y}?")
    l.dataframe(cl["two_fold"].reset_index(), hide_index=True, width="stretch")
    r.markdown("**Manifold:** continent → fell or rose → largest reported fuel source (each split subdivides the last)")
    r.dataframe(cl["manifold"], width="stretch")
    st.subheader("Quantitative (by magnitude): CO₂ per person in classes")
    st.dataframe(cl["quantitative"].reset_index(), hide_index=True, width="stretch")
    st.caption("The full √n-rule frequency distribution is on the Descriptive Statistics page.")

# ------------------------------------------------------------------ tabulation
with t_tab:
    st.markdown("Every table has a **stub** (row headings), **captions** (column headings), a **body** and **totals**.")
    st.subheader(f"Simple (one-way) table: countries by CO₂ per person, {Y}")
    st.dataframe(cl["simple_table"].reset_index(), hide_index=True, width="stretch")
    st.subheader("Two-way table: continent × CO₂ per person")
    st.dataframe(cl["two_way_table"], width="stretch")
    csv_button(cl["two_way_table"].reset_index(), "two_way_table.csv")
    st.subheader(f"Higher-order (manifold) table: continent × change since {B} × CO₂ per person")
    st.dataframe(cl["higher_order_table"], width="stretch", height=520)
    csv_button(cl["higher_order_table"].reset_index(), "higher_order_table.csv")
    tw = cl["two_way_table"]
    note(f"Reading the two-way table: {tw.loc['Africa', '< 1 t']} of Africa's {tw.loc['Africa', 'Total']} countries emit under 1 t per person, "
         f"while {tw.loc['Europe', '3–6 t'] + tw.loc['Europe', '6–10 t']} of Europe's {tw.loc['Europe', 'Total']} sit between 3 and 10 t. "
         "The higher-order table adds a third characteristic, so patterns within each continent can be compared.")

# ------------------------------------------------------------------ one-dimensional diagrams
with t_1d:
    mb = dg["multiple_bar"]
    st.subheader("Multiple bar diagram: continental CO₂ in three years")
    long = mb.reset_index().melt("continent", var_name="year", value_name="Mt")
    long["year"] = long["year"].astype(str)
    fig = px.bar(long, x="continent", y="Mt", color="year", barmode="group",
                 color_discrete_sequence=["#cfe1a8", LIME, GREEN], labels={"Mt": "Mt CO₂"})
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")
    st.subheader(f"Sub-divided bar diagram: CO₂ by source, {Y} (Mt)")
    fm = dg["fuel_mt"].reset_index().melt("continent", var_name="source", value_name="Mt")
    fig = px.bar(fm, x="continent", y="Mt", color="source", color_discrete_sequence=FUEL_COLORS)
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Percentage bar diagram: the same, each bar = 100%")
    fp = dg["fuel_pct"].reset_index().melt("continent", var_name="source", value_name="%")
    fig = px.bar(fp, x="continent", y="%", color="source", color_discrete_sequence=FUEL_COLORS, text=fp["%"].map(lambda v: f"{v:.0f}" if v >= 6 else ""))
    fig.update_layout(height=420, yaxis_range=[0, 100])
    st.plotly_chart(fig, width="stretch")
    note("The percentage bars answer a different question from the sub-divided bars: Asia emits the most in total, "
         f"but coal is {dg['fuel_pct'].loc['Asia', 'coal']:.0f}% of Asia's CO₂ against {dg['fuel_pct'].loc['Europe', 'coal']:.0f}% of Europe's.")

    st.subheader("Broken bar diagram: CO₂ per person, top 10")
    pt = dg["per_capita_top"].iloc[::-1]
    second = pt["co2_per_capita"].iloc[:-1].max()
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2], horizontal_spacing=0.03)
    fig.add_bar(x=pt["co2_per_capita"], y=pt["country"], orientation="h", marker_color=GREEN, row=1, col=1, showlegend=False,
                text=pt["co2_per_capita"].map(lambda v: f"{v:.1f}"), textposition="inside")
    fig.add_bar(x=pt["co2_per_capita"], y=pt["country"], orientation="h", marker_color=GREEN, row=1, col=2, showlegend=False)
    fig.update_xaxes(range=[0, second * 1.1], row=1, col=1)
    fig.update_xaxes(range=[pt["co2_per_capita"].max() - 4, pt["co2_per_capita"].max() + 1], row=1, col=2)
    fig.update_layout(height=440, title="The axis is broken between the panels so the smaller bars stay readable; the long bar's value is labelled")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Without the break, {pt['country'].iloc[-1]} ({pt['co2_per_capita'].iloc[-1]:.1f} t) would squeeze every other bar.")

# ------------------------------------------------------------------ two-dimensional and pictorial
with t_2d:
    st.subheader("Rectangular diagram: width = population, height = CO₂ per person, area = total CO₂")
    rc = dg["rectangles"].sort_values("co2_per_capita", ascending=False).reset_index(drop=True)
    widths = rc["population"] / 1e6
    lefts = np.r_[0, widths.cumsum().to_numpy()[:-1]]
    fig = go.Figure()
    for i, row in rc.iterrows():
        fig.add_bar(x=[lefts[i] + widths[i] / 2], y=[row["co2_per_capita"]], width=[widths[i] * 0.98], name=row["country"],
                    marker_color=GREEN if i % 2 == 0 else LIME, showlegend=False,
                    hovertemplate=f"{row['country']}<br>population {widths[i]:,.0f} M<br>{row['co2_per_capita']:.2f} t/person"
                                  f"<br>total {row['co2']:,.0f} Mt<extra></extra>")
        if widths[i] > 80:
            fig.add_annotation(x=lefts[i] + widths[i] / 2, y=row["co2_per_capita"], text=row["country"], showarrow=False, yshift=12)
    fig.update_layout(height=440, xaxis_title="population (millions, cumulative)", yaxis_title="t CO₂ per person",
                      title=f"Top 10 emitters, {Y}: each rectangle's area is the country's total emissions")
    st.plotly_chart(fig, width="stretch")
    st.caption("This one diagram shows all three measures: China's and India's wide rectangles have very different heights.")

    sq = dg["squares"].reset_index(drop=True)
    st.subheader("Square and circle diagrams: total CO₂ of the top 5")
    scale = 1 / sq["side"].max()
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Squares: side ∝ √value", "Circles: r = √(value / π)"])
    x0 = 0.0
    for i, row in sq.iterrows():
        s_ = row["side"] * scale
        fig.add_shape(type="rect", x0=x0, y0=0, x1=x0 + s_, y1=s_, fillcolor=GREEN, line_color=CHAR, row=1, col=1)
        fig.add_annotation(x=x0 + s_ / 2, y=-0.06, yanchor="top", text=f"{row['country']}<br>{row['co2']:,.0f} Mt", showarrow=False, row=1, col=1, font={"size": 10})
        x0 += s_ + 0.22
    cx = 0.0
    rscale = 0.5 / sq["radius"].max()
    for i, row in sq.iterrows():
        rr = row["radius"] * rscale
        fig.add_shape(type="circle", x0=cx, y0=0.5 - rr, x1=cx + 2 * rr, y1=0.5 + rr, fillcolor=LIME, line_color=CHAR, row=1, col=2)
        fig.add_annotation(x=cx + rr, y=0.5 - rr - 0.04, yanchor="top", text=row["country"], showarrow=False, row=1, col=2, font={"size": 10})
        cx += 2 * rr + 0.16
    fig.update_xaxes(visible=False, range=[-0.05, x0 + 0.05], row=1, col=1)
    fig.update_xaxes(visible=False, range=[-0.05, cx + 0.05], row=1, col=2)
    fig.update_yaxes(visible=False, range=[-0.3, 1.05], scaleanchor="x", row=1, col=1)
    fig.update_yaxes(visible=False, range=[-0.3, 1.05], scaleanchor="x2", row=1, col=2)
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(sq.assign(**{"side = √Mt": sq["side"].round(2), "radius = √(Mt/π)": sq["radius"].round(2)})[["country", "co2", "side = √Mt", "radius = √(Mt/π)"]],
                 hide_index=True, width="stretch")
    st.caption("Area carries the value, so sides and radii grow with the square root: China emits 12.8× Japan, but its square's side is only 3.6× longer.")

    st.subheader("Pictogram: one symbol = 1,000 Mt (1 Gt) of CO₂")
    rows = []
    for i, row in sq.iterrows():
        n_full = int(row["co2"] // 1000)
        frac = row["co2"] / 1000 - n_full
        for j in range(n_full):
            rows.append({"country": row["country"], "x": j, "size": 1.0})
        if frac >= 0.05:
            rows.append({"country": row["country"], "x": n_full, "size": frac})
    pg = pd.DataFrame(rows)
    fig = px.scatter(pg, x="x", y="country", size="size", size_max=22, color_discrete_sequence=[GREEN],
                     category_orders={"country": sq["country"].tolist()})
    fig.update_traces(marker={"symbol": "square", "line": {"width": 0}})
    fig.update_layout(height=320, xaxis={"visible": False}, yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    st.caption("Easy to read at a glance, imprecise for fractions (the last symbol is scaled). The world map on the Country & Time page is the cartogram.")

# ------------------------------------------------------------------ graphs
with t_graph:
    w = dg["world_recent"] / 1000
    st.subheader("Line graph: natural scale from zero vs a false baseline")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Y-axis from zero", "False baseline (axis starts at 24 Gt)"])
    fig.add_scatter(x=w.index, y=w.values, line={"color": GREEN, "width": 3}, showlegend=False, row=1, col=1)
    fig.add_scatter(x=w.index, y=w.values, line={"color": GREEN, "width": 3}, showlegend=False, row=1, col=2)
    fig.update_yaxes(range=[0, w.max() * 1.1], title="Gt CO₂", row=1, col=1)
    fig.update_yaxes(range=[24, w.max() * 1.02], row=1, col=2)
    fig.add_annotation(x=w.index.min(), y=24.3, text="≈≈ axis break", showarrow=False, xanchor="left", row=1, col=2, font={"color": "#c0392b"})
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")
    note("A false baseline (marked with a zig-zag break) shows the year-to-year variation (2009, 2020) clearly and saves space, "
         "but it exaggerates changes. The break must be marked so the reader knows the axis does not start at zero.")
    st.caption("Histogram, frequency polygon and ogives are on the Descriptive Statistics page; the scatter and bubble charts are on Statistical Analysis.")

# ------------------------------------------------------------------ correlation types
with t_corr:
    st.markdown("Correlation by **direction** (positive / negative / none), by **shape** (linear / non-linear) and by "
                "**number of variables** (simple / partial / multiple).")
    d = cf["data"]
    pairs = [
        ("Positive (log GDP vs log CO₂ per person)", d["log_gdp_per_capita"], d["log_co2_per_capita"], cf["positive"], "log₁₀ GDP pc", "log₁₀ CO₂ pc"),
        ("Negative, across countries (log GDP vs log CO₂ per unit energy)", d["log_gdp_per_capita"], d["log_co2_per_unit_energy"], cf["negative_cross_section"], "log₁₀ GDP pc", "log₁₀ kg CO₂/kWh"),
        ("Negative, over time (UK CO₂ vs year)", cf["uk"]["year"], cf["uk"]["co2"], cf["negative_time"], "year", "UK CO₂ (Mt)"),
        ("No correlation (population vs CO₂ per person)", cf["pop_data"]["population"], cf["pop_data"]["co2_per_capita"], cf["none"], "population", "t CO₂ pc"),
        ("Non-linear (raw GDP vs raw CO₂ per person)", d["gdp_per_capita"], d["co2_per_capita"], cf["nonlinear_raw"], "GDP pc (int-$)", "t CO₂ pc"),
    ]
    cols = st.columns(2)
    for i, (title, xx, yy, a, xl, yl) in enumerate(pairs):
        fig = px.scatter(x=xx, y=yy, color_discrete_sequence=[GREEN], labels={"x": xl, "y": yl}, log_x=(xl == "population"))
        fig.update_traces(marker={"size": 6, "opacity": 0.6})
        fig.update_layout(height=320, title=f"{title}<br><sup>r = {a['r']:.3f} · n = {a['n']} · {a['degree']}</sup>")
        cols[i % 2].plotly_chart(fig, width="stretch")
    st.subheader("Simple, partial and multiple correlation (2022, log₁₀ scales)")
    tbl = pd.DataFrame({
        "type": ["Simple", "Simple", "Partial", "Partial", "Multiple"],
        "coefficient": ["r(CO₂, GDP)", "r(CO₂, energy)", "r(CO₂, GDP | energy)", "r(CO₂, energy | GDP)", "R(CO₂ ; GDP, energy)"],
        "value": [cf["simple_r_gdp"], cf["simple_r_energy"], cf["partial_gdp_given_energy"], cf["partial_energy_given_gdp"], cf["multiple_R"]],
    })
    st.dataframe(tbl.round(4), hide_index=True, width="stretch")
    st.latex(r"r_{y1\cdot2} = \frac{r_{y1} - r_{y2}\,r_{12}}{\sqrt{(1-r_{y2}^2)(1-r_{12}^2)}}\qquad R_{y\cdot12}^2 = \frac{r_{y1}^2 + r_{y2}^2 - 2\,r_{y1}r_{y2}r_{12}}{1 - r_{12}^2}")
    note(f"GDP alone correlates {cf['simple_r_gdp']:.2f} with CO₂ per person, but holding energy use constant the partial correlation is "
         f"{cf['partial_gdp_given_energy']:.2f}: almost nothing is left. Energy keeps a partial r of {cf['partial_energy_given_gdp']:.2f}. "
         f"Multiple R = {cf['multiple_R']:.3f} is the square root of the regression R² on the Advanced Analytics page. "
         "The time-series negative correlation (UK) reflects a trend over time, not a relationship between countries.")
