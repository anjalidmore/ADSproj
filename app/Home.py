"""Overview — run with:  uv run streamlit run app/Home.py"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import GREEN, LIME, all_results, config, fmt, kpi, note, page

page("Global CO₂ Emissions Analytics",
     "An Applied Data Science exploration of country-level carbon emissions · Our World in Data CO₂ dataset · "
     f"{config.LATEST_YEAR} snapshot")

res = all_results()
g, q, top, ch = res["global_trend"], res["quality"], res["top"], res["changes"]
dist = res["distribution"]

c1, c2, c3, c4 = st.columns(4)
kpi(c1, f"{g['co2_latest'] / 1000:.1f} Gt", f"World CO₂ emissions, {config.LATEST_YEAR}",
    f"{g['multiple_1950_latest']:.1f}× the 1950 level")
kpi(c2, f"{g['change_1990_latest_pct']:+.0f}%", f"Change since 1990", f"from {g['co2_1990'] / 1000:.1f} Gt")
kpi(c3, f"{g['per_capita_latest']:.2f} t", "CO₂ per person (world)",
    f"median country: {dist['summary']['median']:.2f} t")
kpi(c4, f"{top['top3_share_pct']:.0f}%", "Share of China + US + India",
    f"top 10 countries: {top['top_n_share_pct']:.0f}%")

st.markdown("&nbsp;")
left, right = st.columns([3, 2])
with left:
    st.subheader("How have global emissions changed?")
    start = st.slider("Start year", 1750, 2000, 1850, step=10)
    w = g["series"].loc[start:]
    fig = go.Figure(go.Scatter(x=w.index, y=w.values / 1000, mode="lines", line={"color": GREEN, "width": 3},
                               fill="tozeroy", fillcolor="rgba(31,77,58,0.10)",
                               hovertemplate="%{x}: %{y:.2f} Gt<extra></extra>"))
    for yr, txt, dx in [(2009, "Financial crisis", -70), (2020, "COVID-19", -20)]:
        if yr >= start:
            fig.add_annotation(x=yr, y=w.loc[yr] / 1000, text=txt, showarrow=True, arrowhead=0, ax=dx, ay=-45)
    fig.update_layout(height=380, yaxis_title="Gt CO₂ per year", xaxis_title=None,
                      title=f"World fossil-fuel & industry CO₂, {start}–{config.LATEST_YEAR}")
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Who emits the most?")
    t = top["table"].iloc[::-1]
    fig = px.bar(t, x="share_pct", y="country", orientation="h", text=t["share_pct"].map(lambda v: f"{v:.1f}%"),
                 color_discrete_sequence=[GREEN], hover_data={"co2": ":,.0f", "co2_per_capita": ":.2f"},
                 labels={"share_pct": "% of world CO₂", "country": "", "co2": "Mt", "co2_per_capita": "t/person"})
    fig.update_layout(height=380, title=f"Top 10 emitters, share of world total ({config.LATEST_YEAR})")
    st.plotly_chart(fig, width="stretch")

st.subheader("Research questions & what the data says")
tr = top["table"]
india = tr[tr["country"] == "India"].iloc[0]
us = tr[tr["country"] == "United States"].iloc[0]
inf, rel, reg, tsr = res["inference"], res["relationships"], res["regression"], res["time_series"]
best = tsr["comparison"]["MASE"].idxmin()
findings = [
    ("1. How have global CO₂ emissions changed over time?",
     f"World emissions reached a record {g['co2_latest'] / 1000:.1f} Gt in {config.LATEST_YEAR}, {g['change_1990_latest_pct']:.0f}% above 1990. "
     f"The only large fall was 2020 ({g['covid_drop_pct']:.1f}%), fully reversed by {g['rebound_2021_pct']:+.1f}% in 2021."),
    ("2. Which countries contribute the largest shares?",
     f"Three countries emit {top['top3_share_pct']:.0f}% of the world total; the top ten emit {top['top_n_share_pct']:.0f}%."),
    ("3. How do total and per-person emissions differ?",
     f"India is the #{int(india['rank_total'])} emitter in total but ranks #{int(india['rank_per_capita'])} per person "
     f"({india['co2_per_capita']:.2f} t); the US is #{int(us['rank_total'])} in total and #{int(us['rank_per_capita'])} per person ({us['co2_per_capita']:.1f} t)."),
    (f"4. Which countries changed most since {config.BASE_YEAR}?",
     f"China added {ch['abs_increase'].iloc[0]['abs_change']:,.0f} Mt. Among countries emitting ≥10 Mt in {config.BASE_YEAR}, the largest percentage falls were "
     f"{', '.join(ch['pct_decrease']['country'].head(3))}. {ch['n_decreased']} of {ch['n_pairs']} countries emit less than in {config.BASE_YEAR}."),
    ("5. How are per-person emissions distributed?",
     f"Strongly right-skewed: mean {dist['summary']['mean']:.2f} t vs median {dist['summary']['median']:.2f} t, β₂ = {dist['summary']['beta2']:.1f}. "
     "The logarithm is close to normal."),
    ("6. Which countries are unusual?",
     f"IQR rule: {dist['box']['n_high_outliers']} high per-person emitters (mostly oil- and gas-exporting states). LOF (two variables) also flags "
     "low emitters for their income level."),
    ("7. What relates to emissions?",
     f"log GDP per person and log CO₂ per person: r = {rel['gdp_log']['r']:.2f}. Energy use per person explains even more "
     f"(multiple R² = {reg['multiple']['r2']:.2f}), and once energy is included GDP is no longer significant. Association, not causation."),
    ("8. Do continents differ, and has the average country changed?",
     f"Yes: one-way ANOVA F = {inf['anova']['F']:.1f} (p < 0.001). Paired t-test: mean per-person CO₂ fell from "
     f"{inf['paired']['mean_before']:.2f} t to {inf['paired']['mean_after']:.2f} t since {config.BASE_YEAR} (p = {inf['paired']['p_value']:.3f})."),
    ("9. How sensitive are the findings?",
     f"Conclusions about 'the average country' differ from 'the average person' (simple mean {dist['summary']['mean']:.2f} t vs population-weighted "
     f"{dist['population_weighted_mean']:.2f} t). For forecasting the 2020–24 hold-out, the best method by MASE was: {best}."),
]
for qn, ans in findings:
    with st.expander(qn, expanded=False):
        st.write(ans)

note(f"<b>Data:</b> {q['rows']:,} rows × {q['columns']} columns, {q['year_min']}–{q['year_max']}, {q['entities']} entities of which "
     f"{q['countries']} are countries. Source: Our World in Data CO₂ dataset (Global Carbon Project and others), CC BY 4.0. "
     "Every number on this page is computed from the raw file at start-up. Nothing is typed in by hand.")
