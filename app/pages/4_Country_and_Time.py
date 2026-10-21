import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/common.py importable

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (CONTINENT_COLORS, GREEN, LIME, SEQ, clean_df, config, countries_df, csv_button, data, label, note, page)

page("Country & Time Analysis", "Rankings, total vs per-person emissions, change over time, maps and fuel mix (Module 2 diagrams & graphs)")

df = clean_df()
cdf = countries_df()
tab_rank, tab_map, tab_trend, tab_change, tab_fuel = st.tabs(["Rankings", "World map", "Compare countries", "Change between years", "Fuel mix"])

with tab_rank:
    c1, c2, c3 = st.columns(3)
    year = c1.selectbox("Year", list(range(config.LATEST_YEAR, 1949, -1)), key="rank_year")
    metric = c2.selectbox("Rank by", ["co2", "co2_per_capita", "cumulative_co2", "co2_per_gdp", "consumption_co2_per_capita"], format_func=label)
    n = c3.slider("Show top", 5, 40, 15)
    cs = data.cross_section(df, year, [metric])
    if cs.empty:
        st.warning("No data for this metric in this year (e.g. GDP ends in 2022, consumption-based CO₂ starts in 1990).")
    else:
        t = cs.nlargest(n, metric).iloc[::-1]
        fig = px.bar(t, x=metric, y="country", orientation="h", color="continent", color_discrete_map=CONTINENT_COLORS,
                     labels={metric: label(metric), "country": ""}, hover_data={"population": ":,.0f"})
        fig.update_layout(height=max(380, 24 * n), title=f"Top {n} countries by {label(metric)}, {year}")
        st.plotly_chart(fig, width="stretch")
        csv_button(cs.sort_values(metric, ascending=False), f"ranking_{metric}_{year}.csv")
    st.markdown("#### Total vs per person: the same countries, two very different rankings")
    tot = data.cross_section(df, year, ["co2", "co2_per_capita"])
    if not tot.empty:
        tot["rank_total"] = tot["co2"].rank(ascending=False).astype(int)
        tot["rank_per_person"] = tot["co2_per_capita"].rank(ascending=False).astype(int)
        st.dataframe(tot.nsmallest(15, "rank_total")[["country", "co2", "rank_total", "co2_per_capita", "rank_per_person", "population"]],
                     width="stretch", hide_index=True)
        note("A country can be a top emitter in total only because it is populous (India), or a small emitter in total but "
             "extreme per person (Qatar). Always say which measure a comparison uses.")

with tab_map:
    c1, c2 = st.columns(2)
    m = c1.selectbox("Indicator", ["co2_per_capita", "co2", "co2_per_gdp", "coal_co2_per_capita", "consumption_co2_per_capita"], format_func=label, key="map_m")
    yr = c2.slider("Year", 1950, config.LATEST_YEAR, config.LATEST_YEAR, key="map_y")
    cs = data.cross_section(df, yr, [m])
    cs = cs[cs["iso_code"].notna()]
    if cs.empty:
        st.warning("No data for this indicator and year.")
    else:
        hi = cs[m].quantile(0.95)
        fig = px.choropleth(cs, locations="iso_code", color=m, hover_name="country", color_continuous_scale=SEQ,
                            range_color=(0, hi), labels={m: label(m)})
        fig.update_layout(height=520, title=f"{label(m)}, {yr} (colour scale capped at the 95th percentile, {hi:,.2f})",
                          geo={"showframe": False, "projection_type": "natural earth", "bgcolor": "rgba(0,0,0,0)"})
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{len(cs)} countries with data. Grey = not reported. The cap stops a few extreme values from washing out the map.")

with tab_trend:
    c1, c2, c3 = st.columns([2.5, 1.2, 1])
    pick = c1.multiselect("Countries", sorted(cdf["country"].unique()),
                          default=["China", "United States", "India", "Germany", "United Kingdom"])
    met = c2.selectbox("Indicator", ["co2", "co2_per_capita", "cumulative_co2", "co2_per_gdp"], format_func=label, key="tr_m")
    start = c3.number_input("From", 1750, config.LATEST_YEAR - 5, 1950)
    add_world = st.toggle("Add World line (for total CO₂ / per person)", value=False)
    if not pick:
        st.info("Pick at least one country.")
    else:
        sub = cdf[cdf["country"].isin(pick) & (cdf["year"] >= start)].dropna(subset=[met])
        fig = px.line(sub, x="year", y=met, color="country", labels={met: label(met)})
        if add_world and met in ("co2", "co2_per_capita"):
            w = data.world_series(df, met, start)
            fig.add_scatter(x=w.index, y=w.values, name="World", line={"color": "#1d211e", "dash": "dot"})
        fig.update_layout(height=460, title=f"{label(met)} since {start}")
        st.plotly_chart(fig, width="stretch")
        csv_button(sub[["country", "year", met]], "country_series.csv")

with tab_change:
    c1, c2, c3 = st.columns(3)
    y0 = c1.number_input("From year", 1950, config.LATEST_YEAR - 1, config.BASE_YEAR)
    y1 = c2.number_input("To year", 1951, config.LATEST_YEAR, config.LATEST_YEAR)
    min_base = c3.number_input("Min. emissions in base year for % ranking (Mt)", 0.0, 1000.0, 10.0,
                               help="Tiny emitters produce meaningless percentages (e.g. 0.1 → 0.5 Mt = +400%).")
    if y1 <= y0:
        st.warning("'To year' must be after 'From year'.")
    else:
        p = data.paired_years(df, "co2", int(y0), int(y1))
        p["abs_change"] = p[f"co2_{y1}"] - p[f"co2_{y0}"]
        p["pct_change"] = (p[f"co2_{y1}"] / p[f"co2_{y0}"] - 1) * 100
        elig = p[p[f"co2_{y0}"] >= min_base]
        st.write(f"{len(p)} countries have CO₂ in both years; {(p['abs_change'] < 0).sum()} decreased, {(p['abs_change'] > 0).sum()} increased.")
        l, r = st.columns(2)
        dev = p.reindex(p["abs_change"].abs().sort_values(ascending=False).index).head(20).sort_values("abs_change")
        fig = go.Figure(go.Bar(x=dev["abs_change"], y=dev["country"], orientation="h",
                               marker_color=[LIME if v < 0 else GREEN for v in dev["abs_change"]]))
        fig.update_layout(height=560, title=f"Deviation bar: absolute change {y0}→{y1} (Mt), 20 largest", xaxis_title="Mt CO₂")
        l.plotly_chart(fig, width="stretch")
        r.markdown(f"**Largest % increases** (≥{min_base:g} Mt in {y0})")
        r.dataframe(elig.nlargest(8, "pct_change")[["country", f"co2_{y0}", f"co2_{y1}", "pct_change"]].round(1), hide_index=True, width="stretch")
        r.markdown("**Largest % decreases**")
        r.dataframe(elig.nsmallest(8, "pct_change")[["country", f"co2_{y0}", f"co2_{y1}", "pct_change"]].round(1), hide_index=True, width="stretch")
        csv_button(p, f"change_{y0}_{y1}.csv")

with tab_fuel:
    who = st.selectbox("Country (or World)", ["World"] + sorted(cdf["country"].unique()), key="fuel_c")
    src = df[df["country"] == who].set_index("year")[config.FUEL_COLUMNS].loc[1900:].dropna(how="all")
    if src.empty:
        st.info("No fuel breakdown for this entity.")
    else:
        long = src.reset_index().melt("year", var_name="source", value_name="Mt")
        long["source"] = long["source"].str.replace("_co2", "").str.replace("_", " ")
        pct = st.toggle("Percentage (sub-divided bar on percentage basis)", value=False)
        fig = px.area(long, x="year", y="Mt", color="source", groupnorm="percent" if pct else None,
                      color_discrete_sequence=["#1d211e", GREEN, LIME, "#d98c3f", "#3d7ea6", "#9aa39c"])
        fig.update_layout(height=440, title=f"CO₂ by source, {who}", yaxis_title="% of total" if pct else "Mt CO₂")
        st.plotly_chart(fig, width="stretch")
        last = src.dropna(how="all").iloc[-1]
        pie = px.pie(values=last.fillna(0).values, names=[c.replace("_co2", "") for c in last.index], hole=0.5,
                     color_discrete_sequence=["#1d211e", GREEN, LIME, "#d98c3f", "#3d7ea6", "#9aa39c"],
                     title=f"Pie diagram: {who}, {int(src.dropna(how='all').index[-1])}")
        pie.update_layout(height=360)
        st.plotly_chart(pie, width="stretch")
