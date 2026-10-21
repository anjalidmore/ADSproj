import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/common.py importable

import plotly.express as px
import streamlit as st

from common import GREEN, LIME, all_results, clean_df, codebook, csv_button, data, note, page, quality, raw_df

page("Dataset Explorer", "Module 1 · data understanding & preparation — raw vs clean data, structure, quality checks")

df, raw, cb = clean_df(), raw_df(), codebook()
q = all_results()["quality"]

tab_data, tab_cols, tab_quality, tab_cov = st.tabs(["Browse data", "Columns & types", "Data quality", "Coverage"])

with tab_data:
    c1, c2, c3 = st.columns([1.2, 2, 1.4])
    types = c1.multiselect("Entity type", sorted(df["entity_type"].unique()), default=["country"])
    sub = df[df["entity_type"].isin(types)] if types else df.iloc[0:0]
    names = c2.multiselect("Countries / entities (empty = all)", sorted(sub["country"].unique()))
    yr = c3.slider("Years", int(df["year"].min()), int(df["year"].max()), (2000, int(df["year"].max())))
    sub = sub[sub["year"].between(*yr)]
    if names:
        sub = sub[sub["country"].isin(names)]
    default_cols = ["country", "iso_code", "continent", "year", "population", "co2", "co2_per_capita", "gdp_per_capita", "entity_type"]
    cols = st.multiselect("Columns", list(df.columns), default=default_cols)
    if sub.empty or not cols:
        st.warning("No rows match these filters. Widen the year range or pick an entity type.")
    else:
        st.caption(f"{len(sub):,} rows × {len(cols)} columns")
        st.dataframe(sub[cols], width="stretch", height=420, hide_index=True)
        csv_button(sub[cols], "co2_filtered.csv", "Download filtered rows (CSV)")
    note("<b>Raw vs clean:</b> the clean table keeps every raw row and value. It only adds "
         "<code>entity_type</code>, <code>continent</code>, <code>gdp_per_capita</code> and flag columns. "
         "Missing values are never filled in, and a blank is never treated as zero emissions.")

with tab_cols:
    mt = quality.missing_table(raw, cb)
    mt["level of measurement"] = [quality.measurement_level(c) for c in mt.index]
    st.caption(f"{raw.shape[1]} raw columns: {q['numeric_columns']} numeric, {q['text_columns']} text. Units and titles from the OWID codebook.")
    st.dataframe(mt.reset_index(names="column"), width="stretch", height=520, hide_index=True)
    csv_button(mt.reset_index(names="column"), "column_dictionary.csv")

with tab_quality:
    a, b, c, d = st.columns(4)
    a.metric("Duplicate rows", q["duplicate_rows"])
    b.metric("Duplicate (country, year) keys", q["duplicate_keys"])
    c.metric("Negative values in non-negative columns", q["negative_values_in_nonnegative_columns"])
    d.metric("Raw file hashes match manifest", "Yes" if q["raw_hashes_ok"] else "NO")
    st.markdown("#### Entities: countries vs aggregates")
    et = df.groupby("entity_type")["country"].nunique().rename("entities").to_frame()
    et["rows"] = df["entity_type"].value_counts()
    st.dataframe(et, width="stretch")
    st.write("Entities without an ISO code (all are aggregates or special sources except **Kosovo**):")
    st.write(", ".join(q["entities_without_iso"]))
    st.markdown("#### Double-counting check: World = Σ countries + international transport")
    rec = quality.world_reconciliation(df)
    st.dataframe(rec.round(2), width="stretch")
    st.caption("A residual under 0.5% shows that countries plus international aviation & shipping (and the 1991 Kuwaiti oil fires) "
               "make up the World total, so global totals must never add regions on top of countries.")
    st.markdown("#### Zero and suspicious values")
    z = df[df["flag_zero_co2"]].groupby("country")["year"].agg(["min", "max", "count"]).sort_values("count", ascending=False)
    st.write(f"{q['zero_co2_country_rows']} country-year rows have CO₂ = 0 ({q['zero_co2_after_1950']} after 1950). "
             "Early zeros are years before a country's record starts; recent ones are Antarctica and tiny territories.")
    st.dataframe(z.head(15), width="stretch")
    st.write(f"Largest gap between total CO₂ and the sum of its fuel components: **{q['max_fuel_sum_gap_mt']:.2f} Mt** (rounding).")

with tab_cov:
    cov = quality.coverage_by_year(df).reset_index().melt("year", var_name="variable", value_name="countries with data")
    fig = px.line(cov, x="year", y="countries with data", color="variable",
                  color_discrete_sequence=[GREEN, LIME, "#d98c3f", "#3d7ea6"])
    fig.update_layout(height=400, title="How many countries report each variable, by year")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"GDP ends in {q['gdp_last_year']}; energy starts in {q['energy_first_year']}. "
               "Analyses therefore use 2024 for emissions and 2022 when GDP is needed.")
    m = quality.missing_table(data.countries(df)[data.countries(df)["year"] >= 1990])
    m = m[m.index.isin(raw.columns)].sort_values("missing_pct", ascending=False).head(25)
    fig = px.bar(m.reset_index(names="column"), x="missing_pct", y="column", orientation="h",
                 color_discrete_sequence=[GREEN], labels={"missing_pct": "% missing (countries, 1990+)"})
    fig.update_layout(height=600, title="Most incomplete columns (countries, 1990 onwards)", yaxis={"autorange": "reversed"})
    st.plotly_chart(fig, width="stretch")
