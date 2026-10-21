import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/common.py importable

import json

import streamlit as st

from common import all_results, config, data, note, page

page("Insights & Methodology", "Findings, data provenance, cleaning decisions, assumptions, limitations and how to reproduce everything")

res = all_results()
q = res["quality"]
manifest = json.loads(config.MANIFEST.read_text())
g, inf, reg, dist = res["global_trend"], res["inference"], res["regression"], res["distribution"]

st.header("Key evidence-based findings")
st.markdown(f"""
1. **Emissions are still rising.** World fossil & industry CO₂ was {g['co2_latest'] / 1000:.1f} Gt in {config.LATEST_YEAR}, a record and {g['change_1990_latest_pct']:.0f}% above 1990. 2020 fell {abs(g['covid_drop_pct']):.1f}% and rebounded the next year.
2. **Highly concentrated.** China, the US and India emit {res['top']['top3_share_pct']:.0f}% of the total; the top ten emit {res['top']['top_n_share_pct']:.0f}%.
3. **Total ≠ per person.** The distribution of per-person emissions is strongly right-skewed (mean {dist['summary']['mean']:.2f} t vs median {dist['summary']['median']:.2f} t, β₂ = {dist['summary']['beta2']:.1f}); it is close to log-normal.
4. **Continents differ strongly** (one-way ANOVA F = {inf['anova']['F']:.1f}, η² = {inf['anova']['eta_squared']:.2f}), and the average country's emissions per person fell from {inf['paired']['mean_before']:.2f} t ({config.BASE_YEAR}) to {inf['paired']['mean_after']:.2f} t (paired t = {inf['paired']['statistic']:.2f}, p = {inf['paired']['p_value']:.3f}).
5. **Prosperity, energy and emissions move together.** log GDP vs log CO₂ per person r = {res['relationships']['gdp_log']['r']:.2f}; adding energy use gives R² = {reg['multiple']['r2']:.2f}, and GDP stops being significant (multicollinearity). This is association, not causation.
6. **Outliers are real.** Gulf oil and gas states lead every per-person outlier list. LOF also identifies rich low emitters (Switzerland, Sweden, Norway) and very poor countries with almost no fossil-fuel use.
""")

st.header("Data source & licence")
st.markdown(f"""
- **Primary dataset:** Our World in Data, *CO₂ and Greenhouse Gas Emissions* ([catalog]({manifest['files']['owid_co2.csv']['url']})), {q['rows']:,} rows × {q['columns']} columns, {q['year_min']}–{q['year_max']}.
- **Downloaded:** {manifest['downloaded_utc']} (server last-modified {manifest['files']['owid_co2.csv']['server_last_modified']}). SHA-256 `{manifest['files']['owid_co2.csv']['sha256'][:16]}…`. Hash re-checked at start-up: **{'OK' if q['raw_hashes_ok'] else 'MISMATCH'}**.
- **Supplementary:** OWID country → continent lookup, used only to form groups for ANOVA and t-tests.
- **Licence:** {manifest['license']}
""")

st.header("Cleaning & analytical decisions")
st.markdown(f"""
| Decision | Why |
|---|---|
| Keep all {q['rows']:,} raw rows; add `entity_type` | Raw data stays reproducible; aggregates are separated from countries instead of deleted |
| Analyse only `entity_type = country` ({q['countries']} countries incl. Kosovo) | Regions, income groups and '(GCP)' duplicates would double-count |
| World totals taken from the `World` row | It equals Σ countries + international aviation & shipping (verified, < 0.5% residual) |
| No imputation; per-analysis listwise deletion | A missing value is 'not reported', never zero; each result reports its own n |
| Outliers flagged, not removed | They are plausible real values (Gulf states); medians, IQR and log scales limit their influence |
| Emissions analysed for {config.LATEST_YEAR}; anything needing GDP uses {config.GDP_YEAR} | The GDP series ends in {q['gdp_last_year']} |
| log₁₀ transform for tests and regression | Per-person CO₂, GDP and energy are right-skewed; the logs are close to normal |
| Quartiles use the (N+1) position rule | Matches the course formula sheet (pandas' default differs) |
| ANOVA grand mean weighted by group size | Continents have unequal numbers of countries |
| Random seed {config.SEED} everywhere | CLT, bootstrap, CV and error simulations are reproducible |
""")

st.header("Assumptions & limitations")
st.markdown(f"""
- **Territorial, production-based CO₂** from fossil fuels & industry. Land-use change and other greenhouse gases are separate columns and not the focus. Consumption-based CO₂ (which counts emissions embodied in imports) exists only from 1990 for about 120 countries.
- **Countries are (almost) the whole population**, so confidence intervals and tests are read in a super-population sense.
- **Unequal units:** the simple mean treats every country equally; the population-weighted mean ({dist['population_weighted_mean']:.2f} t) describes the average person.
- **Correlation and regression are cross-sectional associations** with no causal identification. Energy and CO₂ are partly the same quantity, so the energy model explains a lot by construction.
- **Historical coverage is uneven:** about 5 countries in 1750, about 175 in 1950, 216 from 2000. Long-run country comparisons should start in 1950 or later.
- **Data revisions:** OWID updates the file regularly; results are tied to the hashed snapshot in `data/raw/`.
- **Time series:** annual data has no seasonal component; five hold-out years is a short evaluation window that includes the 2020 shock.
- **Assumption checks:** variances differ across continents (ANOVA, pooled t). Welch's t is reported as a sensitivity check, and p-values should be treated as approximate.
""")

st.header("Reproduce")
st.code("""uv sync                                 # Python 3.12 environment from pyproject.toml / uv.lock
uv run pytest                           # 56 tests: course examples + library cross-checks + data checks + every page
uv run python scripts/export_results.py # results.json, figures, CONCEPT_MATRIX.md
uv run streamlit run app/Home.py        # this dashboard""", language="bash")

st.header("Possible extensions")
st.markdown("""
- Consumption-based vs territorial emissions (carbon embodied in trade).
- Decomposing emission changes into population, GDP per person, energy intensity and carbon intensity (Kaya identity).
- Clustering countries on emission profiles (k-means, also listed in Module 1 tasks).
- Monthly emissions data (e.g. Carbon Monitor) to apply seasonal decomposition properly.
""")
note("Every number on this page is computed from the data at start-up. Nothing is hard-coded.")
