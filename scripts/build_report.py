"""Build the academic report (docs/REPORT.md and docs/REPORT.docx) from docs/results.json.

Every number in the report is read from results.json, which scripts/export_results.py
computes from the raw data, so the report cannot drift from the code.

    uv run python scripts/export_results.py
    uv run --with pypandoc_binary python scripts/build_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
R = json.loads((DOCS / "results.json").read_text())


def f(x, d=2):
    return f"{x:,.{d}f}"


def table(rows: list[dict], cols: list[tuple[str, str, int | None]]) -> str:
    """cols = [(key, header, decimals or None for text)]."""
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---" if d is None else "---:" for _, _, d in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for k, _, d in cols:
            v = r.get(k)
            cells.append("—" if v is None else (str(v) if d is None or isinstance(v, str) else f(v, d)))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


q, g, top, ch = R["quality"], R["global_trend"], R["top"], R["changes"]
dist, rel, inf, reg, lof, ts = R["distribution"], R["relationships"], R["inference"], R["regression"], R["lof"], R["time_series"]
cl, ux, cfm = R["classification"], R["univariate_extras"], R["correlation_forms"]
dts, mo, cm = ux["discrete_summary"], ux["moments"], ux["correction"]
s, grp, box = dist["summary"], dist["grouped"], dist["box"]
mc = dist["mean_comparison"]
gvr_table = table(dist["grouped_vs_raw"], [("measure", "Measure", None), ("raw", "Raw data", 3), ("grouped", "Grouped data", 3),
                                           ("difference", "Grouped − raw", 3), ("difference_pct", "Difference %", 1)])
an, tw, pt, it, z, ci = inf["anova"], inf["two_way"], inf["paired"], inf["independent"], inf["z_test"], inf["ci_mean"]
sm, mm = reg["simple"], reg["multiple"]
c1, c2 = mm["coefficients"]
Y, G, B = q["latest_year"], 2022, 2005
india = next(r for r in top["table"] if r["country"] == "India")
us = next(r for r in top["table"] if r["country"] == "United States")
best_ts = min(ts["comparison"], key=lambda r: r["MASE"])
n_concepts = len(R["concepts"])

md = f"""---
title: "Global CO₂ Emissions Analytics"
subtitle: "An Applied Data Science Exploration of Global Carbon Emissions"
author: "[Team member names · roll numbers]"
date: "[Course name · Instructor · Submission date]"
---

# Abstract

This project applies the full Applied Data Science (ADS) syllabus — the data science process, descriptive statistics, multivariate exploration, inferential statistics, regression, anomaly detection and time-series analysis — to a real dataset of {q['rows']:,} country-year observations: the Our World in Data (OWID) CO₂ and Greenhouse Gas Emissions dataset ({q['columns']} variables, {q['year_min']}–{q['year_max']}). After separating {q['countries']} countries from regional aggregates to avoid double counting, we find that world fossil and industry CO₂ emissions reached a record {f(g['co2_latest'] / 1000, 1)} Gt in {Y}, {f(g['change_1990_latest_pct'], 0)}% above 1990, and that three countries emit {f(top['top3_share_pct'], 0)}% of the total. Emissions per person are extremely unequal and right-skewed (mean {f(s['mean'])} t, median {f(s['median'])} t, kurtosis β₂ = {f(s['beta2'], 1)}), and close to log-normal. Continents differ significantly (one-way ANOVA F = {f(an['F'], 1)}, p < 0.001), and the average country's per-person emissions fell between {B} and {Y} (paired t = {f(pt['statistic'])}, p = {f(pt['p_value'], 3)}). Across countries, log GDP per person and log CO₂ per person are strongly correlated (r = {f(rel['gdp_log']['r'], 3)}). Adding energy use per person raises R² to {f(mm['r2'], 3)}, and GDP then becomes non-significant: the relationship runs through energy use. All {n_concepts} implemented concepts are validated against the course's worked examples and independent library implementations in an automated test suite, and are presented in an interactive Streamlit dashboard.

# 1. Introduction

## 1.1 Motivation

Carbon dioxide from burning fossil fuels and making cement is the main driver of human-caused climate change. Who emits, how much, and how that is changing are central questions in climate policy. They are also a good test bed for data science: the data are large, messy (countries mixed with regional totals, uneven historical coverage), heavily skewed, and easy to misread. The best-known trap is confusing total emissions with emissions per person.

The ADS course taught each technique on small classroom examples of 4 to 15 observations. The aim of this project is to show how the same concepts behave on a real dataset with tens of thousands of observations, where data preparation decisions matter and assumptions can fail.

## 1.2 Objectives

1. Build a reproducible pipeline that preserves the raw data and produces analysis-ready country data.
2. Implement every applicable syllabus concept with the course's own formulas, apply it to real variables, and validate it.
3. Answer a focused set of research questions with computed evidence, separating description from causation.
4. Present the analysis in an interactive dashboard with an "ADS Concepts Lab" linking each concept to its formula, result and code.

## 1.3 Research questions

1. How have global CO₂ emissions changed over time?
2. Which countries contribute the largest shares of emissions?
3. How do total emissions differ from emissions per person?
4. Which countries changed most between {B} and {Y}, in absolute and percentage terms?
5. How are per-person emissions distributed across countries?
6. Which countries are unusual under clearly defined statistical criteria?
7. How are emissions related to GDP and energy use?
8. Do continents differ, and has the average country changed significantly?
9. How sensitive are the findings to the choice of measure, period and method?

# 2. Dataset

## 2.1 Source and provenance

The primary dataset is the OWID *CO₂ and Greenhouse Gas Emissions* dataset (catalog.ourworldindata.org/emissions/owid_co2), built mainly from the Global Carbon Project's Global Carbon Budget, with population, GDP (Maddison Project, international-$ at 2011 prices) and energy (Energy Institute) series. It is licensed CC BY 4.0. The file was downloaded on 21 October 2026 (server revision of 30 September 2026). Its SHA-256 hash is stored in `data/raw/MANIFEST.json` and re-checked every time the pipeline runs, because OWID revises the file regularly.

One supplementary file was added: OWID's country-to-continent lookup, used only to form groups for the t-test and ANOVA.

## 2.2 Structure

| Property | Value |
|---|---|
| Rows × columns | {q['rows']:,} × {q['columns']} ({q['numeric_columns']} numeric, {q['text_columns']} text) |
| Years | {q['year_min']}–{q['year_max']} |
| Entities | {q['entities']}, of which {q['countries']} are countries |
| Unique key | (country, year): {q['duplicate_keys']} duplicates, {q['duplicate_rows']} duplicate rows |
| GDP coverage | ends in {q['gdp_last_year']} |
| Energy coverage | starts in {q['energy_first_year']} |

The variables used most are annual CO₂ (`co2`, million tonnes), CO₂ per person (`co2_per_capita`, tonnes), cumulative CO₂, CO₂ by fuel (coal, oil, gas, cement, flaring, other), population, GDP and primary energy per person (kWh).

**Levels of measurement (Module 2.1).** `country`, `iso_code` and `continent` are nominal; `year` and growth rates are interval (no true zero, can be negative); emissions, population, GDP and per-person measures are ratio variables, so means, ratios and coefficients of variation are meaningful for them.

# 3. Methodology

## 3.1 Process

The project follows the course's data science process (CRISP-DM): prior knowledge and research questions → data preparation → modelling → evaluation → application (dashboard) → knowledge (findings). The code mirrors this: `data.py` and `quality.py` (preparation), `descriptive.py`, `correlation.py`, `inference.py`, `regression.py`, `lof.py`, `timeseries.py` (analysis, one module per syllabus unit), `results.py` (application to the dataset) and a Streamlit app (presentation).

## 3.2 Data preparation

The pipeline keeps three layers: **raw** (never modified), **clean** (every raw row, plus derived columns and flags) and **analysis-ready** (country-only cross-sections and series built for each analysis).

**Countries vs aggregates.** {len(q['entities_without_iso'])} entities have no ISO code: continents, income groups, a second set of "(GCP)" regions, the EU, international aviation and shipping, the 1991 Kuwaiti oil fires and the historical Ryukyu Islands. All are aggregates or special sources except Kosovo, which is a country. Summing aggregates with countries would double-count, so every analysis filters `entity_type = country`. As a check, the World total equals the sum of countries plus international aviation and shipping to within 0.5% in every test year.

**Missing values.** Missingness ranges from 0% to over 95% depending on the column. A blank means "not reported", never zero emissions, so nothing is imputed and rows are never dropped globally. Each analysis removes missing values only for the columns it uses and reports its own n.

**Invalid values and outliers.** No non-negative column contains a negative value ({q['negative_values_in_nonnegative_columns']}). Negative land-use or growth values are legitimate (carbon sinks, falling emissions). Of {q['zero_co2_country_rows']} zero-emission country-years, {q['zero_co2_country_rows'] - q['zero_co2_after_1950']} are before 1950 (years before a country's record starts). Fuel components add up to the total to within {f(q['max_fuel_sum_gap_mt'])} Mt. Extreme values such as Qatar's are flagged but kept, because they are plausible real values.

**Transformations.** GDP per person = GDP / population. Per-person CO₂, GDP and energy are strongly right-skewed, so tests and regressions use log₁₀ values. LOF uses min–max normalisation so neither variable dominates the distance.

**Analysis years.** Emissions use {Y} (latest). Anything involving GDP uses {G} (the last GDP year). Change comparisons use {B} as the base year.

## 3.3 Tools and validation

Python 3.12 with pandas, NumPy and SciPy (for critical values and p-values, i.e. the "tables"), Plotly and Streamlit; statsmodels and scikit-learn are used only as independent cross-checks. All formulas are written by hand following the course formula sheet, for example quartiles at the k(N+1)/4 position and multiple regression from deviation sums. An automated test suite of 60 tests checks that:

- every formula reproduces the worked examples in the course slides (one-sample, independent and paired t-tests, z-test, confidence intervals, one-way ANOVA, multiple regression, simple regression, LOF, additive and multiplicative decomposition, forecast error measures, Poisson, and the Module 2 discrete-series, grouped-quartile, open-end-class, corrected-mean and cumulative-frequency examples);
- every formula agrees with an independent library implementation (NumPy, SciPy, statsmodels, scikit-learn);
- the raw file hash, shape, key uniqueness, entity classification and World reconciliation are correct;
- every dashboard page renders.

Random processes (CLT, bootstrap, cross-validation, error simulation) use a fixed seed (42).

# 4. Exploratory and descriptive analysis

## 4.1 Global trend (RQ1)

![World CO₂ emissions 1850–{Y}](figures/01_global_trend.png)

World emissions were {f(g['co2_1950'] / 1000, 1)} Gt in 1950 and {f(g['co2_latest'] / 1000, 1)} Gt in {Y}, {f(g['multiple_1950_latest'], 1)} times higher. The {Y} value is the highest on record and {f(g['change_1990_latest_pct'], 0)}% above 1990. Emissions fell in only a few years, all tied to known shocks: the oil crises (1974–75, 1980–82), the post-Soviet collapse (1992), the financial crisis (2009) and COVID-19 (2020, {f(g['covid_drop_pct'], 1)}%), followed by a {f(g['rebound_2021_pct'], 1)}% rebound in 2021. World emissions per person were {f(g['per_capita_latest'])} t in {Y}.

## 4.2 Who emits (RQ2, RQ3)

{table(top['table'], [('country', 'Country', None), ('co2', 'CO₂ (Mt)', 0), ('share_pct', '% of world', 1), ('rank_total', 'Rank (total)', None), ('co2_per_capita', 't / person', 2), ('rank_per_capita', 'Rank (per person)', None)])}

*Table 1: Top ten emitters in {Y}.*

The top three countries emit {f(top['top3_share_pct'], 1)}% of world CO₂ and the top ten {f(top['top_n_share_pct'], 1)}%. Rankings change completely when measured per person. India is the #{india['rank_total']} emitter in total but #{india['rank_per_capita']} per person ({f(india['co2_per_capita'])} t). The United States is #{us['rank_total']} in total and #{us['rank_per_capita']} per person ({f(us['co2_per_capita'], 1)} t). Historically (cumulative emissions since 1750) the United States is responsible for {f(top['cumulative_top'][0]['share_pct'], 1)}% and China for {f(top['cumulative_top'][1]['share_pct'], 1)}%. Any comparison must therefore state whether it is about total, per-person or cumulative emissions.

![Total vs per-person emissions](figures/02_total_vs_per_capita.png)

## 4.3 Change since {B} (RQ4)

Of {ch['n_pairs']} countries with data in both years, {ch['n_decreased']} emit less in {Y} than in {B} and {ch['n_increased']} emit more. China added {f(ch['abs_increase'][0]['abs_change'], 0)} Mt, more than every other country's increase combined, followed by India ({f(ch['abs_increase'][1]['abs_change'], 0)} Mt). Percentage changes are computed only for the {ch['n_pct_eligible']} countries emitting at least {f(ch['min_base_mt'], 0)} Mt in {B}, because tiny bases produce meaningless percentages. The largest falls were {', '.join(f"{r['country']} ({f(r['pct_change'], 0)}%)" for r in ch['pct_decrease'][:5])}, mostly European. The largest rises were {', '.join(f"{r['country']} (+{f(r['pct_change'], 0)}%)" for r in ch['pct_increase'][:3])}.

![Emissions by source](figures/10_fuel_mix.png)

## 4.4 Distribution of per-person emissions (RQ5)

All measures below follow the course formula sheet and use the {s['n']} countries with per-person data in {Y}.

| Measure | Formula | Value |
|---|---|---:|
| Mean | ΣX / N | {f(s['mean'], 3)} |
| Median | (N+1)/2 th item | {f(s['median'], 3)} |
| Mode (grouped) | L + Δ₁/(Δ₁+Δ₂)·i | {f(s['grouped_mode'], 3)} |
| Q1 / Q3 | (N+1)/4, 3(N+1)/4 th item | {f(s['q1'], 3)} / {f(s['q3'], 3)} |
| D1 / D9 | k(N+1)/10 th item | {f(s['d1'], 3)} / {f(s['d9'], 3)} |
| Range; coefficient of range | L − S; (L−S)/(L+S) | {f(s['range'], 3)}; {f(s['coef_range'], 3)} |
| IQR; quartile deviation; coeff. of QD | Q3−Q1; (Q3−Q1)/2; (Q3−Q1)/(Q3+Q1) | {f(s['iqr'], 3)}; {f(s['quartile_deviation'], 3)}; {f(s['coef_qd'], 3)} |
| Standard deviation σ (÷N); s (÷(n−1)) | √[Σ(X−X̄)²/N] | {f(s['std'], 3)}; {f(s['std_sample'], 3)} |
| Coefficient of variation | σ/X̄ × 100 | {f(s['coef_variation_pct'], 1)}% |
| Karl Pearson skewness | (X̄ − Mode)/σ; 3(X̄ − Med)/σ | {f(s['pearson_sk_mode'], 3)}; {f(s['pearson_sk_median'], 3)} |
| Bowley skewness | (Q3+Q1−2Med)/(Q3−Q1) | {f(s['bowley_sk'], 3)} |
| Kurtosis β₂ | μ₄/μ₂² | {f(s['beta2'], 2)} |

*Table 2: Descriptive statistics of CO₂ per person (t), {Y}.*

![Histogram, frequency polygon and box plot](figures/03_distribution.png)

The distribution is strongly **positively skewed**: Mode ({f(s['grouped_mode'])}) < Median ({f(s['median'])}) < Mean ({f(s['mean'])}), the order the course gives for right skew. The coefficient of variation exceeds 100%, and β₂ = {f(s['beta2'], 1)} (well above 3) shows a leptokurtic shape: a sharp peak of low emitters with a heavy upper tail. Bowley's coefficient ({f(s['bowley_sk'], 3)}) is much smaller than Pearson's ({f(s['pearson_sk_median'], 3)}) because it only uses the middle 50% and ignores the tail. Note: the course formula sheet's kurtosis table pairs the shape descriptions with the wrong β₂ values; the standard meaning (β₂ > 3 = more peaked, heavier tails) is used here.

**Grouped-data estimates.** With {s['classes']} classes of width {f(s['class_width'], 0)} t (≈ √n rule), the grouped mean is {f(grp['mean_direct'], 3)} by both the direct and step-deviation methods (assumed mean A = {f(grp['assumed_mean'], 1)}), the grouped median is {f(grp['median'], 3)} and the grouped σ is {f(grp['std'], 3)}. They differ from the exact values because grouping assumes values sit at class mid-points (mean, σ) or spread evenly through each class (median, quantiles):

{gvr_table}

The lower quantiles are worst, because {dist['freq_table'][0]['f']} of {s['n']} countries fall in the first class and are spread unevenly inside it. This is the cost of grouping when the raw data are available.

**Weighted vs simple mean.** The simple mean ({f(s['mean'], 3)} t) describes the typical *country*. The population-weighted mean ΣWX/ΣW ({f(dist['population_weighted_mean'], 3)} t) describes the typical *person*. The World figure ({f(dist['world_per_capita'], 3)} t) is slightly higher again. The countries emit {f(mc['country_co2_mt'], 0)} Mt for {f(mc['country_population'] / 1e9, 3)} bn people; adding international aviation and shipping ({f(mc['international_mt'], 0)} Mt), which belong to no country, gives {f(mc['with_international'], 3)} t, and the World row's slightly larger population ({f(mc['world_population'] / 1e9, 3)} bn) brings it to {f(mc['world_per_capita'], 3)} t. The combined mean of the six continent means, weighted by group size, reproduces the overall mean exactly ({f(dist['combined_mean_check'], 3)}).

## 4.5 Outliers (RQ6, part 1)

With the box-plot rule, the limits are Q1 − 1.5·IQR = {f(box['lower_limit'], 2)} and Q3 + 1.5·IQR = {f(box['upper_limit'], 2)}. {box['n_high_outliers']} countries are high outliers: {', '.join(r['country'] for r in dist['outliers'])}. Most are oil- and gas-exporting states, several with small populations (Qatar, Kuwait, Brunei, Bahrain); the rest are large fossil-fuel producers or energy-intensive economies (Saudi Arabia, Kazakhstan, Australia, the United States). They are genuine values and are kept; medians, IQR and log scales limit their influence.

## 4.6 Classification, tabulation and diagrams

The course names four types of classification; each was applied to the {Y} country data ({cl['n']} countries with a continent).

- **Geographical (area-wise).** By continent: Asia's {next(r for r in cl['geographical'] if r['continent'] == 'Asia')['countries']} countries emit {f(next(r for r in cl['geographical'] if r['continent'] == 'Asia')['share_of_world_pct'], 1)}% of world CO₂, North America {f(next(r for r in cl['geographical'] if r['continent'] == 'North America')['share_of_world_pct'], 1)}%, Europe {f(next(r for r in cl['geographical'] if r['continent'] == 'Europe')['share_of_world_pct'], 1)}%, Africa {f(next(r for r in cl['geographical'] if r['continent'] == 'Africa')['share_of_world_pct'], 1)}%. Per person, North America ({f(next(r for r in cl['geographical'] if r['continent'] == 'North America')['co2_per_person_t'])} t) is ten times Africa ({f(next(r for r in cl['geographical'] if r['continent'] == 'Africa')['co2_per_person_t'])} t).
- **Chronological (time-wise).** World emissions by decade: mean annual CO₂ rose from {f(cl['chronological'][0]['mean_annual_mt'] / 1000, 1)} Gt in the 1950s to {f(cl['chronological'][-2]['mean_annual_mt'] / 1000, 1)} Gt in the 2010s and {f(cl['chronological'][-1]['mean_annual_mt'] / 1000, 1)} Gt in {cl['chronological'][-1]['period']}. Growth slowed from over 50% per decade in the 1960s–70s to {f(cl['chronological'][-1]['change_in_mean_pct'], 0)}% in the latest period.
- **Qualitative (by attribute).** Two-fold: {next(r['countries'] for r in cl['two_fold'] if r['attribute'] == 'Fell')} countries emit less than in {B} and {next(r['countries'] for r in cl['two_fold'] if r['attribute'] != 'Fell')} the same or more. Manifold: continent → fell or rose → largest reported fuel source (oil is the largest source in most countries, coal in most of the biggest Asian emitters).
- **Quantitative (by magnitude).** Classes of CO₂ per person (Table 3a).

{table(cl['two_way_table'], [('continent', 'Continent', None), ('< 1 t', '< 1 t', None), ('1–3 t', '1–3 t', None), ('3–6 t', '3–6 t', None), ('6–10 t', '6–10 t', None), ('10 t and above', '10 t +', None), ('Total', 'Total', None)])}

*Table 3a: Two-way table, continent × CO₂ per person ({Y}). The Total row is the simple (one-way) table; adding "fell or rose since {B}" as a third characteristic gives the higher-order table in the dashboard.*

Diagrams were chosen by question. One-dimensional bars cover rankings (simple), comparisons across years (multiple), composition (sub-divided and percentage), change (deviation) and one extreme value (broken bar). Two-dimensional diagrams are used where area should carry the value: squares (side = √value), circles (r = √(value/π)), pie, and a rectangular diagram whose width is population, height is CO₂ per person, and area is therefore total CO₂. That one figure shows all three measures of RQ3 at once.

![Rectangular diagram](figures/13_rectangles.png)

![Percentage bar diagram](figures/12_percentage_bar.png)

The percentage bars show that coal is {f(R['diagrams']['fuel_pct'][1]['coal'], 0)}% of Asia's emissions against {f(R['diagrams']['fuel_pct'][2]['coal'], 0)}% of Europe's: composition differs as much as size. Line graphs use a zero baseline by default; the dashboard also shows a false baseline (marked break), which reveals the 2009 and 2020 dips but exaggerates change.

## 4.7 Further univariate methods

**Discrete series.** A naturally discrete variable was built from the data: for each of {dts['n']} countries, {ux['discrete_variable']}. Discrete-series formulas give mean ΣfX/N = {f(dts['mean'], 3)} (short-cut method: {f(dts['mean_shortcut'], 3)}), median = the X whose cf first reaches (N+1)/2 = {dts['median']:g}, mode = {', '.join(f'{v:g}' for v in dts['mode'])}, Q1 = {dts['q1']:g}, Q3 = {dts['q3']:g} and σ = {f(dts['std'], 3)} (actual-mean and assumed-mean methods agree). Unlike per-person emissions, this count is close to symmetric. Grouped into inclusive classes 0–3, 4–7, …, the classes were converted to exclusive form (−0.5 / +0.5) before applying the interpolation formula (grouped median {f(ux['inclusive_grouped_median'], 3)}).

**Cumulative frequency and ogive.** The "less than" and "more than" ogives of CO₂ per person cross at N/2, giving a graphical median of {f(ux['ogive_median'], 4)} t, identical to the formula L + (N/2 − cf)/f × i ({f(ux['formula_median'], 4)}). Quartiles read from the ogive are {f(ux['ogive_q1'], 3)} and {f(ux['ogive_q3'], 3)} t.

![Ogives](figures/11_ogive.png)

**Open-end classes.** With the last class written as "{ux['open_from']:g} t and above", the course rule (assume the neighbour's width) gives a mean of {f(ux['open_mean'], 3)} t, against {f(ux['closed_grouped_mean'], 3)} t with all classes closed and {f(ux['actual_mean'], 3)} t exactly. The median ({f(ux['open_median'], 3)} t) does not depend on the open limit, which is why the course recommends the median for open-ended, skewed distributions.

**Correcting incorrect items.** If the {cm['n_wrong']} aggregate rows (regions, income groups, World) were averaged in with the countries by mistake, the per-person mean would be {f(cm['incorrect_mean'], 4)} t. The course correction ΣX = N·X̄ − ΣX_wrong, with N reduced accordingly, recovers {f(cm['correct_mean'], 4)} t, exactly the countries-only mean. This is a concrete demonstration of why separating aggregates matters.

**Moments.** Central moments μ₂ = {f(mo['central']['mu2'], 2)}, μ₃ = {f(mo['central']['mu3'], 1)}, μ₄ = {f(mo['central']['mu4'], 0)} (identical whether computed directly or from moments about the origin), giving β₁ = {f(mo['beta1'], 3)} and β₂ = {f(mo['beta2'], 2)}. Pearson's empirical relation Mode ≈ 3·Median − 2·Mean gives {f(ux['empirical_mode'], 3)}, far from the grouped mode ({f(ux['grouped_mode'], 3)}). The relation assumes moderate skew, and its failure is another sign of how skewed the data are.

# 5. Multivariate exploration

![Bubble chart of GDP vs CO₂ per person](figures/04_bubble_gdp_co2.png)

For {rel['gdp_raw']['n']} countries in {G}, the central data point (Module 2.4) is (GDP per person = {f(rel['central_point']['gdp_per_capita'], 0)} int-$, CO₂ per person = {f(rel['central_point']['co2_per_capita'])} t). The countries farthest from it are {', '.join(r['country'] for r in rel['farthest_from_centre'][:3])}.

| Pair | n | Pearson r | Degree | r² | Probable error |
|---|---:|---:|---|---:|---:|
| GDP pc vs CO₂ pc (raw) | {rel['gdp_raw']['n']} | {f(rel['gdp_raw']['r'], 3)} | {rel['gdp_raw']['degree']} | {f(rel['gdp_raw']['r_squared'], 3)} | {f(rel['gdp_raw']['probable_error'], 4)} |
| log GDP pc vs log CO₂ pc | {rel['gdp_log']['n']} | {f(rel['gdp_log']['r'], 3)} | {rel['gdp_log']['degree']} | {f(rel['gdp_log']['r_squared'], 3)} | {f(rel['gdp_log']['probable_error'], 4)} |
| log energy pc vs log CO₂ pc | {rel['energy_log']['n']} | {f(rel['energy_log']['r'], 3)} | {rel['energy_log']['degree']} | {f(rel['energy_log']['r_squared'], 3)} | {f(rel['energy_log']['probable_error'], 4)} |

*Table 3: Karl Pearson correlation coefficients ({G}).*

All three course methods (actual-mean deviations, direct method and assumed-mean deviations) give identical r, and |r| > 6 × probable error in every case. The relationship is stronger on log scales because both variables are skewed and the relationship is multiplicative. Population is barely correlated with emissions per person (log r = {f(next(r for r in rel['matrix_log'] if r['index'] == 'population')['co2_per_capita'], 2)}) but strongly with total emissions. **Correlation is not causation:** richer countries use more energy, most of it fossil, but the energy mix differs widely. Sweden and Germany have almost the same GDP per person, yet Sweden emits less than half as much CO₂ per person while using more energy per person, thanks to hydro and nuclear power.

**Types of correlation.** By direction and shape: positive (log GDP vs log CO₂ per person, r = {f(cfm['positive']['r'], 3)}); negative across countries (log GDP vs log CO₂ per kWh of energy, r = {f(cfm['negative_cross_section']['r'], 3)}: richer countries have a cleaner energy mix); negative over time (UK emissions vs year 1990–{Y}, r = {f(cfm['negative_time']['r'], 3)}); none (population vs CO₂ per person, r = {f(cfm['none']['r'], 3)}); and non-linear (raw GDP vs raw CO₂ per person, where the curved pattern gives a lower r of {f(cfm['nonlinear_raw']['r'], 3)}). By number of variables: the simple correlations of log CO₂ per person with log GDP ({f(cfm['simple_r_gdp'], 3)}) and log energy ({f(cfm['simple_r_energy'], 3)}) are both high, but the **partial** correlation with GDP holding energy constant is only {f(cfm['partial_gdp_given_energy'], 3)}, while energy holding GDP constant keeps {f(cfm['partial_energy_given_gdp'], 3)}. The **multiple** correlation R = {f(cfm['multiple_R'], 3)} equals the square root of the regression R² in Section 7.

# 6. Inferential statistics

## 6.1 Normal distribution and the central limit theorem

| Range | Normal | Observed (raw) | Observed (log₁₀) |
|---|---:|---:|---:|
""" + "\n".join(f"| μ ± {r['k_sigma']}σ | {f(r['normal_pct'], 1)}% | {f(r['observed_pct'], 1)}% | {f(l['observed_pct'], 1)}% |"
                for r, l in zip(inf['empirical_raw'], inf['empirical_log'])) + f"""

*Table 4: Empirical-rule check for CO₂ per person, {Y}.*

Raw per-person emissions break the 68–95–99.7 rule (too many values within 1σ, too few within 3σ). Their logarithms follow it closely, and the kurtosis of the logs is β₂ = {f(dist['log_summary']['beta2'])}, close to the normal value of 3. The variable is therefore approximately **log-normal**, although the logs keep a mild left skew (Bowley {f(dist['log_summary']['bowley_sk'])}) from the very lowest emitters. This justifies using the log scale for tests and regression.

![Central limit theorem on real data](figures/06_clt.png)

**CLT.** Treating the {s['n']} countries as a population, 10,000 random samples of n = 30 were drawn. The sample means average {f(inf['clt']['mean_of_means'], 3)} (population mean {f(inf['clt']['population_mean'], 3)}) with standard deviation {f(inf['clt']['sd_of_means'], 3)}, matching the theoretical σ/√n = {f(inf['clt']['theoretical_se'], 3)}. Skewness drops from {f(inf['clt']['population_skew'], 2)} to {f(inf['clt']['skew_of_means'], 2)}: the sampling distribution of the mean is close to normal even though the data are not.

## 6.2 Confidence intervals

The 95% z-interval (n ≥ 30) for mean CO₂ per person in {Y} is {f(ci['mean'], 3)} ± {f(ci['critical'], 2)} × {f(ci['sd'], 3)}/√{ci['n']} = **({f(ci['lower'], 3)}, {f(ci['upper'], 3)})** t. A percentile bootstrap (Module 1, 5,000 resamples) gives a 95% interval for the *median* of ({f(inf['bootstrap_median']['lower'], 3)}, {f(inf['bootstrap_median']['upper'], 3)}) t. Since the countries are nearly the whole population rather than a random sample, these intervals are interpreted in a super-population sense: uncertainty about the process that generates national emissions.

## 6.3 Hypothesis tests

| Test | H₀ | Statistic | Critical (5%) | p-value | Decision |
|---|---|---:|---:|---:|---|
| z, single mean | μ = world per-person value ({f(z['mu0'], 3)} t) | Z = {f(z['statistic'], 3)} | ±{f(z['critical'], 2)} | {f(z['p_value'], 3)} | {z['decision']} |
| Paired t ({B} vs {Y}, n = {pt['n']}) | μ_d = 0 | t = {f(pt['statistic'], 3)} | ±{f(pt['critical'], 3)} | {f(pt['p_value'], 4)} | {pt['decision']} |
| Independent t, log₁₀ (Europe n={it['n1']} vs Asia n={it['n2']}) | μ₁ = μ₂ | t = {f(it['statistic'], 3)} | ±{f(it['critical'], 3)} | {f(it['p_value'], 3)} | {it['decision']} |

*Table 5: z and t tests on CO₂ per person.*

- **Paired t-test.** Mean per-person emissions of the same {pt['n']} countries fell from {f(pt['mean_before'], 3)} t to {f(pt['mean_after'], 3)} t (d̄ = {f(pt['mean_diff'], 3)}), which is significant at 5%. Yet {pt['n_increased']} countries rose and only {pt['n_decreased']} fell: large reductions in high-emitting countries outweigh many small increases. The mean and the count tell different stories.
- **Independent t-test.** Europe's geometric-mean emissions ({f(inf['independent_back']['europe_geo_mean'])} t) exceed Asia's ({f(inf['independent_back']['asia_geo_mean'])} t), but the difference is not significant at 5%. The variance ratio is {f(it['variance_ratio'], 1)}, which violates the pooled-variance assumption, so Welch's test is also reported (p = {f(inf['independent_welch_p'], 3)}). It leads to the same conclusion.
- **z-test.** The average country is not significantly different from the average person's value, partly because country values are so dispersed.

## 6.4 Analysis of variance

![CO₂ per person by continent](figures/05_anova_continents.png)

""" + table(an["table"], [("index", "Source", None), ("SS", "SS", 3), ("df", "d.f.", 0), ("MS", "MS", 3), ("F", "F", 2)]) + f"""

*Table 6: One-way ANOVA, log₁₀ CO₂ per person by continent ({Y}).*

F = {f(an['F'], 2)} far exceeds the 5% critical value F({an['df_between']}, {an['df_within']}) = {f(an['critical'], 2)} (p = {an['p_value']:.1e}), so mean log emissions per person differ between continents. η² = {f(an['eta_squared'], 3)}: continent accounts for {f(an['eta_squared'] * 100, 0)}% of the variation. **Correction to the course method:** the slides compute the grand mean as the mean of the sample means. That is correct only when all groups are the same size. Here the groups range from 12 to 55 countries, so the grand mean of all observations ({f(an['grand_mean'], 4)}) is used, not the mean of means ({f(an['mean_of_means'], 4)}). With this choice SS_total = SS_between + SS_within holds exactly. Europe has a much smaller spread than Africa, so the equal-variance assumption is only approximate.

**Two-way ANOVA (without replication)** on continental per-person emissions (rows: 6 continents; columns: 1990, 2000, 2010, 2020, {Y}):

""" + table(tw["table"], [("index", "Source", None), ("SS", "SS", 3), ("df", "d.f.", 0), ("MS", "MS", 3), ("F", "F", 2), ("F_crit_5pct", "F crit (5%)", 2)]) + f"""

*Table 7: Two-way ANOVA, continent × year.*

The continent effect is significant ({tw['decision_rows']}); the year effect is not ({tw['decision_columns']}). This is not because nothing changed: Asia rose from {f(inf['two_way_matrix'][1]['1990'])} to {f(inf['two_way_matrix'][1][str(Y)])} t while Europe fell from {f(inf['two_way_matrix'][2]['1990'])} to {f(inf['two_way_matrix'][2][str(Y)])} t. Opposite movements cancel in the column totals and end up in the residual (a continent × year interaction, which a design without replication cannot test).

## 6.5 Type I and Type II errors

Repeatedly sampling n = {inf['errors']['n']} countries and testing the true mean at α = 0.05 gave a Type I error rate of {f(inf['errors']['type1_rate'], 3)}, above the nominal 0.05: with heavily skewed data the normal approximation is not yet accurate at this sample size. Testing a mean that is wrong by {inf['errors']['shift']} t gave a Type II error rate β = {f(inf['errors']['type2_rate'], 3)} (power {f(inf['errors']['power'], 3)}). Increasing n is the only way to reduce both errors together, as the course notes state.

## 6.6 Poisson distribution

The yearly number of countries (emitting ≥ 1 Mt) whose emissions fell by more than 10% was tested as a candidate Poisson count, 1991–{Y}. A Poisson variable has mean = variance. Here the mean is {f(inf['poisson']['mean'])} and the variance {f(inf['poisson']['variance'])} (dispersion index {f(inf['poisson']['dispersion_index'])}, χ² test p = {inf['poisson']['p_value']:.1e}). **The Poisson model is rejected**: sharp drops cluster in global crisis years (42 countries in 2020, 27 in 2009) instead of occurring independently at a constant rate. The Poisson formula itself is implemented and checked against the course example (P(X ≤ 2 | μ = 4) = 0.238).

# 7. Regression analysis

![Simple linear regression](figures/07_regression.png)

**Simple regression** of log₁₀ CO₂ per person on log₁₀ GDP per person ({G}, n = {sm['n']}), with slope a = (nΣxy − ΣxΣy)/(nΣx² − (Σx)²) and intercept b = (Σy − aΣx)/n:

log₁₀ CO₂pc = {f(sm['slope'], 4)} · log₁₀ GDPpc {sm['intercept']:+.4f},   R² = {f(sm['metrics']['R2'], 3)},   t(slope) = {f(sm['t_slope'], 1)}.

On log–log scales the slope is an elasticity: across countries, 1% higher GDP per person goes with about {f(sm['slope'])}% higher CO₂ per person.

**Multiple regression** adds X₂ = log₁₀ energy per person, with coefficients from the course's deviation-sum formulas:

log₁₀ CO₂pc = {f(mm['b0'], 4)} {mm['b1']:+.4f} · log₁₀ GDPpc {mm['b2']:+.4f} · log₁₀ Energy pc

| Statistic | Value |
|---|---:|
| R² / adjusted R² | {f(mm['r2'], 4)} / {f(mm['adj_r2'], 4)} |
| F (2, {mm['anova']['df_error']}) | {f(mm['F'], 1)} (critical {f(mm['F_critical'], 2)}) |
| b₁ (GDP): t, p | {f(c1['t'], 3)}, {f(c1['p_value'], 3)} |
| b₂ (energy): t, p | {f(c2['t'], 2)}, {c2['p_value']:.1e} |
| r₁₂ between predictors | {f(mm['r12'], 3)} |
| MAE / RMSE (log₁₀ scale) | {f(mm['metrics']['MAE'], 3)} / {f(mm['metrics']['RMSE'], 3)} |
| MAPE after back-transforming to tonnes | {f(reg['back_transformed']['multi_mape_original'], 1)}% (simple model: {f(reg['back_transformed']['simple_mape_original'], 1)}%) |

*Table 8: Multiple regression results ({G}).*

| Metric | Simple (GDP) | Multiple (GDP + energy) |
|---|---:|---:|
| R / R² | {f(abs(sm['r']), 3)} / {f(sm['r2'], 4)} | {f(mm['R'], 3)} / {f(mm['r2'], 4)} |
| Adjusted R² | {f(sm['adj_r2'], 4)} | {f(mm['adj_r2'], 4)} |
| F (df) | {f(sm['F'], 1)} (1, {sm['anova']['df_error']}) | {f(mm['F'], 1)} (2, {mm['anova']['df_error']}) |
| SS regression / SS error | {f(sm['anova']['ss_regression'], 3)} / {f(sm['anova']['ss_error'], 3)} | {f(mm['anova']['ss_regression'], 3)} / {f(mm['anova']['ss_error'], 3)} |
| MAE / MSE / RMSE (log₁₀) | {f(sm['metrics']['MAE'], 3)} / {f(sm['metrics']['MSE'], 3)} / {f(sm['metrics']['RMSE'], 3)} | {f(mm['metrics']['MAE'], 3)} / {f(mm['metrics']['MSE'], 3)} / {f(mm['metrics']['RMSE'], 3)} |
| MAPE (log₁₀ / tonnes) | {f(sm['metrics']['MAPE'], 1)}% / {f(reg['back_transformed']['simple_mape_original'], 1)}% | {f(mm['metrics']['MAPE'], 1)}% / {f(reg['back_transformed']['multi_mape_original'], 1)}% |
| b₁ (GDP): coefficient, SE, t | {f(sm['slope'], 3)}, {f(sm['se_slope'], 4)}, {f(sm['t_slope'], 2)} | {f(mm['b1'], 3)}, {f(c1['se'], 4)}, {f(c1['t'], 2)} |
| Prediction for the median country | {f(reg['prediction']['co2_pc_simple'])} t | {f(reg['prediction']['co2_pc_multi'])} t |

*Table 9: Simple vs multiple regression, every metric from the Module 4 slides ({G}). The median country has GDP {f(reg['prediction']['gdp_per_capita'], 0)} int-$ and energy use {f(reg['prediction']['energy_per_capita'], 0)} kWh per person.*

Following the course procedure, the overall F-test is applied first. It rejects H₀ (b₁ = b₂ = 0), so the individual coefficients are then tested. Energy per person is highly significant. GDP becomes **not significant** (p = {f(c1['p_value'], 3)}), and its sign even turns slightly negative. The reason is **multicollinearity**, which the analysis flags automatically: GDP and energy per person are correlated at r₁₂ = {f(mm['r12'], 2)}, the tolerance 1 − r₁₂² is only {f(reg['collinearity']['tolerance'], 3)} and VIF = {f(reg['collinearity']['vif'], 1)}. The (1 − r₁₂²) term in the slide formula for SE(bᵢ) therefore multiplies the standard errors by √VIF = {f(reg['collinearity']['se_inflation'], 1)}. Adding energy also shrinks the error variance, so the net effect on GDP's standard error is {f(sm['se_slope'], 4)} → {f(c1['se'], 4)}, while its coefficient collapses from {f(sm['slope'], 3)} to {f(mm['b1'], 3)}. The model cannot separate the two effects. R², F and the predictions are not affected. The economic reading is that income is associated with emissions mainly *through* energy use. Energy and CO₂ are also partly the same quantity, since most energy is fossil, so the high R² is partly by construction and is not evidence of a causal mechanism.

**Model evaluation (Module 1).** On a 2/3 : 1/3 hold-out split ({reg['holdout_multi']['n_train']} / {reg['holdout_multi']['n_test']} countries), test R² = {f(reg['holdout_multi']['test']['R2'], 3)}. With 5-fold cross-validation, mean validation R² = {f(reg['cv_multi']['mean']['R2'], 3)} for the multiple model and {f(reg['cv_simple']['mean']['R2'], 3)} for the simple model. Validation scores are close to the in-sample ones, so neither model overfits. MAPE is reported after back-transforming to tonnes, because on the log scale values near zero make MAPE meaningless (a limitation the slides note).

# 8. Anomaly detection: Local Outlier Factor (RQ6, part 2)

![LOF scores](figures/08_lof.png)

LOF was implemented step by step as in Module 5: Manhattan distance → k-distance and k-neighbourhood → reachability distance max(k-dist(o), d(p,o)) → local reachability density → LOF = average neighbour lrd / own lrd. It was applied to min–max scaled log GDP and log CO₂ per person ({G}, n = {lof['n']}, k = {lof['k']}). {lof['n_flagged']} countries have LOF > {lof['threshold']}:

""" + table(lof["table"][:8], [("country", "Country", None), ("continent", "Continent", None), ("gdp_per_capita", "GDP pc ($)", 0), ("co2_per_capita", "CO₂ pc (t)", 2), ("lof", "LOF", 3)]) + f"""

*Table 10: Highest LOF scores.*

Unlike the one-dimensional IQR rule, which flagged only high emitters, LOF is local and two-dimensional. It flags Qatar (extreme on both axes), but also countries whose emissions are unusual *for their income*: rich low-carbon economies (Norway, Switzerland, Sweden, Singapore), very poor countries with almost no fossil-fuel use (Burundi, DR Congo), and North Korea (high CO₂ for its very low recorded GDP). Of LOF's top ten, only {', '.join(lof['lof_top10_also_iqr'])} were also IQR outliers. The two methods answer different questions.

# 9. Time-series analysis

![Forecasts on the hold-out period](figures/09_forecasts.png)

**Decomposition.** Classical decomposition (Y = T + S + R or T × S × R) requires a seasonal pattern within each year. OWID data are annual, so no seasonal component exists. The method is implemented and verified on the course's quarterly-sales example (indices −10.44, −11.69, 6.06, 16.06; forecasts 23.31 and 43.56), and only its trend step, a centred 5-year moving average, is applied to world CO₂.

**Forecasting.** World emissions 1950–2019 form the training period and 2020–{Y} the hold-out. Average, moving-average (k = 3) and naïve forecasts are one-step-ahead, using actual values up to the previous year as in the slides. The least-squares trend is fitted on 2000–2019 and extrapolated. MASE scales MAE by the in-sample naïve error of the training period.

""" + table(ts["comparison"], [("method", "Method", None), ("MAE", "MAE (Mt)", 0), ("RMSE", "RMSE (Mt)", 0), ("MAPE", "MAPE (%)", 2), ("MASE", "MASE", 2)]) + f"""

*Table 11: Forecast accuracy on 2020–{Y}.*

The average method is useless for a trending series ({f(ts['comparison'][0]['MAPE'], 0)}% error), exactly as the slides warn: old values never leave the average. The 2000–2019 trend line (slope {f(ts['trend_fit_train']['slope'], 0)} Mt per year, R² = {f(ts['trend_fit_train']['r2'], 3)}) overshoots because growth slowed after about 2013 and 2020 fell sharply. The best method by MASE was **{best_ts['method']}**. All MASE values exceed 1, so over these five volatile years no method beat simply carrying last year's value forward. Fitting the trend on 2000–{Y} gives a slope of {f(ts['trend_full']['slope'], 0)} Mt per year. Mechanically extended to 2030 that implies about {f(ts['projection']['2030'] / 1000, 1)} Gt, but this is an extrapolation, not a prediction: it ignores policy and technology change.

# 10. Validation and academic integrity

- **Formula correctness:** each implementation reproduces the course worked examples to the printed precision. The one discrepancy (independent t = 0.610 vs 0.609 on the slide) comes from the slide rounding an intermediate value.
- **Independent cross-checks:** quantiles (NumPy), moments and kurtosis (SciPy), t-tests and one-way ANOVA (SciPy), two-way ANOVA and regression including standard errors (statsmodels), LOF (scikit-learn).
- **Data integrity:** raw file hash, 50,411 × 79 shape, unique keys, entity classification, World reconciliation, and no imputed values (tests in `tests/test_data.py`).
- **Consistency:** the dashboard, this report, the slides and the viva guide all read numbers from the same `results.py` code, so tables, charts and text cannot disagree.
- **Errors in course material found and handled:** kurtosis shape labels swapped on the formula sheet; ANOVA grand mean as an unweighted mean of means; the Module 4 slide on b₂ reuses b₁'s numbers (correct: SE(b₂) = 0.976, t = −1.697, still not significant), and slide 30 substitutes Σx₂y = −683.5 instead of −953.5 although its printed b₂ = −1.6561 is right; the Module 2 "median from cumulative data" example uses i = 10 for classes 5 wide and prints 14.29 instead of 12.15.

# 11. Discussion and limitations

- **Measures matter.** Total, per-person, cumulative and consumption-based emissions rank countries very differently. The simple mean across countries (one vote per country) differs from the population-weighted mean (one vote per person).
- **Coverage.** About 5 countries are reported in 1750, about 175 in 1950 and 216 from 2000. GDP ends in {G}, and consumption-based emissions start in 1990 for about 120 countries. Long-run comparisons start in 1950 or later.
- **Scope.** Only fossil and industrial CO₂ (territorial, production-based) is analysed. Land-use change and other greenhouse gases are excluded.
- **Inference on a near-census.** The countries are not a random sample, so p-values and confidence intervals are read in a super-population sense.
- **Assumptions.** Normality holds only on the log scale; variances differ across continents (Welch's test is reported as a sensitivity check); regression predictors are collinear; the time-series hold-out is short and contains the 2020 shock.
- **No causal claims.** Every relationship reported is an association between countries.
- **Revisions.** OWID revises historical values; results refer to the hashed snapshot.

# 12. Conclusions and future work

Global CO₂ emissions are still rising and are highly concentrated, while per-person emissions are extremely unequal and close to log-normal. The average country's per-person emissions have fallen since {B}, driven by large cuts in high emitters, while most countries' emissions rose. Continents differ strongly. Across countries, income and emissions are linked mainly through energy use. Methodologically, the project shows how classroom techniques must be adapted to real data: separating aggregates before summing, logging skewed variables, weighting means and grand means correctly, checking assumptions (Poisson, equal variances, normality) instead of assuming them, and preferring simple honest baselines in forecasting.

Possible extensions: consumption-based vs territorial emissions; decomposing changes into population, GDP per person, energy intensity and carbon intensity (Kaya identity); clustering countries on emission profiles; and monthly data (e.g. Carbon Monitor) to apply seasonal decomposition.

# References

1. Our World in Data (2026). *CO₂ and Greenhouse Gas Emissions dataset*. https://github.com/owid/co2-data · https://catalog.ourworldindata.org/emissions/owid_co2/ (CC BY 4.0).
2. Friedlingstein, P. et al. *Global Carbon Budget* (Global Carbon Project), the primary source of national CO₂ estimates.
3. Kotu, V. & Deshpande, B. *Data Science: Concepts and Practice*. Elsevier / Morgan Kaufmann.
4. ADS course materials, Modules 1–6, and the course formula sheet.
5. Breunig, M. M., Kriegel, H.-P., Ng, R. T. & Sander, J. (2000). LOF: Identifying density-based local outliers. *Proc. ACM SIGMOD*.
6. Hyndman, R. J. & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4).

# Appendix A — Concept-to-implementation summary

| Module | Concept | Code |
|---|---|---|
""" + "\n".join(f"| {c['module']} | {c['concept']} | `{c['code']}` |" for c in R["concepts"]) + """

The full matrix with formulas, computed outputs, interpretations and validation is in `docs/CONCEPT_MATRIX.md` and in the dashboard's ADS Concepts Lab.

# Appendix B — Reproducing the results

```
uv sync
uv run pytest
uv run python scripts/export_results.py
uv run --with pypandoc_binary python scripts/build_report.py
uv run streamlit run app/Home.py
```
"""

(DOCS / "REPORT.md").write_text(md)
print("wrote", DOCS / "REPORT.md", f"({len(md.split()):,} words)")

try:
    import pypandoc
except ImportError:
    print("pypandoc not installed: run with `uv run --with pypandoc_binary ...` to also build REPORT.docx")
else:
    pypandoc.convert_file(str(DOCS / "REPORT.md"), "docx", outputfile=str(DOCS / "REPORT.docx"),
                          extra_args=[f"--resource-path={DOCS}", "--toc", "--toc-depth=2"])
    print("wrote", DOCS / "REPORT.docx")
