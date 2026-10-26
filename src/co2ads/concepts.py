"""Concept-to-implementation registry: every syllabus concept, its formula, the real computed
output, an interpretation, how it was validated, and where the code lives.

build(res) takes the dict of computed results (see scripts/export_results.py:compute_all)
and returns a list of entries. The ADS Concepts Lab page and docs/CONCEPT_MATRIX.md are
both generated from this list.
"""

from __future__ import annotations

from . import config

Y, G, B = config.LATEST_YEAR, config.GDP_YEAR, config.BASE_YEAR


def f(x, d=2):
    return f"{x:,.{d}f}"


def build(res: dict) -> list[dict]:
    q = res["quality"]
    g = res["global_trend"]
    top = res["top"]
    dist = res["distribution"]
    s = dist["summary"]
    grp = dist["grouped"]
    box = dist["box"]
    rel = res["relationships"]
    inf = res["inference"]
    reg = res["regression"]
    lof = res["lof"]
    tsr = res["time_series"]
    pc = f"CO₂ per person (t), {Y}, n = {s['n']} countries"
    E = []

    def add(module, concept, variables, formula, output, interpretation, validation, code, objective=""):
        E.append({"module": module, "concept": concept, "variables": variables, "objective": objective,
                  "formula": formula, "output": output, "interpretation": interpretation,
                  "validation": validation, "code": code})

    # ------------------------------------------------------------------ Module 1
    add("M1 Data Science Process", "CRISP-DM / data science process",
        "Whole project", r"\text{Prior knowledge} \rightarrow \text{Prepare} \rightarrow \text{Model} \rightarrow \text{Apply} \rightarrow \text{Knowledge}",
        {"Research questions": "9", "Pipeline layers": "raw → clean → analysis-ready", "Syllabus concepts implemented": "see this list"},
        "The project follows the course framework: questions first, then data understanding and preparation, then modelling and evaluation, then a dashboard that reports the knowledge gained.",
        "Each stage has its own module and tests.", "src/co2ads/ (package layout)",
        "Structure the project the way the course defines the data science process.")
    add("M1 Data Science Process", "Data exploration: volume & dimensions",
        "All 79 raw columns", r"\text{rows} \times \text{columns}",
        {"Rows": f"{q['rows']:,}", "Columns": str(q["columns"]), "Years": f"{q['year_min']}–{q['year_max']}",
         "Entities": str(q["entities"]), "Countries": str(q["countries"])},
        f"{q['rows']:,} observations is the 'volume' motivation from Module 1; with {q['columns']} attributes, picking relevant variables (feature selection) matters.",
        "Shape asserted in tests/test_data.py::test_shape_and_keys; file hash checked against MANIFEST.json.",
        "src/co2ads/quality.py::report")
    add("M1 Data Science Process", "Data cleaning: duplicates",
        "country, year", r"\#\{(country, year)\ \text{appearing more than once}\}",
        {"Duplicate rows": str(q["duplicate_rows"]), "Duplicate (country, year) keys": str(q["duplicate_keys"])},
        "(country, year) is a valid unique key, so no rows need dropping for duplication.",
        "tests/test_data.py::test_shape_and_keys", "src/co2ads/quality.py::report")
    add("M1 Data Science Process", "Data cleaning: inconsistent entities / aggregates",
        "country, iso_code", r"\text{entity\_type} = f(\text{country}, \text{iso\_code})",
        {"Entities without ISO code": str(len(q["entities_without_iso"])),
         "Of which genuine countries": "1 (Kosovo)",
         "Rows by type": ", ".join(f"{k}: {v:,}" for k, v in q["entity_type_rows"].items())},
        "Regions, income groups and duplicate '(GCP)' regions sit in the same table as countries. Summing them together would double-count, so every analysis filters entity_type = country.",
        f"World total reconciles with Σ countries + international aviation & shipping to within 0.5% (tests/test_data.py::test_world_equals_countries_plus_international).",
        "src/co2ads/data.py::classify_entity")
    add("M1 Data Science Process", "Data cleaning: missing values",
        "All columns", r"\text{missing\%} = \frac{\#\text{NaN}}{N}\times 100",
        {"GDP last year": str(q["gdp_last_year"]), "Energy first year": str(q["energy_first_year"]),
         "Policy": "no deletion, no imputation"},
        "A blank means 'not reported', never zero emissions. Each analysis drops missing values only for the columns it uses (listwise per analysis), and reports its own n.",
        "tests/test_data.py::test_clean_keeps_every_row_and_value checks no value was imputed.",
        "src/co2ads/data.py::clean, cross_section")
    add("M1 Data Science Process", "Data cleaning: invalid values & outliers",
        "co2, fuel columns, population", r"x < 0 \text{ for non-negative columns};\ \ |co2 - \Sigma\text{fuels}|",
        {"Negative values in non-negative columns": str(q["negative_values_in_nonnegative_columns"]),
         "Zero-CO₂ country rows": f"{q['zero_co2_country_rows']} ({q['zero_co2_after_1950']} after 1950)",
         "Max |total − Σ fuel parts|": f"{q['max_fuel_sum_gap_mt']:.2f} Mt"},
        "No impossible values. Negative land-use or growth values are real (carbon sinks, falling emissions) and are kept. Early zeros are years before a country's record starts. Extreme values (e.g. Qatar) are genuine and are flagged, not deleted.",
        "Flags stored as columns (flag_negative_nonneg_col, flag_zero_co2, fuel_sum_gap).",
        "src/co2ads/data.py::clean")
    add("M1 Data Science Process", "Transformation: normalisation (min–max) & log scale",
        f"gdp_per_capita, co2_per_capita ({G})", r"x' = \frac{x - x_{min}}{x_{max} - x_{min}},\quad \log_{10} x",
        {"Used for": "LOF distances; regression on log scale",
         "Kurtosis β₂ (raw → log₁₀)": f"{f(s['beta2'])} → {f(dist['log_summary']['beta2'])}",
         "Pearson skewness (raw → log₁₀)": f"{f(s['pearson_sk_median'])} → {f(dist['log_summary']['pearson_sk_median'])}"},
        "Per-person CO₂ and GDP are strongly right-skewed with a heavy upper tail. Logging removes the heavy tail (β₂ close to the normal value 3, and the empirical rule holds), though a mild left skew remains from the very lowest emitters. Min–max scaling stops one unit from dominating a distance.",
        "Range of scaled column is exactly [0, 1].", "src/co2ads/regression.py::min_max, lof.py::lof_table")
    cv = reg["cv_multi"]
    ho = reg["holdout_multi"]
    add("M1 Data Science Process", "Train/test split & k-fold cross-validation",
        f"log CO₂pc ~ log GDPpc + log energy pc ({G})", r"\text{Accuracy}_{cv} = \frac{1}{k}\sum_{i=1}^{k}\text{score}_i",
        {"Hold-out (2/3 train)": f"train n={ho['n_train']}, test n={ho['n_test']}, test R² = {f(ho['test']['R2'], 3)}",
         "5-fold CV mean R²": f(cv["mean"]["R2"], 3), "5-fold CV mean RMSE (log₁₀)": f(cv["mean"]["RMSE"], 3)},
        "Cross-validated R² is close to the in-sample R², so the regression is not overfitting. Every country is used for validation exactly once.",
        "tests/test_cross_checks.py::test_kfold_uses_every_record_once; fixed seed 42.",
        "src/co2ads/regression.py::holdout, kfold_cv")
    bm = inf["bootstrap_median"]
    add("M1 Data Science Process", "Bootstrapping",
        pc, r"\text{resample } n \text{ values with replacement, } B = 5000 \text{ times}",
        {"Median": f(bm["estimate"], 3), "95% bootstrap CI": f"{f(bm['lower'], 3)} – {f(bm['upper'], 3)}",
         "Bootstrap SE": f(bm["boot_se"], 3)},
        "The median has no simple standard-error formula; bootstrapping gives an interval for it straight from the data.",
        "Same seed gives the same interval (tests/test_cross_checks.py::test_bootstrap_reproducible).",
        "src/co2ads/regression.py::bootstrap_ci")
    add("M1 Data Science Process", "Correlation vs causation",
        f"gdp_per_capita, co2_per_capita ({G})", r"r \neq \text{causation}",
        {"log–log r": f(rel["gdp_log"]["r"], 3)},
        "Richer countries emit more per person, but GDP does not 'cause' CO₂ by itself: energy use and energy mix sit in between (see the multiple regression). All findings are worded as associations.",
        "Multiple regression shows GDP loses significance once energy is controlled.",
        "src/co2ads/results.py::relationships")

    # ------------------------------------------------------------------ Module 2
    add("M2 Descriptive Statistics", "Types of data / levels of measurement",
        "country, continent, year, co2, co2_per_capita, growth %", r"\text{Nominal} \subset \text{Ordinal} \subset \text{Interval} \subset \text{Ratio}",
        {"Nominal": "country, iso_code, continent", "Interval": "year, growth %, temperature change",
         "Ratio": "co2, population, gdp, per-capita measures"},
        "Ratio variables support every statistic (means, ratios, CV). Interval ones (year) do not have a true zero, so 'twice as many years' is meaningless.",
        "Mapping shown on the Dataset Explorer page.", "src/co2ads/quality.py::measurement_level")
    add("M2 Descriptive Statistics", "Classification & frequency distribution",
        pc, r"\text{classes} \approx \sqrt{n};\ \text{exclusive intervals } [L, U)",
        {"Classes": str(s["classes"]), "Class width": f"{s['class_width']:g} t",
         "Countries in first class": f"{int(dist['freq_table']['f'].iloc[0])} ({f(dist['freq_table']['rel_f_pct'].iloc[0], 1)}%)"},
        "Most countries emit little per person, with a long tail of high emitters: a positively skewed distribution.",
        "Σf equals n (tests/test_cross_checks.py::test_grouped_formulas_consistent).",
        "src/co2ads/descriptive.py::frequency_table")
    add("M2 Descriptive Statistics", "Tabulation: two-way table",
        "continent × year → CO₂ per person", r"\text{cell} = \frac{\Sigma\,CO_2}{\Sigma\,\text{population}}",
        {"Rows × columns": f"{inf['two_way_matrix'].shape[0]} continents × {inf['two_way_matrix'].shape[1]} years"},
        "The two-way table is also the input of the two-way ANOVA (Module 3).",
        "Built from country rows only, so no aggregate is double-counted.", "src/co2ads/results.py::inference")
    add("M2 Descriptive Statistics", "Arithmetic mean (direct, short-cut, grouped step-deviation)",
        pc, r"\bar X = \frac{\Sigma X}{N};\quad \bar X = A + \frac{\Sigma f d'}{N}\times i",
        {"Mean (individual)": f(s["mean"], 3), "Grouped mean (direct)": f(grp["mean_direct"], 3),
         "Grouped mean (step deviation)": f(grp["mean_step"], 3),
         "Largest grouped-vs-raw gap": "{measure} {difference_pct:+.1f}%".format(**dist["grouped_vs_raw"].dropna().loc[lambda d: d["difference_pct"].abs().idxmax()])},
        "The grouped mean differs from the exact mean because it assumes values sit at class mid-points. The lower quantiles suffer most, because nearly half the countries fall in the first class. With raw data available, the individual-series value is the one to report.",
        "Short-cut = direct and step-deviation = direct (tests/test_cross_checks.py).", "src/co2ads/descriptive.py::mean_direct, grouped_mean")
    add("M2 Descriptive Statistics", "Weighted mean & combined mean",
        f"co2_per_capita weighted by population ({Y})", r"\bar X_w = \frac{\Sigma W X}{\Sigma W};\quad \bar X_{12} = \frac{N_1\bar X_1 + N_2 \bar X_2}{N_1+N_2}",
        {"Simple mean (typical country)": f(s["mean"], 3), "Population-weighted mean (typical person)": f(dist["population_weighted_mean"], 3),
         "Countries + international aviation & shipping": f(dist["mean_comparison"]["with_international"], 3),
         "World per capita (World row)": f(dist["world_per_capita"], 3),
         "Combined mean of 6 continent means": f(dist["combined_mean_check"], 3)},
        "The simple mean treats Tuvalu and India equally; the weighted mean gives the average for a person. The weighted value is a little below the World figure because international aviation and shipping belong to no country.",
        "Combined mean of continent groups reproduces the overall mean exactly.", "src/co2ads/descriptive.py::weighted_mean, combined_mean")
    add("M2 Descriptive Statistics", "Median & mode",
        pc, r"\text{Median} = \tfrac{N+1}{2}\text{th item};\ \ \text{Mode} = L + \frac{\Delta_1}{\Delta_1+\Delta_2}\times i",
        {"Median": f(s["median"], 3), "Grouped median": f(grp["median"], 3), "Grouped mode": f(grp["mode"]["mode"], 3),
         "Modal class": f"{grp['mode']['modal_class'][0]:g}–{grp['mode']['modal_class'][1]:g} t"},
        "Mode < Median < Mean: the order the course gives for a positively skewed distribution. Continuous values never repeat exactly, so the mode comes from the grouped formula.",
        "Grouped median lies within one class width of the exact median.", "src/co2ads/descriptive.py::median, grouped_mode")
    add("M2 Descriptive Statistics", "Quartiles, deciles, percentiles",
        pc, r"Q_k = \frac{k(N+1)}{4}\text{th item};\ D_k = \frac{k(N+1)}{10};\ P_k = \frac{k(N+1)}{100}",
        {"Q1": f(s["q1"], 3), "Q3": f(s["q3"], 3), "D1": f(s["d1"], 3), "D9": f(s["d9"], 3), "P90": f(s["p90"], 3)},
        f"A quarter of countries emit under {f(s['q1'])} t per person; the top 10% emit over {f(s['d9'])} t.",
        "Equals numpy's 'weibull' quantile method, which is the (N+1) rule (tests/test_cross_checks.py).", "src/co2ads/descriptive.py::quantile")
    add("M2 Descriptive Statistics", "Range, IQR, quartile deviation & coefficients",
        pc, r"R = L - S;\ \ \frac{L-S}{L+S};\ \ IQR = Q_3-Q_1;\ \ QD = \frac{Q_3-Q_1}{2};\ \ \frac{Q_3-Q_1}{Q_3+Q_1}",
        {"Range": f(s["range"], 3), "Coefficient of range": f(s["coef_range"], 3), "IQR": f(s["iqr"], 3),
         "Quartile deviation": f(s["quartile_deviation"], 3), "Coefficient of QD": f(s["coef_qd"], 3)},
        "The range is driven by a single country (Qatar); the IQR describes the middle half and is not affected by extremes.",
        "Hand formulas; quartiles cross-checked.", "src/co2ads/descriptive.py")
    add("M2 Descriptive Statistics", "Variance & standard deviation",
        pc, r"\sigma = \sqrt{\frac{\Sigma (X-\bar X)^2}{N}};\ \ \sigma = \sqrt{\frac{\Sigma d^2}{N} - \left(\frac{\Sigma d}{N}\right)^2}",
        {"σ (÷N, formula sheet)": f(s["std"], 3), "s (÷(n−1), sample)": f(s["std_sample"], 3),
         "Grouped σ (step deviation)": f(grp["std"], 3), "Coefficient of variation": f"{f(s['coef_variation_pct'], 1)}%"},
        "A CV above 100% means the spread is larger than the average itself: countries are extremely unequal emitters.",
        "Matches numpy.std (ddof 0 and 1); assumed-mean method equals actual-mean method.", "src/co2ads/descriptive.py::std, std_assumed_mean, grouped_std")
    add("M2 Descriptive Statistics", "Skewness: Karl Pearson & Bowley",
        pc, r"Sk_P = \frac{\bar X - \text{Mode}}{\sigma} \text{ or } \frac{3(\bar X - \text{Med})}{\sigma};\quad Sk_B = \frac{Q_3+Q_1-2\,\text{Med}}{Q_3-Q_1}",
        {"Pearson (mode)": f(s["pearson_sk_mode"], 3), "Pearson (3(mean−median)/σ)": f(s["pearson_sk_median"], 3),
         "Bowley": f(s["bowley_sk"], 3)},
        "All three are positive (right skew). Bowley is much smaller because it only looks at the middle 50% and ignores the long tail of extreme emitters.",
        "Signs agree with Mean > Median.", "src/co2ads/descriptive.py")
    add("M2 Descriptive Statistics", "Kurtosis β₂",
        pc, r"\beta_2 = \frac{\mu_4}{\mu_2^2}",
        {"β₂": f(s["beta2"], 3), "Shape": s["kurtosis_label"]},
        "β₂ far above 3: a sharp peak of low emitters plus heavy tails. (The course formula sheet's shape descriptions are swapped; the standard meaning is used here.)",
        "Equals scipy.stats.kurtosis(fisher=False, bias=True).", "src/co2ads/descriptive.py::kurtosis_beta2")
    add("M2 Descriptive Statistics", "Box plot & IQR outliers",
        pc, r"\text{fences} = Q_1 - 1.5\,IQR,\ Q_3 + 1.5\,IQR",
        {"Lower / upper limit": f"{f(box['lower_limit'])} / {f(box['upper_limit'])}",
         "High outliers": str(box["n_high_outliers"]), "Low outliers": str(box["n_low_outliers"]),
         "Examples": ", ".join(dist["outliers"]["country"].head(5))},
        "The flagged countries are genuine (mostly oil- and gas-exporting states, plus large energy-intensive economies such as Australia and the US), not data errors, so they are kept and discussed.",
        "Count reproduced on the Descriptive Statistics page for any year.", "src/co2ads/descriptive.py::box_plot, iqr_outliers")

    ux, cl = res["univariate_extras"], res["classification"]
    geo = cl["geographical"]
    add("M2 Statistical Data", "Classification: geographical, chronological, qualitative, quantitative",
        f"continent, year, change since {B}, largest fuel source, co2_per_capita ({Y})", r"\text{group facts into classes sharing one characteristic}",
        {"Geographical": f"Asia {f(geo.loc['Asia', 'share_of_world_pct'], 1)}% of world CO₂ from {int(geo.loc['Asia', 'countries'])} countries",
         "Chronological": f"world mean annual CO₂ by decade, 1950s → 2020–{Y}",
         "Qualitative (two-fold)": ", ".join(f"{k}: {v}" for k, v in cl["two_fold"]["countries"].items()),
         "Qualitative (manifold)": "continent → fell/rose → largest fuel source",
         "Quantitative": "5 magnitude classes of CO₂ per person"},
        "Each type answers a different question: where (continent), when (decade), what kind (attribute) and how much (magnitude).",
        "Class counts sum to the number of countries; continent totals use countries only.", "src/co2ads/results.py::classification_and_tabulation")
    add("M2 Statistical Data", "Tabulation: simple, two-way and higher-order tables",
        f"continent × CO₂-per-person band × change since {B}", r"\text{stub} \times \text{caption} \rightarrow \text{body} + \text{totals}",
        {"Simple table": f"{len(cl['simple_table']) - 1} classes, total {cl['n']}", "Two-way": "6 continents × 5 classes with margins",
         "Higher-order": "continent × fell/rose × 5 classes"},
        f"In the two-way table, {cl['two_way_table'].loc['Africa', '< 1 t']} of Africa's {cl['two_way_table'].loc['Africa', 'Total']} countries emit under 1 t per person; most European countries sit between 3 and 10 t.",
        "Row and column totals reproduce the one-way table (pandas crosstab with margins).", "src/co2ads/results.py::classification_and_tabulation")
    dts, dsd = ux["discrete_summary"], ux["discrete_std_methods"]
    add("M2 Descriptive Statistics", "Discrete series: mean, median, mode, quartiles, σ",
        f"{ux['discrete_variable']} per country (n = {dts['n']})",
        r"\bar X = \frac{\Sigma fX}{N};\ \text{Median} = X \text{ at } cf \ge \tfrac{N+1}{2};\ \sigma = \sqrt{\tfrac{\Sigma fd^2}{N} - \left(\tfrac{\Sigma fd}{N}\right)^2}",
        {"Mean (direct = short-cut)": f"{f(dts['mean'], 3)} = {f(dts['mean_shortcut'], 3)}", "Median / mode": f"{dts['median']:g} / {', '.join(f'{v:g}' for v in dts['mode'])}",
         "Q1 / Q3": f"{dts['q1']:g} / {dts['q3']:g}", "σ (actual = assumed mean)": f"{f(dsd['actual_mean_method'], 3)} = {f(dsd['assumed_mean_method'], 3)}"},
        f"A typical country saw its emissions fall in about 7 of the 19 years; mean, median and mode are close, so this count is roughly symmetric, unlike per-person emissions.",
        "Reproduces the course examples (marks mean 41, median income ₹16,800, P20 = Q1 = D3 = 20, Q3 = 40).", "src/co2ads/descriptive.py::discrete_*")
    add("M2 Statistical Data", "Inclusive → exclusive classes; class limits from mid-values",
        "the discrete count grouped as 0–3, 4–7, …", r"L' = L - \tfrac{1}{2}\,\text{gap},\ U' = U + \tfrac{1}{2}\,\text{gap};\ \ L = m - \tfrac{i}{2},\ U = m + \tfrac{i}{2}",
        {"Classes": ", ".join(f"{a} → {b}" for a, b in zip(ux["inclusive_table"]["inclusive_class"], ux["inclusive_table"]["exclusive_class"])),
         "Grouped median (exclusive classes)": f(ux["inclusive_grouped_median"], 3)},
        "Whole-number classes leave gaps (3 to 4); converting to exclusive classes makes the series continuous so the interpolation formulas apply.",
        "Tested on simple inclusive classes and the slide's mid-value example (115 → 110–120).", "src/co2ads/descriptive.py::inclusive_to_exclusive, classes_from_midpoints")
    add("M2 Descriptive Statistics", "Cumulative frequency ('less than' / 'more than') & ogive",
        pc, r"\text{Median} = \text{x where the two ogives cross} = L + \frac{N/2 - cf}{f}\times i",
        {"Ogive intersection (median)": f(ux["ogive_median"], 4), "Formula median": f(ux["formula_median"], 4),
         "Q1 / Q3 read from ogive": f"{f(ux['ogive_q1'], 3)} / {f(ux['ogive_q3'], 3)}"},
        "The graphical and formula medians agree exactly, because the ogive interpolates linearly inside the median class just as the formula does.",
        "Tested on the course 'median from cumulative data' example. The slide uses i = 10 for classes 5 wide and prints 14.29; the correct median is 12.15.",
        "src/co2ads/descriptive.py::cumulative_frequencies, ogive_value, ogive_intersection")
    add("M2 Descriptive Statistics", "Mean with open-end classes",
        pc + f"; last class '{ux['open_from']:g} t and above'", r"\text{assume open class width} = \text{neighbour's width, then } \bar X = \frac{\Sigma fm}{N}",
        {"Mean (open class assumed)": f(ux["open_mean"], 3), "Grouped mean (all closed)": f(ux["closed_grouped_mean"], 3),
         "Exact mean": f(ux["actual_mean"], 3), "Median (unchanged)": f(ux["open_median"], 3)},
        "The open-class mean depends on an assumption about the missing limit; the median does not. With an open top class in skewed data, report the median, as the course advises.",
        "Tested on the course example ('Below 10' … 'Above 50').", "src/co2ads/descriptive.py::open_end_mean, close_top_classes")
    cm = ux["correction"]
    add("M2 Descriptive Statistics", "Correcting incorrect items in a mean",
        f"co2_per_capita {Y}: {cm['n_all']} rows incl. {cm['n_wrong']} aggregate rows", r"\Sigma X_{correct} = N\bar X - \Sigma X_{wrong} + \Sigma X_{right}",
        {"Incorrect mean (aggregates included)": f(cm["incorrect_mean"], 4), "Corrected mean": f"{f(cm['correct_mean'], 4)} (N = {cm['n']})"},
        "If regional totals are averaged in with countries by mistake, the course correction formula recovers the countries-only mean exactly: a concrete reason the cleaning step matters.",
        "Reproduces Illustrations 6 and 7 (39.7 and 40.18).", "src/co2ads/descriptive.py::corrected_mean")
    mo = ux["moments"]
    add("M2 Descriptive Statistics", "Moments (about origin and mean), β₁, β₂ & empirical mode",
        pc, r"\mu'_r = \frac{\Sigma X^r}{N};\ \mu_r = \frac{\Sigma (X-\bar X)^r}{N};\ \beta_1 = \frac{\mu_3^2}{\mu_2^3};\ \beta_2 = \frac{\mu_4}{\mu_2^2};\ \text{Mode} \approx 3\,\text{Med} - 2\,\bar X",
        {"μ₂ / μ₃ / μ₄": f"{f(mo['central']['mu2'])} / {f(mo['central']['mu3'], 1)} / {f(mo['central']['mu4'], 0)}",
         "β₁ / γ₁": f"{f(mo['beta1'], 3)} / {f(mo['gamma1'], 3)}", "β₂ / γ₂": f"{f(mo['beta2'], 3)} / {f(mo['gamma2'], 3)}",
         "Empirical vs grouped mode": f"{f(ux['empirical_mode'], 3)} vs {f(ux['grouped_mode'], 3)}"},
        "μ₃ > 0 confirms right skew and μ₄ the heavy tail. The empirical mode relation only holds for moderate skew; here it is far from the grouped mode, another sign of extreme skewness.",
        "Central moments computed directly and from raw moments agree; γ₁ and β₂ match SciPy.", "src/co2ads/descriptive.py::moments, empirical_mode")
    gr, gl = rel["gdp_raw"], rel["gdp_log"]
    add("M2 Multivariate Exploration", "Central data point",
        f"gdp_per_capita, co2_per_capita ({G}, n = {gr['n']})", r"\bar{\mathbf{x}} = (\bar X_1, \bar X_2, \dots)",
        {"Centre": f"GDP pc = {f(rel['central_point']['gdp_per_capita'], 0)}, CO₂ pc = {f(rel['central_point']['co2_per_capita'])}",
         "Farthest from centre": ", ".join(rel["farthest_from_centre"]["country"].head(3))},
        "Distance from the multivariate centre (in SD units) is a first, simple outlier screen.",
        "Means computed directly.", "src/co2ads/correlation.py::central_data_point")
    add("M2 Multivariate Exploration", "Covariance & Karl Pearson r (3 methods)",
        f"gdp_per_capita vs co2_per_capita ({G}, n = {gr['n']})",
        r"r = \frac{\Sigma xy}{\sqrt{\Sigma x^2\,\Sigma y^2}} = \frac{N\Sigma XY - \Sigma X\Sigma Y}{\sqrt{N\Sigma X^2-(\Sigma X)^2}\sqrt{N\Sigma Y^2-(\Sigma Y)^2}}",
        {"r (raw)": f"{f(gr['r'], 3)} — {gr['degree']}", "r (log–log)": f"{f(gl['r'], 3)} — {gl['degree']}",
         "Probable error (log)": f(gl["probable_error"], 4), "r²(log)": f(gl["r_squared"], 3)},
        "A strong positive relationship that is clearer on the log scale, because both variables are skewed. r² = share of variation in one explained by the other.",
        "Actual-mean, direct and assumed-mean methods agree to 1e-12; equal to numpy.corrcoef.", "src/co2ads/correlation.py::analyse")
    cf_ = res["correlation_forms"]
    add("M2 Multivariate Exploration", "Types of correlation: positive, negative, none, non-linear",
        f"cross-section {G} (n = {cf_['n']}), UK 1990–{Y}, population vs CO₂ pc {Y}", r"\text{sign and shape of the scatter}",
        {"Positive (log GDP vs log CO₂ pc)": f(cf_["positive"]["r"], 3), "Negative (log GDP vs log CO₂ per kWh)": f(cf_["negative_cross_section"]["r"], 3),
         "Negative over time (UK CO₂ vs year)": f(cf_["negative_time"]["r"], 3), "None (population vs CO₂ pc)": f(cf_["none"]["r"], 3),
         "Non-linear (raw GDP vs CO₂ pc)": f(cf_["nonlinear_raw"]["r"], 3)},
        "Richer countries emit more per person but have cleaner energy per kWh. Population says nothing about per-person emissions. Raw GDP vs CO₂ is curved, so r understates the link until both are logged.",
        "Each r computed with the course method; signs match the scatter diagrams.", "src/co2ads/results.py::correlation_forms")
    add("M2 Multivariate Exploration", "Simple, partial & multiple correlation",
        f"log₁₀ CO₂ pc (y), log₁₀ GDP pc (x₁), log₁₀ energy pc (x₂), {G}",
        r"r_{y1\cdot2} = \frac{r_{y1} - r_{y2}r_{12}}{\sqrt{(1-r_{y2}^2)(1-r_{12}^2)}};\ R^2_{y\cdot12} = \frac{r_{y1}^2 + r_{y2}^2 - 2r_{y1}r_{y2}r_{12}}{1-r_{12}^2}",
        {"Simple r(y, GDP) / r(y, energy)": f"{f(cf_['simple_r_gdp'], 3)} / {f(cf_['simple_r_energy'], 3)}",
         "Partial r(y, GDP | energy)": f(cf_["partial_gdp_given_energy"], 3), "Partial r(y, energy | GDP)": f(cf_["partial_energy_given_gdp"], 3),
         "Multiple R": f(cf_["multiple_R"], 3)},
        "Holding energy constant removes almost all of GDP's link with emissions, the correlation-side view of the regression's multicollinearity result. Multiple R² equals the regression R².",
        "Partial r equals the correlation of residuals (statsmodels); multiple R equals √R² of OLS.", "src/co2ads/correlation.py::partial_correlation, multiple_correlation")
    add("M2 Data Visualization", "Diagrams & graphs: bars (simple, multiple, sub-divided, percentage, deviation, broken), rectangle, square, circle, pie, pictogram, cartogram, line (incl. false baseline), histogram, frequency polygon, ogive, box plot, distribution chart, scatter, scatter matrix, bubble",
        "various", r"\text{diagram chosen by question; area} \propto \text{value} \Rightarrow \text{side} = \sqrt{v},\ r = \sqrt{v/\pi}",
        {"Charts in dashboard": "all of these, with units and tooltips",
         "Rectangle diagram": "width = population, height = CO₂ per person, area = total CO₂"},
        "Line for trends, bars for comparisons, two-dimensional shapes when area should carry the value, histogram/box/ogive for distributions, scatter/bubble/scatter matrix for relationships.",
        "Every chart is drawn from computed data, never hard-coded.", "app/pages/3_Classification_and_Diagrams.py and others")

    # ------------------------------------------------------------------ Module 3
    er, el = inf["empirical_raw"], inf["empirical_log"]
    add("M3 Inferential Statistics", "Normal distribution, z-scores & empirical rule",
        pc + " and its log₁₀", r"f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}};\ z = \frac{x-\mu}{\sigma}",
        {"Within ±1σ raw / log / normal": f"{f(er.iloc[0]['observed_pct'], 1)}% / {f(el.iloc[0]['observed_pct'], 1)}% / 68.3%",
         "Within ±2σ raw / log / normal": f"{f(er.iloc[1]['observed_pct'], 1)}% / {f(el.iloc[1]['observed_pct'], 1)}% / 95.4%"},
        "Raw per-person emissions are far from normal; their logarithm is close to normal (log-normal distribution). This is why tests and regression use the log scale.",
        "Normal percentages from scipy.stats.norm.", "src/co2ads/inference.py::empirical_rule, normal_pdf")
    pois = inf["poisson"]
    add("M3 Inferential Statistics", "Poisson distribution (suitability check)",
        "Yearly count of countries (≥1 Mt) whose CO₂ fell by >10%, 1991–2024", r"P(X=x)=\frac{e^{-\mu}\mu^x}{x!};\ \ \text{Poisson} \Rightarrow \text{mean} = \text{variance}",
        {"Mean": f(pois["mean"]), "Variance": f(pois["variance"]), "Dispersion index (var/mean)": f(pois["dispersion_index"]),
         "p (dispersion test)": f"{pois['p_value']:.1e}"},
        "Variance is about five times the mean, so these counts are NOT Poisson: drops cluster in crisis years (2020: 42 countries, 2009: 27, 1991–94 post-Soviet collapse: 17–23 a year) instead of happening independently at a constant rate.",
        "Index-of-dispersion χ² test, n − 1 d.f.", "src/co2ads/inference.py::poisson_check")
    clt = inf["clt"]
    add("M3 Inferential Statistics", "Central limit theorem",
        pc + "; 10,000 samples of n = 30", r"\bar X \sim N\!\left(\mu, \frac{\sigma}{\sqrt n}\right) \text{ for large } n",
        {"Population mean / mean of sample means": f"{f(clt['population_mean'], 3)} / {f(clt['mean_of_means'], 3)}",
         "σ/√n vs SD of sample means": f"{f(clt['theoretical_se'], 3)} / {f(clt['sd_of_means'], 3)}",
         "Skewness: population → sample means": f"{f(clt['population_skew'], 2)} → {f(clt['skew_of_means'], 2)}"},
        "Even from this strongly skewed population, sample means are nearly normal and their spread matches σ/√n: the CLT on real data.",
        "Seeded simulation (seed 42).", "src/co2ads/inference.py::clt_simulation")
    ci = inf["ci_mean"]
    add("M3 Inferential Statistics", "Confidence interval for a mean",
        pc, r"\bar x \pm z\frac{s}{\sqrt n}",
        {"95% CI": f"{f(ci['lower'], 3)} – {f(ci['upper'], 3)} t", "Margin": f(ci["margin"], 3), "Method": f"{ci['method']} (n ≥ 30)"},
        "Read as: if 2024 countries were a sample from the process that generates national emissions, the mean per-person value lies in this range with 95% confidence. The 214 countries are a near-complete set, so this is a model-based interpretation.",
        "Reproduces the course example (45 ± 2.77) in tests.", "src/co2ads/inference.py::ci_mean")
    z = inf["z_test"]
    add("M3 Inferential Statistics", "Z-test (single mean, large sample)",
        pc, r"Z = \frac{\bar x - \mu_0}{s/\sqrt n}",
        {"H₀": f"μ = world per-person value ({f(z['mu0'], 3)} t)", "Z": f(z["statistic"], 3),
         "Critical |Z| (5%)": f(z["critical"], 2), "p": f(z["p_value"], 3), "Decision": z["decision"]},
        "The average country is not significantly different from the average person at 5%. Large spread between countries makes the test weak.",
        "Reproduces the oceanographer example (Z = 2.067) in tests.", "src/co2ads/inference.py::z_test_mean")
    pt = inf["paired"]
    add("M3 Inferential Statistics", "Paired t-test",
        f"co2_per_capita {B} vs {Y}, same countries (n = {pt['n']})", r"t = \frac{\bar d}{s_d/\sqrt n},\ \ df = n-1",
        {"Mean before → after": f"{f(pt['mean_before'], 3)} → {f(pt['mean_after'], 3)} t",
         "Mean difference": f(pt["mean_diff"], 3), "t": f(pt["statistic"], 3), "p": f(pt["p_value"], 4),
         "Countries down / up": f"{pt['n_decreased']} / {pt['n_increased']}", "Decision": pt["decision"]},
        "Average per-person emissions fell significantly since 2005, yet more countries rose than fell: large falls in high emitters (Europe, North America) outweigh many small rises.",
        "Reproduces the soldiers example (|t| = 2.16); p matches scipy.ttest_rel.", "src/co2ads/inference.py::t_test_paired")
    it = inf["independent"]
    add("M3 Inferential Statistics", "Independent two-sample t-test",
        f"log₁₀ CO₂ per person, Europe (n={it['n1']}) vs Asia (n={it['n2']}), {Y}", r"t = \frac{\bar x - \bar y}{\sqrt{s^2/n_1 + s^2/n_2}}",
        {"t": f(it["statistic"], 3), "df": str(it["df"]), "p (pooled)": f(it["p_value"], 3),
         "p (Welch, sensitivity)": f(inf["independent_welch_p"], 3), "Variance ratio": f(it["variance_ratio"], 1), "Decision": it["decision"]},
        f"Typical (geometric-mean) values: Europe {f(inf['independent_back']['europe_geo_mean'])} t vs Asia {f(inf['independent_back']['asia_geo_mean'])} t. The difference is not significant at 5%, and the large variance ratio means the pooled-variance assumption is questionable, so Welch's version is also reported.",
        "Reproduces the diet example; p matches scipy.ttest_ind.", "src/co2ads/inference.py::t_test_independent")
    er_ = inf["errors"]
    add("M3 Inferential Statistics", "Type I & Type II errors",
        pc + f"; 10,000 simulated samples of n = {er_['n']}", r"\alpha = P(\text{reject } H_0 | H_0 \text{ true}),\ \beta = P(\text{accept } H_0 | H_0 \text{ false})",
        {"Nominal α": "0.05", "Observed Type I rate": f(er_["type1_rate"], 3),
         f"Type II rate β (true mean shifted by {er_['shift']} t)": f(er_["type2_rate"], 3), "Power": f(er_["power"], 3)},
        "With skewed data and n = 40 the z-test rejects a true H₀ more often than 5%: the normal approximation is not yet accurate. β shows how often a real 1.5 t difference would be missed.",
        "Type I rate converges toward α as n grows (try it on the Statistical Analysis page).", "src/co2ads/inference.py::error_rate_simulation")
    an = inf["anova"]
    add("M3 Inferential Statistics", "One-way ANOVA",
        f"log₁₀ CO₂ per person by continent ({Y}, k = 6, n = {an['df_between'] + an['df_within'] + 1})",
        r"F = \frac{SS_B/(k-1)}{SS_W/(n-k)}",
        {"F": f(an["F"], 2), "F critical (5%)": f(an["critical"], 2), "p": f"{an['p_value']:.1e}",
         "η² (SS_B/SS_T)": f(an["eta_squared"], 3), "Decision": an["decision"],
         "Grand mean vs mean of means": f"{f(an['grand_mean'], 3)} vs {f(an['mean_of_means'], 3)}"},
        "Continent explains a large share (η²) of the variation in log emissions per person. Group sizes differ, so the grand mean is weighted by n (the slides' 'mean of means' only works for equal groups).",
        "Reproduces the wheat example (F = 1.5); matches scipy.f_oneway; SS_T = SS_B + SS_W.", "src/co2ads/inference.py::one_way_anova")
    tw = inf["two_way"]
    add("M3 Inferential Statistics", "Two-way ANOVA (without replication)",
        "continental CO₂ per person: continent (rows) × year (columns)", r"SS_C = \sum\frac{T_j^2}{n_j} - \frac{T^2}{n},\ SS_R = \sum\frac{T_i^2}{n_i} - \frac{T^2}{n}",
        {"F (continent)": f"{f(tw['F_rows'], 2)} (crit {f(tw['crit_rows'], 2)}) → {tw['decision_rows']}",
         "F (year)": f"{f(tw['F_columns'], 2)} (crit {f(tw['crit_columns'], 2)}) → {tw['decision_columns']}"},
        "Differences between continents are significant; differences between years are not, because continents moved in opposite directions (Asia up, Europe and North America down), which ends up in the residual.",
        "Matches statsmodels anova_lm (tests).", "src/co2ads/inference.py::two_way_anova")

    # ------------------------------------------------------------------ Module 4
    sm, mm = reg["simple"], reg["multiple"]
    add("M4 Regression", "Simple linear regression: slope & intercept",
        f"y = log₁₀ CO₂ pc, x = log₁₀ GDP pc ({G}, n = {sm['n']})",
        r"a = \frac{n\Sigma xy - \Sigma x\Sigma y}{n\Sigma x^2 - (\Sigma x)^2},\ \ b = \frac{\Sigma y - a\Sigma x}{n}",
        {"Slope": f(sm["slope"], 4), "Intercept": f(sm["intercept"], 4), "t (slope)": f(sm["t_slope"], 2),
         "R²": f(sm["metrics"]["R2"], 3)},
        f"On log scales the slope is an elasticity: a 1% higher GDP per person goes with about {f(sm['slope'])}% higher CO₂ per person, on average across countries (association, not causation).",
        "Reproduces the expenditure example (a = 8.4, b = 3.2); matches statsmodels OLS.", "src/co2ads/regression.py::simple_regression")
    add("M4 Regression", "Error measures: MAE, MSE, RMSE, MAPE, R²",
        "same model", r"MAE=\tfrac1n\Sigma|y-\hat y|,\ RMSE=\sqrt{\tfrac1n\Sigma(y-\hat y)^2},\ MAPE=\tfrac{100}{n}\Sigma\left|\tfrac{y-\hat y}{y}\right|,\ R^2 = 1-\tfrac{SS_{res}}{SS_{tot}}",
        {"MAE (log₁₀)": f(sm["metrics"]["MAE"], 3), "RMSE (log₁₀)": f(sm["metrics"]["RMSE"], 3),
         "MAPE on original tonnes": f"{f(reg['back_transformed']['simple_mape_original'], 1)}%"},
        "MAPE is computed after converting predictions back to tonnes, because on the log scale values near 0 make MAPE meaningless (a limitation the slides mention).",
        "Hand formulas; R² matches statsmodels.", "src/co2ads/regression.py::all_metrics")
    c1, c2 = mm["coefficients"]
    add("M4 Regression", "Multiple linear regression & partial regression coefficients",
        f"y = log₁₀ CO₂ pc; X₁ = log₁₀ GDP pc; X₂ = log₁₀ energy pc ({G}, n = {mm['n']})",
        r"b_1 = \frac{\Sigma x_2^2\Sigma x_1y - \Sigma x_1x_2\Sigma x_2y}{\Sigma x_1^2\Sigma x_2^2 - (\Sigma x_1x_2)^2},\ b_0 = \bar Y - b_1\bar X_1 - b_2\bar X_2",
        {"Equation": f"Ŷ = {f(mm['b0'], 3)} {mm['b1']:+.3f}·X₁ {mm['b2']:+.3f}·X₂",
         "R² / adjusted R²": f"{f(mm['r2'], 3)} / {f(mm['adj_r2'], 3)}", "r₁₂ (X₁ vs X₂)": f(mm["r12"], 3)},
        "Holding energy use constant, GDP adds almost nothing; holding GDP constant, 1% more energy per person goes with about 1% more CO₂ per person. GDP and energy are highly correlated (multicollinearity), so the GDP coefficient is unstable.",
        "Reproduces the course example (b1 = 3.1479, b2 = −1.6561, R² = 0.9626); matches statsmodels.", "src/co2ads/regression.py::multiple_regression_2")
    add("M4 Regression", "Overall significance (F-test) & individual coefficients (t-test)",
        "same model", r"F = \frac{(n-k-1)R^2}{k(1-R^2)};\quad t = \frac{b_i}{SE(b_i)},\ SE(b_i)=\sqrt{\frac{RSS/(n-k-1)}{\Sigma(x_i-\bar x_i)^2(1-r_{12}^2)}}",
        {"F": f"{f(mm['F'], 1)} (crit {f(mm['F_critical'], 2)}) → {mm['F_decision']}",
         "t(b₁ GDP)": f"{f(c1['t'], 2)}, p = {f(c1['p_value'], 3)} → {'significant' if c1['significant'] else 'not significant'}",
         "t(b₂ energy)": f"{f(c2['t'], 2)}, p = {c2['p_value']:.1e} → {'significant' if c2['significant'] else 'not significant'}"},
        "The model as a whole is highly significant (F-test first, as the slides require); individually only energy per person is significant at 5%.",
        "SE and t match statsmodels (tests/test_cross_checks.py). Course example: F = 64.37 vs F₀.₀₅(2, 5) = 5.79; t(b₁) = 3.753. "
        "Slide 40 copies b₁'s working for b₂; the correct values are SE = 0.976, t = −1.697 (still not significant).",
        "src/co2ads/regression.py::multiple_regression_2")
    cmp, col, pr = reg["comparison"], reg["collinearity"], reg["prediction"]
    add("M4 Regression", "Goodness of fit & overfit: R, R², adjusted R² (simple vs multiple)",
        "simple (X₁ = GDP) vs multiple (X₁ = GDP, X₂ = energy)",
        r"R^2 = \frac{b_1\Sigma x_1y + b_2\Sigma x_2y}{\Sigma y^2},\ \ \bar R^2 = 1-(1-R^2)\frac{n-1}{n-k-1},\ \ R=\sqrt{R^2}",
        {"R² simple → multiple": f"{f(sm['r2'], 3)} → {f(mm['r2'], 3)} (Δ {cmp['delta_r2']:+.3f})",
         "Adjusted R²": f"{f(sm['adj_r2'], 3)} → {f(mm['adj_r2'], 3)}",
         "F simple / multiple": f"{f(sm['F'], 1)} / {f(mm['F'], 1)}",
         "RMSE (log₁₀)": f"{f(sm['metrics']['RMSE'], 3)} → {f(mm['metrics']['RMSE'], 3)}"},
        "Adjusted R² rises as well as R², so energy adds real explanatory power beyond the extra parameter. In the simple model F = t² of the slope.",
        "Simple-model R², adjusted R², F and SSE match statsmodels; F = t² checked.", "src/co2ads/regression.py::compare_models")
    add("M4 Regression", "Overlap between predictors (multicollinearity flag)",
        "X₁ = log₁₀ GDP pc, X₂ = log₁₀ energy pc",
        r"\text{tolerance}=1-r_{12}^2,\ \ VIF=\frac{1}{1-r_{12}^2},\ \ SE(b_i)\propto\sqrt{VIF}",
        {"r₁₂": f(col["r12"], 3), "Tolerance / VIF": f"{f(col['tolerance'], 3)} / {f(col['vif'], 2)}",
         "Flag": f"{'YES' if col['flagged'] else 'no'} ({col['level']})",
         "b₁ (GDP) alone → with energy": f"{f(cmp['b1_simple'], 3)} → {f(cmp['b1_multiple'], 3)}"
                                        + (" (sign flips)" if cmp["b1_sign_flip"] else "")},
        f"Only {f(col['tolerance'] * 100, 0)}% of the variation in log GDP is not already carried by log energy, so the (1 − r₁₂²) term in SE(bᵢ) "
        f"multiplies the standard errors by √VIF = {f(col['se_inflation'], 1)}. That is why GDP is significant alone but not alongside energy, and its sign flips. "
        "Fit and predictions are fine; only the split of credit between the two predictors is unreliable.",
        "VIF matches statsmodels variance_inflation_factor; course example r₁₂ = −0.88 → VIF = 4.56.",
        "src/co2ads/regression.py::collinearity")
    add("M4 Regression", "Estimate for unknown values (prediction)",
        f"median country: GDP pc = {f(pr['gdp_per_capita'], 0)} int-$, energy pc = {f(pr['energy_per_capita'], 0)} kWh",
        r"\hat Y = b_0 + b_1X_1 + b_2X_2,\ \ \widehat{CO_2} = 10^{\hat Y}",
        {"Simple model": f"{f(pr['co2_pc_simple'])} t per person", "Multiple model": f"{f(pr['co2_pc_multi'])} t per person"},
        "Predictions are made on the log₁₀ scale and converted back to tonnes.",
        "Reproduces slide 32 (X₁ = 74, X₂ = 14 → Ŷ = 202.893).", "src/co2ads/regression.py::predict")

    # ------------------------------------------------------------------ Module 5
    top_l = lof["table"].head(5)
    add("M5 Anomaly Detection", "Local Outlier Factor (Manhattan distance)",
        f"min–max scaled log₁₀ GDP pc & log₁₀ CO₂ pc ({G}, n = {lof['n']}), k = {lof['k']}",
        r"\text{reach}(p,o)=\max(k\text{-dist}(o), d(p,o));\ lrd(p)=\frac{|N_k(p)|}{\Sigma\,\text{reach}};\ LOF(p)=\frac{\overline{lrd(N_k(p))}}{lrd(p)}",
        {"Top 5 by LOF": ", ".join(f"{c} ({v:.2f})" for c, v in zip(top_l["country"], top_l["lof"])),
         "LOF > 1.5": str(lof["n_flagged"]), "Also IQR outliers (of top 10)": ", ".join(lof["lof_top10_also_iqr"]) or "none"},
        "LOF finds countries that are unusual for their neighbourhood: very high emitters for their income (Qatar), and also low emitters for their income (Switzerland, Sweden, North Korea's low GDP with relatively high CO₂). The IQR rule only sees one variable and only high values.",
        f"Reproduces the 4-point slide example (LOF(d) = 2.0). On these {lof['sklearn_check']['n']} countries it matches sklearn LocalOutlierFactor(metric='manhattan') to {lof['sklearn_check']['max_abs_diff']:.0e} ({lof['sklearn_check']['n_tied_neighbourhoods']} tied neighbourhoods, same top 10: {lof['sklearn_check']['same_top10']}).",
        "src/co2ads/lof.py::lof, sklearn_check")
    vi = lof["vs_iqr"].set_index("group")
    add("M5 Anomaly Detection", "LOF vs the IQR (box-plot) rule",
        f"LOF > {lof['threshold']} on both variables vs IQR fences on CO₂ per person alone ({G}, n = {lof['n']})",
        r"\text{IQR outlier: } x<Q_1-1.5\,IQR \ \text{or}\ x>Q_3+1.5\,IQR",
        {**{g: f"{r['n']}" + (f" ({r['countries']})" if r["countries"] else "") for g, r in vi.iterrows()},
         "Flagged at every k (5, 10, 15, 20)": ", ".join(lof["flagged_at_every_k"])},
        "The methods agree only on Qatar. The IQR-only countries (Gulf states, Australia, US) are high emitters but sit in a dense cluster of rich, high-CO₂ countries, so their local density is normal (LOF close to 1). LOF-only countries are unusual for their income, including low emitters the one-sided IQR rule cannot see.",
        "IQR fences use the course quartile rule k(N+1)/4; the LOF threshold 1.5 is a convention, so k = 5–20 is reported as a sensitivity check.",
        "src/co2ads/results.py::outliers_lof")

    # ------------------------------------------------------------------ Module 6
    comp = tsr["comparison"]
    best = comp["MASE"].idxmin()
    add("M6 Time Series", "Components & classical decomposition",
        "World CO₂ 1950–2024 (annual)", r"Y = T + S + R \ \text{or}\ Y = T\times S\times R;\ T = \text{centred MA}",
        {"Seasonal component": "not applicable (annual data, no within-year seasons)",
         "Trend": "centred 5-year moving average",
         "Validated on": "course quarterly-sales example (indices −10.44, −11.69, 6.06, 16.06)"},
        "Seasonality needs sub-annual data. On annual CO₂ only the trend and irregular parts exist; the years in which world emissions fell match known shocks: 1974–75 and 1980–82 (oil crises), 1992 (post-Soviet collapse), 2009 (financial crisis) and 2020 (COVID, −5.2%).",
        "Additive and multiplicative decomposition reproduce the slide example exactly (tests).", "src/co2ads/timeseries.py::classical_decomposition, centred_moving_average")
    add("M6 Time Series", "Average, moving-average & naïve forecasts; least-squares trend",
        "World CO₂; train ≤ 2019, test 2020–2024", r"F_t=\tfrac{1}{t-1}\Sigma_{i<t} y_i;\ \ F_t=\tfrac1k\Sigma_{i=t-k}^{t-1}y_i;\ \ \hat y = a\,t + b",
        {m: f"MAE {f(r['MAE'], 0)} Mt, MAPE {f(r['MAPE'], 2)}%, MASE {f(r['MASE'], 2)}" for m, r in comp.iterrows()},
        f"Best on the hold-out by MASE: {best}. The 2000–2019 linear trend overshoots because emissions growth slowed after 2013 and dropped in 2020. The average method is useless for a trending series, exactly as the slides warn.",
        "Reproduces slide metrics (MAE 2.32, MASE 1.16 / 2.67, 1.14).", "src/co2ads/timeseries.py::holdout_comparison")
    add("M6 Time Series", "Forecast evaluation: MAE, RMSE, MAPE, MASE",
        "same", r"MASE = \frac{MAE}{\frac{1}{n-1}\sum_{j=2}^{n}|y_j-y_{j-1}|}",
        {"MASE < 1 means": "better than the naïve forecast", "Scaling series": "training period 1950–2019"},
        "MASE above 1 for every method shows that, over 2020–2024, simply carrying forward last year's value is hard to beat for annual emissions.",
        "Hand formulas; slide examples reproduced.", "src/co2ads/timeseries.py::evaluate, mase")
    fx, cv = tsr["comparison_fixed"], tsr["covid"]
    best_fx = fx["MASE"].idxmin()
    add("M6 Time Series", "True hold-out (fixed origin) and the 2020 COVID shock",
        "World CO₂; every forecast made from data up to 2019 only",
        r"\hat y_{2019+h} = \bar y_{1950..2019}\ \text{(average)},\ \tfrac13(y_{2017}+y_{2018}+y_{2019})\ \text{(MA)},\ y_{2019}\ \text{(naïve)}",
        {**{m: f"MAE {f(r['MAE'], 0)} Mt, MAPE {f(r['MAPE'], 2)}%, MASE {f(r['MASE'], 2)}" for m, r in fx.iterrows()},
         "2020 drop": f"{f(cv['change'], 0)} Mt ({f(cv['change_pct'], 1)}%); back above 2019 in {cv['back_above_2019_in']}"},
        f"One-step-ahead forecasts (course definition) see each hold-out year once it has happened; the fixed-origin test does not. Best by MASE: {best_fx}. No method anticipates the 2020 fall (2020 is one step ahead in both tests, so each method's 2020 error is the same in both), and the trend line, which assumes growth continues, is furthest off. Emissions rebounded to the 2019 level, so a flat forecast from 2019 did well afterwards.",
        "Forecasts are unchanged when hold-out values are altered (test); trend errors identical in both tables.",
        "src/co2ads/timeseries.py::fixed_origin_forecasts, shock_errors")
    return E


def to_markdown(entries: list[dict]) -> str:
    lines = ["# Concept-to-Implementation Matrix", "",
             "Generated from `src/co2ads/concepts.py` by `scripts/export_results.py`. All numbers are computed from the data snapshot in `data/raw/`.", ""]
    module = None
    for e in entries:
        if e["module"] != module:
            module = e["module"]
            lines += [f"## {module}", ""]
        lines += [f"### {e['concept']}", "",
                  f"- **Variables:** {e['variables']}",
                  f"- **Formula:** ${e['formula']}$",
                  "- **Computed output:**"]
        lines += [f"    - {k}: {v}" for k, v in e["output"].items()]
        lines += [f"- **Interpretation:** {e['interpretation']}",
                  f"- **Validation:** {e['validation']}",
                  f"- **Code:** `{e['code']}`", ""]
    return "\n".join(lines)
