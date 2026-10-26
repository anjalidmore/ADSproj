# Concept-to-Implementation Matrix

Generated from `src/co2ads/concepts.py` by `scripts/export_results.py`. All numbers are computed from the data snapshot in `data/raw/`.

## M1 Data Science Process

### CRISP-DM / data science process

- **Variables:** Whole project
- **Formula:** $\text{Prior knowledge} \rightarrow \text{Prepare} \rightarrow \text{Model} \rightarrow \text{Apply} \rightarrow \text{Knowledge}$
- **Computed output:**
    - Research questions: 9
    - Pipeline layers: raw → clean → analysis-ready
    - Syllabus concepts implemented: see this list
- **Interpretation:** The project follows the course framework: questions first, then data understanding and preparation, then modelling and evaluation, then a dashboard that reports the knowledge gained.
- **Validation:** Each stage has its own module and tests.
- **Code:** `src/co2ads/ (package layout)`

### Data exploration: volume & dimensions

- **Variables:** All 79 raw columns
- **Formula:** $\text{rows} \times \text{columns}$
- **Computed output:**
    - Rows: 50,411
    - Columns: 79
    - Years: 1750–2024
    - Entities: 254
    - Countries: 219
- **Interpretation:** 50,411 observations is the 'volume' motivation from Module 1; with 79 attributes, picking relevant variables (feature selection) matters.
- **Validation:** Shape asserted in tests/test_data.py::test_shape_and_keys; file hash checked against MANIFEST.json.
- **Code:** `src/co2ads/quality.py::report`

### Data cleaning: duplicates

- **Variables:** country, year
- **Formula:** $\#\{(country, year)\ \text{appearing more than once}\}$
- **Computed output:**
    - Duplicate rows: 0
    - Duplicate (country, year) keys: 0
- **Interpretation:** (country, year) is a valid unique key, so no rows need dropping for duplication.
- **Validation:** tests/test_data.py::test_shape_and_keys
- **Code:** `src/co2ads/quality.py::report`

### Data cleaning: inconsistent entities / aggregates

- **Variables:** country, iso_code
- **Formula:** $\text{entity\_type} = f(\text{country}, \text{iso\_code})$
- **Computed output:**
    - Entities without ISO code: 36
    - Of which genuine countries: 1 (Kosovo)
    - Rows by type: country: 42,654, aggregate: 6,500, international: 550, special: 432, world: 275
- **Interpretation:** Regions, income groups and duplicate '(GCP)' regions sit in the same table as countries. Summing them together would double-count, so every analysis filters entity_type = country.
- **Validation:** World total reconciles with Σ countries + international aviation & shipping to within 0.5% (tests/test_data.py::test_world_equals_countries_plus_international).
- **Code:** `src/co2ads/data.py::classify_entity`

### Data cleaning: missing values

- **Variables:** All columns
- **Formula:** $\text{missing\%} = \frac{\#\text{NaN}}{N}\times 100$
- **Computed output:**
    - GDP last year: 2022
    - Energy first year: 1965
    - Policy: no deletion, no imputation
- **Interpretation:** A blank means 'not reported', never zero emissions. Each analysis drops missing values only for the columns it uses (listwise per analysis), and reports its own n.
- **Validation:** tests/test_data.py::test_clean_keeps_every_row_and_value checks no value was imputed.
- **Code:** `src/co2ads/data.py::clean, cross_section`

### Data cleaning: invalid values & outliers

- **Variables:** co2, fuel columns, population
- **Formula:** $x < 0 \text{ for non-negative columns};\ \ |co2 - \Sigma\text{fuels}|$
- **Computed output:**
    - Negative values in non-negative columns: 0
    - Zero-CO₂ country rows: 640 (62 after 1950)
    - Max |total − Σ fuel parts|: 1.44 Mt
- **Interpretation:** No impossible values. Negative land-use or growth values are real (carbon sinks, falling emissions) and are kept. Early zeros are years before a country's record starts. Extreme values (e.g. Qatar) are genuine and are flagged, not deleted.
- **Validation:** Flags stored as columns (flag_negative_nonneg_col, flag_zero_co2, fuel_sum_gap).
- **Code:** `src/co2ads/data.py::clean`

### Transformation: normalisation (min–max) & log scale

- **Variables:** gdp_per_capita, co2_per_capita (2022)
- **Formula:** $x' = \frac{x - x_{min}}{x_{max} - x_{min}},\quad \log_{10} x$
- **Computed output:**
    - Used for: LOF distances; regression on log scale
    - Kurtosis β₂ (raw → log₁₀): 14.11 → 2.82
    - Pearson skewness (raw → log₁₀): 0.78 → -0.75
- **Interpretation:** Per-person CO₂ and GDP are strongly right-skewed with a heavy upper tail. Logging removes the heavy tail (β₂ close to the normal value 3, and the empirical rule holds), though a mild left skew remains from the very lowest emitters. Min–max scaling stops one unit from dominating a distance.
- **Validation:** Range of scaled column is exactly [0, 1].
- **Code:** `src/co2ads/regression.py::min_max, lof.py::lof_table`

### Train/test split & k-fold cross-validation

- **Variables:** log CO₂pc ~ log GDPpc + log energy pc (2022)
- **Formula:** $\text{Accuracy}_{cv} = \frac{1}{k}\sum_{i=1}^{k}\text{score}_i$
- **Computed output:**
    - Hold-out (2/3 train): train n=109, test n=55, test R² = 0.938
    - 5-fold CV mean R²: 0.946
    - 5-fold CV mean RMSE (log₁₀): 0.140
- **Interpretation:** Cross-validated R² is close to the in-sample R², so the regression is not overfitting. Every country is used for validation exactly once.
- **Validation:** tests/test_cross_checks.py::test_kfold_uses_every_record_once; fixed seed 42.
- **Code:** `src/co2ads/regression.py::holdout, kfold_cv`

### Bootstrapping

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $\text{resample } n \text{ values with replacement, } B = 5000 \text{ times}$
- **Computed output:**
    - Median: 3.210
    - 95% bootstrap CI: 2.538 – 3.733
    - Bootstrap SE: 0.341
- **Interpretation:** The median has no simple standard-error formula; bootstrapping gives an interval for it straight from the data.
- **Validation:** Same seed gives the same interval (tests/test_cross_checks.py::test_bootstrap_reproducible).
- **Code:** `src/co2ads/regression.py::bootstrap_ci`

### Correlation vs causation

- **Variables:** gdp_per_capita, co2_per_capita (2022)
- **Formula:** $r \neq \text{causation}$
- **Computed output:**
    - log–log r: 0.907
- **Interpretation:** Richer countries emit more per person, but GDP does not 'cause' CO₂ by itself: energy use and energy mix sit in between (see the multiple regression). All findings are worded as associations.
- **Validation:** Multiple regression shows GDP loses significance once energy is controlled.
- **Code:** `src/co2ads/results.py::relationships`

## M2 Descriptive Statistics

### Types of data / levels of measurement

- **Variables:** country, continent, year, co2, co2_per_capita, growth %
- **Formula:** $\text{Nominal} \subset \text{Ordinal} \subset \text{Interval} \subset \text{Ratio}$
- **Computed output:**
    - Nominal: country, iso_code, continent
    - Interval: year, growth %, temperature change
    - Ratio: co2, population, gdp, per-capita measures
- **Interpretation:** Ratio variables support every statistic (means, ratios, CV). Interval ones (year) do not have a true zero, so 'twice as many years' is meaningless.
- **Validation:** Mapping shown on the Dataset Explorer page.
- **Code:** `src/co2ads/quality.py::measurement_level`

### Classification & frequency distribution

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $\text{classes} \approx \sqrt{n};\ \text{exclusive intervals } [L, U)$
- **Computed output:**
    - Classes: 14
    - Class width: 3 t
    - Countries in first class: 104 (48.6%)
- **Interpretation:** Most countries emit little per person, with a long tail of high emitters: a positively skewed distribution.
- **Validation:** Σf equals n (tests/test_cross_checks.py::test_grouped_formulas_consistent).
- **Code:** `src/co2ads/descriptive.py::frequency_table`

### Tabulation: two-way table

- **Variables:** continent × year → CO₂ per person
- **Formula:** $\text{cell} = \frac{\Sigma\,CO_2}{\Sigma\,\text{population}}$
- **Computed output:**
    - Rows × columns: 6 continents × 5 years
- **Interpretation:** The two-way table is also the input of the two-way ANOVA (Module 3).
- **Validation:** Built from country rows only, so no aggregate is double-counted.
- **Code:** `src/co2ads/results.py::inference`

### Arithmetic mean (direct, short-cut, grouped step-deviation)

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $\bar X = \frac{\Sigma X}{N};\quad \bar X = A + \frac{\Sigma f d'}{N}\times i$
- **Computed output:**
    - Mean (individual): 4.632
    - Grouped mean (direct): 4.907
    - Grouped mean (step deviation): 4.907
    - Largest grouped-vs-raw gap: D1 +124.8%
- **Interpretation:** The grouped mean differs from the exact mean because it assumes values sit at class mid-points. The lower quantiles suffer most, because nearly half the countries fall in the first class. With raw data available, the individual-series value is the one to report.
- **Validation:** Short-cut = direct and step-deviation = direct (tests/test_cross_checks.py).
- **Code:** `src/co2ads/descriptive.py::mean_direct, grouped_mean`

### Weighted mean & combined mean

- **Variables:** co2_per_capita weighted by population (2024)
- **Formula:** $\bar X_w = \frac{\Sigma W X}{\Sigma W};\quad \bar X_{12} = \frac{N_1\bar X_1 + N_2 \bar X_2}{N_1+N_2}$
- **Computed output:**
    - Simple mean (typical country): 4.632
    - Population-weighted mean (typical person): 4.587
    - Countries + international aviation & shipping: 4.734
    - World per capita (World row): 4.729
    - Combined mean of 6 continent means: 4.632
- **Interpretation:** The simple mean treats Tuvalu and India equally; the weighted mean gives the average for a person. The weighted value is a little below the World figure because international aviation and shipping belong to no country.
- **Validation:** Combined mean of continent groups reproduces the overall mean exactly.
- **Code:** `src/co2ads/descriptive.py::weighted_mean, combined_mean`

### Median & mode

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $\text{Median} = \tfrac{N+1}{2}\text{th item};\ \ \text{Mode} = L + \frac{\Delta_1}{\Delta_1+\Delta_2}\times i$
- **Computed output:**
    - Median: 3.210
    - Grouped median: 3.161
    - Grouped mode: 2.053
    - Modal class: 0–3 t
- **Interpretation:** Mode < Median < Mean: the order the course gives for a positively skewed distribution. Continuous values never repeat exactly, so the mode comes from the grouped formula.
- **Validation:** Grouped median lies within one class width of the exact median.
- **Code:** `src/co2ads/descriptive.py::median, grouped_mode`

### Quartiles, deciles, percentiles

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $Q_k = \frac{k(N+1)}{4}\text{th item};\ D_k = \frac{k(N+1)}{10};\ P_k = \frac{k(N+1)}{100}$
- **Computed output:**
    - Q1: 0.985
    - Q3: 6.019
    - D1: 0.275
    - D9: 11.183
    - P90: 11.183
- **Interpretation:** A quarter of countries emit under 0.99 t per person; the top 10% emit over 11.18 t.
- **Validation:** Equals numpy's 'weibull' quantile method, which is the (N+1) rule (tests/test_cross_checks.py).
- **Code:** `src/co2ads/descriptive.py::quantile`

### Range, IQR, quartile deviation & coefficients

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $R = L - S;\ \ \frac{L-S}{L+S};\ \ IQR = Q_3-Q_1;\ \ QD = \frac{Q_3-Q_1}{2};\ \ \frac{Q_3-Q_1}{Q_3+Q_1}$
- **Computed output:**
    - Range: 41.217
    - Coefficient of range: 0.997
    - IQR: 5.034
    - Quartile deviation: 2.517
    - Coefficient of QD: 0.719
- **Interpretation:** The range is driven by a single country (Qatar); the IQR describes the middle half and is not affected by extremes.
- **Validation:** Hand formulas; quartiles cross-checked.
- **Code:** `src/co2ads/descriptive.py`

### Variance & standard deviation

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $\sigma = \sqrt{\frac{\Sigma (X-\bar X)^2}{N}};\ \ \sigma = \sqrt{\frac{\Sigma d^2}{N} - \left(\frac{\Sigma d}{N}\right)^2}$
- **Computed output:**
    - σ (÷N, formula sheet): 5.501
    - s (÷(n−1), sample): 5.513
    - Grouped σ (step deviation): 5.336
    - Coefficient of variation: 118.7%
- **Interpretation:** A CV above 100% means the spread is larger than the average itself: countries are extremely unequal emitters.
- **Validation:** Matches numpy.std (ddof 0 and 1); assumed-mean method equals actual-mean method.
- **Code:** `src/co2ads/descriptive.py::std, std_assumed_mean, grouped_std`

### Skewness: Karl Pearson & Bowley

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $Sk_P = \frac{\bar X - \text{Mode}}{\sigma} \text{ or } \frac{3(\bar X - \text{Med})}{\sigma};\quad Sk_B = \frac{Q_3+Q_1-2\,\text{Med}}{Q_3-Q_1}$
- **Computed output:**
    - Pearson (mode): 0.469
    - Pearson (3(mean−median)/σ): 0.776
    - Bowley: 0.116
- **Interpretation:** All three are positive (right skew). Bowley is much smaller because it only looks at the middle 50% and ignores the long tail of extreme emitters.
- **Validation:** Signs agree with Mean > Median.
- **Code:** `src/co2ads/descriptive.py`

### Kurtosis β₂

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $\beta_2 = \frac{\mu_4}{\mu_2^2}$
- **Computed output:**
    - β₂: 14.111
    - Shape: Leptokurtic (more peaked, heavier tails than normal)
- **Interpretation:** β₂ far above 3: a sharp peak of low emitters plus heavy tails. (The course formula sheet's shape descriptions are swapped; the standard meaning is used here.)
- **Validation:** Equals scipy.stats.kurtosis(fisher=False, bias=True).
- **Code:** `src/co2ads/descriptive.py::kurtosis_beta2`

### Box plot & IQR outliers

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $\text{fences} = Q_1 - 1.5\,IQR,\ Q_3 + 1.5\,IQR$
- **Computed output:**
    - Lower / upper limit: -6.57 / 13.57
    - High outliers: 13
    - Low outliers: 0
    - Examples: Qatar, Kuwait, Brunei, Bahrain, Trinidad and Tobago
- **Interpretation:** The flagged countries are genuine (mostly oil- and gas-exporting states, plus large energy-intensive economies such as Australia and the US), not data errors, so they are kept and discussed.
- **Validation:** Count reproduced on the Descriptive Statistics page for any year.
- **Code:** `src/co2ads/descriptive.py::box_plot, iqr_outliers`

## M2 Statistical Data

### Classification: geographical, chronological, qualitative, quantitative

- **Variables:** continent, year, change since 2005, largest fuel source, co2_per_capita (2024)
- **Formula:** $\text{group facts into classes sharing one characteristic}$
- **Computed output:**
    - Geographical: Asia 60.6% of world CO₂ from 50 countries
    - Chronological: world mean annual CO₂ by decade, 1950s → 2020–2024
    - Qualitative (two-fold): Rose or same: 152, Fell: 62
    - Qualitative (manifold): continent → fell/rose → largest fuel source
    - Quantitative: 5 magnitude classes of CO₂ per person
- **Interpretation:** Each type answers a different question: where (continent), when (decade), what kind (attribute) and how much (magnitude).
- **Validation:** Class counts sum to the number of countries; continent totals use countries only.
- **Code:** `src/co2ads/results.py::classification_and_tabulation`

### Tabulation: simple, two-way and higher-order tables

- **Variables:** continent × CO₂-per-person band × change since 2005
- **Formula:** $\text{stub} \times \text{caption} \rightarrow \text{body} + \text{totals}$
- **Computed output:**
    - Simple table: 5 classes, total 214
    - Two-way: 6 continents × 5 classes with margins
    - Higher-order: continent × fell/rose × 5 classes
- **Interpretation:** In the two-way table, 38 of Africa's 55 countries emit under 1 t per person; most European countries sit between 3 and 10 t.
- **Validation:** Row and column totals reproduce the one-way table (pandas crosstab with margins).
- **Code:** `src/co2ads/results.py::classification_and_tabulation`

## M2 Descriptive Statistics

### Discrete series: mean, median, mode, quartiles, σ

- **Variables:** years with falling CO₂, 2006–2024 per country (n = 214)
- **Formula:** $\bar X = \frac{\Sigma fX}{N};\ \text{Median} = X \text{ at } cf \ge \tfrac{N+1}{2};\ \sigma = \sqrt{\tfrac{\Sigma fd^2}{N} - \left(\tfrac{\Sigma fd}{N}\right)^2}$
- **Computed output:**
    - Mean (direct = short-cut): 7.168 = 7.168
    - Median / mode: 7 / 6
    - Q1 / Q3: 5 / 10
    - σ (actual = assumed mean): 3.370 = 3.370
- **Interpretation:** A typical country saw its emissions fall in about 7 of the 19 years; mean, median and mode are close, so this count is roughly symmetric, unlike per-person emissions.
- **Validation:** Reproduces the course examples (marks mean 41, median income ₹16,800, P20 = Q1 = D3 = 20, Q3 = 40).
- **Code:** `src/co2ads/descriptive.py::discrete_*`

## M2 Statistical Data

### Inclusive → exclusive classes; class limits from mid-values

- **Variables:** the discrete count grouped as 0–3, 4–7, …
- **Formula:** $L' = L - \tfrac{1}{2}\,\text{gap},\ U' = U + \tfrac{1}{2}\,\text{gap};\ \ L = m - \tfrac{i}{2},\ U = m + \tfrac{i}{2}$
- **Computed output:**
    - Classes: 0–3 → -0.5–3.5, 4–7 → 3.5–7.5, 8–11 → 7.5–11.5, 12–15 → 11.5–15.5, 16–19 → 15.5–19.5
    - Grouped median (exclusive classes): 6.652
- **Interpretation:** Whole-number classes leave gaps (3 to 4); converting to exclusive classes makes the series continuous so the interpolation formulas apply.
- **Validation:** Tested on simple inclusive classes and the slide's mid-value example (115 → 110–120).
- **Code:** `src/co2ads/descriptive.py::inclusive_to_exclusive, classes_from_midpoints`

## M2 Descriptive Statistics

### Cumulative frequency ('less than' / 'more than') & ogive

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $\text{Median} = \text{x where the two ogives cross} = L + \frac{N/2 - cf}{f}\times i$
- **Computed output:**
    - Ogive intersection (median): 3.1607
    - Formula median: 3.1607
    - Q1 / Q3 read from ogive: 1.543 / 6.058
- **Interpretation:** The graphical and formula medians agree exactly, because the ogive interpolates linearly inside the median class just as the formula does.
- **Validation:** Tested on the course 'median from cumulative data' example. The slide uses i = 10 for classes 5 wide and prints 14.29; the correct median is 12.15.
- **Code:** `src/co2ads/descriptive.py::cumulative_frequencies, ogive_value, ogive_intersection`

### Mean with open-end classes

- **Variables:** CO₂ per person (t), 2024, n = 214 countries; last class '15 t and above'
- **Formula:** $\text{assume open class width} = \text{neighbour's width, then } \bar X = \frac{\Sigma fm}{N}$
- **Computed output:**
    - Mean (open class assumed): 4.598
    - Grouped mean (all closed): 4.907
    - Exact mean: 4.632
    - Median (unchanged): 3.161
- **Interpretation:** The open-class mean depends on an assumption about the missing limit; the median does not. With an open top class in skewed data, report the median, as the course advises.
- **Validation:** Tested on the course example ('Below 10' … 'Above 50').
- **Code:** `src/co2ads/descriptive.py::open_end_mean, close_top_classes`

### Correcting incorrect items in a mean

- **Variables:** co2_per_capita 2024: 231 rows incl. 17 aggregate rows
- **Formula:** $\Sigma X_{correct} = N\bar X - \Sigma X_{wrong} + \Sigma X_{right}$
- **Computed output:**
    - Incorrect mean (aggregates included): 4.6981
    - Corrected mean: 4.6325 (N = 214)
- **Interpretation:** If regional totals are averaged in with countries by mistake, the course correction formula recovers the countries-only mean exactly: a concrete reason the cleaning step matters.
- **Validation:** Reproduces Illustrations 6 and 7 (39.7 and 40.18).
- **Code:** `src/co2ads/descriptive.py::corrected_mean`

### Moments (about origin and mean), β₁, β₂ & empirical mode

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $\mu'_r = \frac{\Sigma X^r}{N};\ \mu_r = \frac{\Sigma (X-\bar X)^r}{N};\ \beta_1 = \frac{\mu_3^2}{\mu_2^3};\ \beta_2 = \frac{\mu_4}{\mu_2^2};\ \text{Mode} \approx 3\,\text{Med} - 2\,\bar X$
- **Computed output:**
    - μ₂ / μ₃ / μ₄: 30.26 / 462.4 / 12,918
    - β₁ / γ₁: 7.719 / 2.778
    - β₂ / γ₂: 14.111 / 11.111
    - Empirical vs grouped mode: 0.365 vs 2.053
- **Interpretation:** μ₃ > 0 confirms right skew and μ₄ the heavy tail. The empirical mode relation only holds for moderate skew; here it is far from the grouped mode, another sign of extreme skewness.
- **Validation:** Central moments computed directly and from raw moments agree; γ₁ and β₂ match SciPy.
- **Code:** `src/co2ads/descriptive.py::moments, empirical_mode`

## M2 Multivariate Exploration

### Central data point

- **Variables:** gdp_per_capita, co2_per_capita (2022, n = 164)
- **Formula:** $\bar{\mathbf{x}} = (\bar X_1, \bar X_2, \dots)$
- **Computed output:**
    - Centre: GDP pc = 19,149, CO₂ pc = 4.53
    - Farthest from centre: Qatar, United Arab Emirates, Kuwait
- **Interpretation:** Distance from the multivariate centre (in SD units) is a first, simple outlier screen.
- **Validation:** Means computed directly.
- **Code:** `src/co2ads/correlation.py::central_data_point`

### Covariance & Karl Pearson r (3 methods)

- **Variables:** gdp_per_capita vs co2_per_capita (2022, n = 164)
- **Formula:** $r = \frac{\Sigma xy}{\sqrt{\Sigma x^2\,\Sigma y^2}} = \frac{N\Sigma XY - \Sigma X\Sigma Y}{\sqrt{N\Sigma X^2-(\Sigma X)^2}\sqrt{N\Sigma Y^2-(\Sigma Y)^2}}$
- **Computed output:**
    - r (raw): 0.745 — Moderate degree positive
    - r (log–log): 0.907 — High degree positive
    - Probable error (log): 0.0093
    - r²(log): 0.823
- **Interpretation:** A strong positive relationship that is clearer on the log scale, because both variables are skewed. r² = share of variation in one explained by the other.
- **Validation:** Actual-mean, direct and assumed-mean methods agree to 1e-12; equal to numpy.corrcoef.
- **Code:** `src/co2ads/correlation.py::analyse`

### Types of correlation: positive, negative, none, non-linear

- **Variables:** cross-section 2022 (n = 164), UK 1990–2024, population vs CO₂ pc 2024
- **Formula:** $\text{sign and shape of the scatter}$
- **Computed output:**
    - Positive (log GDP vs log CO₂ pc): 0.907
    - Negative (log GDP vs log CO₂ per kWh): -0.461
    - Negative over time (UK CO₂ vs year): -0.931
    - None (population vs CO₂ pc): -0.002
    - Non-linear (raw GDP vs CO₂ pc): 0.745
- **Interpretation:** Richer countries emit more per person but have cleaner energy per kWh. Population says nothing about per-person emissions. Raw GDP vs CO₂ is curved, so r understates the link until both are logged.
- **Validation:** Each r computed with the course method; signs match the scatter diagrams.
- **Code:** `src/co2ads/results.py::correlation_forms`

### Simple, partial & multiple correlation

- **Variables:** log₁₀ CO₂ pc (y), log₁₀ GDP pc (x₁), log₁₀ energy pc (x₂), 2022
- **Formula:** $r_{y1\cdot2} = \frac{r_{y1} - r_{y2}r_{12}}{\sqrt{(1-r_{y2}^2)(1-r_{12}^2)}};\ R^2_{y\cdot12} = \frac{r_{y1}^2 + r_{y2}^2 - 2r_{y1}r_{y2}r_{12}}{1-r_{12}^2}$
- **Computed output:**
    - Simple r(y, GDP) / r(y, energy): 0.907 / 0.974
    - Partial r(y, GDP | energy): -0.151
    - Partial r(y, energy | GDP): 0.850
    - Multiple R: 0.975
- **Interpretation:** Holding energy constant removes almost all of GDP's link with emissions, the correlation-side view of the regression's multicollinearity result. Multiple R² equals the regression R².
- **Validation:** Partial r equals the correlation of residuals (statsmodels); multiple R equals √R² of OLS.
- **Code:** `src/co2ads/correlation.py::partial_correlation, multiple_correlation`

## M2 Data Visualization

### Diagrams & graphs: bars (simple, multiple, sub-divided, percentage, deviation, broken), rectangle, square, circle, pie, pictogram, cartogram, line (incl. false baseline), histogram, frequency polygon, ogive, box plot, distribution chart, scatter, scatter matrix, bubble

- **Variables:** various
- **Formula:** $\text{diagram chosen by question; area} \propto \text{value} \Rightarrow \text{side} = \sqrt{v},\ r = \sqrt{v/\pi}$
- **Computed output:**
    - Charts in dashboard: all of these, with units and tooltips
    - Rectangle diagram: width = population, height = CO₂ per person, area = total CO₂
- **Interpretation:** Line for trends, bars for comparisons, two-dimensional shapes when area should carry the value, histogram/box/ogive for distributions, scatter/bubble/scatter matrix for relationships.
- **Validation:** Every chart is drawn from computed data, never hard-coded.
- **Code:** `app/pages/3_Classification_and_Diagrams.py and others`

## M3 Inferential Statistics

### Normal distribution, z-scores & empirical rule

- **Variables:** CO₂ per person (t), 2024, n = 214 countries and its log₁₀
- **Formula:** $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}};\ z = \frac{x-\mu}{\sigma}$
- **Computed output:**
    - Within ±1σ raw / log / normal: 88.8% / 70.1% / 68.3%
    - Within ±2σ raw / log / normal: 95.3% / 93.9% / 95.4%
- **Interpretation:** Raw per-person emissions are far from normal; their logarithm is close to normal (log-normal distribution). This is why tests and regression use the log scale.
- **Validation:** Normal percentages from scipy.stats.norm.
- **Code:** `src/co2ads/inference.py::empirical_rule, normal_pdf`

### Poisson distribution (suitability check)

- **Variables:** Yearly count of countries (≥1 Mt) whose CO₂ fell by >10%, 1991–2024
- **Formula:** $P(X=x)=\frac{e^{-\mu}\mu^x}{x!};\ \ \text{Poisson} \Rightarrow \text{mean} = \text{variance}$
- **Computed output:**
    - Mean: 11.97
    - Variance: 65.42
    - Dispersion index (var/mean): 5.47
    - p (dispersion test): 6.4e-22
- **Interpretation:** Variance is about five times the mean, so these counts are NOT Poisson: drops cluster in crisis years (2020: 42 countries, 2009: 27, 1991–94 post-Soviet collapse: 17–23 a year) instead of happening independently at a constant rate.
- **Validation:** Index-of-dispersion χ² test, n − 1 d.f.
- **Code:** `src/co2ads/inference.py::poisson_check`

### Central limit theorem

- **Variables:** CO₂ per person (t), 2024, n = 214 countries; 10,000 samples of n = 30
- **Formula:** $\bar X \sim N\!\left(\mu, \frac{\sigma}{\sqrt n}\right) \text{ for large } n$
- **Computed output:**
    - Population mean / mean of sample means: 4.632 / 4.640
    - σ/√n vs SD of sample means: 1.004 / 1.018
    - Skewness: population → sample means: 0.78 → 0.30
- **Interpretation:** Even from this strongly skewed population, sample means are nearly normal and their spread matches σ/√n: the CLT on real data.
- **Validation:** Seeded simulation (seed 42).
- **Code:** `src/co2ads/inference.py::clt_simulation`

### Confidence interval for a mean

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $\bar x \pm z\frac{s}{\sqrt n}$
- **Computed output:**
    - 95% CI: 3.894 – 5.371 t
    - Margin: 0.739
    - Method: z (n ≥ 30)
- **Interpretation:** Read as: if 2024 countries were a sample from the process that generates national emissions, the mean per-person value lies in this range with 95% confidence. The 214 countries are a near-complete set, so this is a model-based interpretation.
- **Validation:** Reproduces the course example (45 ± 2.77) in tests.
- **Code:** `src/co2ads/inference.py::ci_mean`

### Z-test (single mean, large sample)

- **Variables:** CO₂ per person (t), 2024, n = 214 countries
- **Formula:** $Z = \frac{\bar x - \mu_0}{s/\sqrt n}$
- **Computed output:**
    - H₀: μ = world per-person value (4.729 t)
    - Z: -0.256
    - Critical |Z| (5%): 1.96
    - p: 0.798
    - Decision: Fail to reject H0
- **Interpretation:** The average country is not significantly different from the average person at 5%. Large spread between countries makes the test weak.
- **Validation:** Reproduces the oceanographer example (Z = 2.067) in tests.
- **Code:** `src/co2ads/inference.py::z_test_mean`

### Paired t-test

- **Variables:** co2_per_capita 2005 vs 2024, same countries (n = 214)
- **Formula:** $t = \frac{\bar d}{s_d/\sqrt n},\ \ df = n-1$
- **Computed output:**
    - Mean before → after: 5.279 → 4.632 t
    - Mean difference: -0.647
    - t: -2.864
    - p: 0.0046
    - Countries down / up: 87 / 127
    - Decision: Reject H0
- **Interpretation:** Average per-person emissions fell significantly since 2005, yet more countries rose than fell: large falls in high emitters (Europe, North America) outweigh many small rises.
- **Validation:** Reproduces the soldiers example (|t| = 2.16); p matches scipy.ttest_rel.
- **Code:** `src/co2ads/inference.py::t_test_paired`

### Independent two-sample t-test

- **Variables:** log₁₀ CO₂ per person, Europe (n=44) vs Asia (n=50), 2024
- **Formula:** $t = \frac{\bar x - \bar y}{\sqrt{s^2/n_1 + s^2/n_2}}$
- **Computed output:**
    - t: 1.752
    - df: 92
    - p (pooled): 0.083
    - p (Welch, sensitivity): 0.070
    - Variance ratio: 9.3
    - Decision: Fail to reject H0
- **Interpretation:** Typical (geometric-mean) values: Europe 5.03 t vs Asia 3.54 t. The difference is not significant at 5%, and the large variance ratio means the pooled-variance assumption is questionable, so Welch's version is also reported.
- **Validation:** Reproduces the diet example; p matches scipy.ttest_ind.
- **Code:** `src/co2ads/inference.py::t_test_independent`

### Type I & Type II errors

- **Variables:** CO₂ per person (t), 2024, n = 214 countries; 10,000 simulated samples of n = 40
- **Formula:** $\alpha = P(\text{reject } H_0 | H_0 \text{ true}),\ \beta = P(\text{accept } H_0 | H_0 \text{ false})$
- **Computed output:**
    - Nominal α: 0.05
    - Observed Type I rate: 0.090
    - Type II rate β (true mean shifted by 1.5 t): 0.511
    - Power: 0.489
- **Interpretation:** With skewed data and n = 40 the z-test rejects a true H₀ more often than 5%: the normal approximation is not yet accurate. β shows how often a real 1.5 t difference would be missed.
- **Validation:** Type I rate converges toward α as n grows (try it on the Statistical Analysis page).
- **Code:** `src/co2ads/inference.py::error_rate_simulation`

### One-way ANOVA

- **Variables:** log₁₀ CO₂ per person by continent (2024, k = 6, n = 214)
- **Formula:** $F = \frac{SS_B/(k-1)}{SS_W/(n-k)}$
- **Computed output:**
    - F: 30.87
    - F critical (5%): 2.26
    - p: 2.0e-23
    - η² (SS_B/SS_T): 0.426
    - Decision: Reject H0
    - Grand mean vs mean of means: 0.357 vs 0.404
- **Interpretation:** Continent explains a large share (η²) of the variation in log emissions per person. Group sizes differ, so the grand mean is weighted by n (the slides' 'mean of means' only works for equal groups).
- **Validation:** Reproduces the wheat example (F = 1.5); matches scipy.f_oneway; SS_T = SS_B + SS_W.
- **Code:** `src/co2ads/inference.py::one_way_anova`

### Two-way ANOVA (without replication)

- **Variables:** continental CO₂ per person: continent (rows) × year (columns)
- **Formula:** $SS_C = \sum\frac{T_j^2}{n_j} - \frac{T^2}{n},\ SS_R = \sum\frac{T_i^2}{n_i} - \frac{T^2}{n}$
- **Computed output:**
    - F (continent): 60.13 (crit 2.71) → Reject H0
    - F (year): 1.36 (crit 2.87) → Fail to reject H0
- **Interpretation:** Differences between continents are significant; differences between years are not, because continents moved in opposite directions (Asia up, Europe and North America down), which ends up in the residual.
- **Validation:** Matches statsmodels anova_lm (tests).
- **Code:** `src/co2ads/inference.py::two_way_anova`

## M4 Regression

### Simple linear regression: slope & intercept

- **Variables:** y = log₁₀ CO₂ pc, x = log₁₀ GDP pc (2022, n = 164)
- **Formula:** $a = \frac{n\Sigma xy - \Sigma x\Sigma y}{n\Sigma x^2 - (\Sigma x)^2},\ \ b = \frac{\Sigma y - a\Sigma x}{n}$
- **Computed output:**
    - Slope: 1.0447
    - Intercept: -3.8696
    - t (slope): 27.40
    - R²: 0.823
- **Interpretation:** On log scales the slope is an elasticity: a 1% higher GDP per person goes with about 1.04% higher CO₂ per person, on average across countries (association, not causation).
- **Validation:** Reproduces the expenditure example (a = 8.4, b = 3.2); matches statsmodels OLS.
- **Code:** `src/co2ads/regression.py::simple_regression`

### Error measures: MAE, MSE, RMSE, MAPE, R²

- **Variables:** same model
- **Formula:** $MAE=\tfrac1n\Sigma|y-\hat y|,\ RMSE=\sqrt{\tfrac1n\Sigma(y-\hat y)^2},\ MAPE=\tfrac{100}{n}\Sigma\left|\tfrac{y-\hat y}{y}\right|,\ R^2 = 1-\tfrac{SS_{res}}{SS_{tot}}$
- **Computed output:**
    - MAE (log₁₀): 0.200
    - RMSE (log₁₀): 0.256
    - MAPE on original tonnes: 49.0%
- **Interpretation:** MAPE is computed after converting predictions back to tonnes, because on the log scale values near 0 make MAPE meaningless (a limitation the slides mention).
- **Validation:** Hand formulas; R² matches statsmodels.
- **Code:** `src/co2ads/regression.py::all_metrics`

### Multiple linear regression & partial regression coefficients

- **Variables:** y = log₁₀ CO₂ pc; X₁ = log₁₀ GDP pc; X₂ = log₁₀ energy pc (2022, n = 164)
- **Formula:** $b_1 = \frac{\Sigma x_2^2\Sigma x_1y - \Sigma x_1x_2\Sigma x_2y}{\Sigma x_1^2\Sigma x_2^2 - (\Sigma x_1x_2)^2},\ b_0 = \bar Y - b_1\bar X_1 - b_2\bar X_2$
- **Computed output:**
    - Equation: Ŷ = -3.169 -0.117·X₁ +0.986·X₂
    - R² / adjusted R²: 0.951 / 0.950
    - r₁₂ (X₁ vs X₂): 0.942
- **Interpretation:** Holding energy use constant, GDP adds almost nothing; holding GDP constant, 1% more energy per person goes with about 1% more CO₂ per person. GDP and energy are highly correlated (multicollinearity), so the GDP coefficient is unstable.
- **Validation:** Reproduces the course example (b1 = 3.1479, b2 = −1.6561, R² = 0.9626); matches statsmodels.
- **Code:** `src/co2ads/regression.py::multiple_regression_2`

### Overall significance (F-test) & individual coefficients (t-test)

- **Variables:** same model
- **Formula:** $F = \frac{(n-k-1)R^2}{k(1-R^2)};\quad t = \frac{b_i}{SE(b_i)},\ SE(b_i)=\sqrt{\frac{RSS/(n-k-1)}{\Sigma(x_i-\bar x_i)^2(1-r_{12}^2)}}$
- **Computed output:**
    - F: 1,554.2 (crit 3.05) → Reject H0 (model significant)
    - t(b₁ GDP): -1.94, p = 0.054 → not significant
    - t(b₂ energy): 20.47, p = 1.1e-46 → significant
- **Interpretation:** The model as a whole is highly significant (F-test first, as the slides require); individually only energy per person is significant at 5%.
- **Validation:** SE and t match statsmodels (tests/test_cross_checks.py). Course example: F = 64.37 vs F₀.₀₅(2, 5) = 5.79; t(b₁) = 3.753. Slide 40 copies b₁'s working for b₂; the correct values are SE = 0.976, t = −1.697 (still not significant).
- **Code:** `src/co2ads/regression.py::multiple_regression_2`

### Goodness of fit & overfit: R, R², adjusted R² (simple vs multiple)

- **Variables:** simple (X₁ = GDP) vs multiple (X₁ = GDP, X₂ = energy)
- **Formula:** $R^2 = \frac{b_1\Sigma x_1y + b_2\Sigma x_2y}{\Sigma y^2},\ \ \bar R^2 = 1-(1-R^2)\frac{n-1}{n-k-1},\ \ R=\sqrt{R^2}$
- **Computed output:**
    - R² simple → multiple: 0.823 → 0.951 (Δ +0.128)
    - Adjusted R²: 0.821 → 0.950
    - F simple / multiple: 751.0 / 1,554.2
    - RMSE (log₁₀): 0.256 → 0.135
- **Interpretation:** Adjusted R² rises as well as R², so energy adds real explanatory power beyond the extra parameter. In the simple model F = t² of the slope.
- **Validation:** Simple-model R², adjusted R², F and SSE match statsmodels; F = t² checked.
- **Code:** `src/co2ads/regression.py::compare_models`

### Overlap between predictors (multicollinearity flag)

- **Variables:** X₁ = log₁₀ GDP pc, X₂ = log₁₀ energy pc
- **Formula:** $\text{tolerance}=1-r_{12}^2,\ \ VIF=\frac{1}{1-r_{12}^2},\ \ SE(b_i)\propto\sqrt{VIF}$
- **Computed output:**
    - r₁₂: 0.942
    - Tolerance / VIF: 0.112 / 8.93
    - Flag: YES (high)
    - b₁ (GDP) alone → with energy: 1.045 → -0.117 (sign flips)
- **Interpretation:** Only 11% of the variation in log GDP is not already carried by log energy, so the (1 − r₁₂²) term in SE(bᵢ) multiplies the standard errors by √VIF = 3.0. That is why GDP is significant alone but not alongside energy, and its sign flips. Fit and predictions are fine; only the split of credit between the two predictors is unreliable.
- **Validation:** VIF matches statsmodels variance_inflation_factor; course example r₁₂ = −0.88 → VIF = 4.56.
- **Code:** `src/co2ads/regression.py::collinearity`

### Estimate for unknown values (prediction)

- **Variables:** median country: GDP pc = 12,606 int-$, energy pc = 14,490 kWh
- **Formula:** $\hat Y = b_0 + b_1X_1 + b_2X_2,\ \ \widehat{CO_2} = 10^{\hat Y}$
- **Computed output:**
    - Simple model: 2.59 t per person
    - Multiple model: 2.87 t per person
- **Interpretation:** Predictions are made on the log₁₀ scale and converted back to tonnes.
- **Validation:** Reproduces slide 32 (X₁ = 74, X₂ = 14 → Ŷ = 202.893).
- **Code:** `src/co2ads/regression.py::predict`

## M5 Anomaly Detection

### Local Outlier Factor (Manhattan distance)

- **Variables:** min–max scaled log₁₀ GDP pc & log₁₀ CO₂ pc (2022, n = 164), k = 10
- **Formula:** $\text{reach}(p,o)=\max(k\text{-dist}(o), d(p,o));\ lrd(p)=\frac{|N_k(p)|}{\Sigma\,\text{reach}};\ LOF(p)=\frac{\overline{lrd(N_k(p))}}{lrd(p)}$
- **Computed output:**
    - Top 5 by LOF: Qatar (2.37), North Korea (1.86), Norway (1.67), Switzerland (1.59), Venezuela (1.59)
    - LOF > 1.5: 6
    - Also IQR outliers (of top 10): Qatar, Trinidad and Tobago
- **Interpretation:** LOF finds countries that are unusual for their neighbourhood: very high emitters for their income (Qatar), and also low emitters for their income (Switzerland, Sweden, North Korea's low GDP with relatively high CO₂). The IQR rule only sees one variable and only high values.
- **Validation:** Reproduces the 4-point slide example (LOF(d) = 2.0). On these 164 countries it matches sklearn LocalOutlierFactor(metric='manhattan') to 1e-09 (0 tied neighbourhoods, same top 10: True).
- **Code:** `src/co2ads/lof.py::lof, sklearn_check`

### LOF vs the IQR (box-plot) rule

- **Variables:** LOF > 1.5 on both variables vs IQR fences on CO₂ per person alone (2022, n = 164)
- **Formula:** $\text{IQR outlier: } x<Q_1-1.5\,IQR \ \text{or}\ x>Q_3+1.5\,IQR$
- **Computed output:**
    - both: 1 (Qatar)
    - LOF only: 5 (Burundi, North Korea, Norway, Switzerland, Venezuela)
    - IQR only: 8 (Australia, Bahrain, Kuwait, Oman, Saudi Arabia, Trinidad and Tobago, United Arab Emirates, United States)
    - neither: 150
    - Flagged at every k (5, 10, 15, 20): North Korea, Qatar
- **Interpretation:** The methods agree only on Qatar. The IQR-only countries (Gulf states, Australia, US) are high emitters but sit in a dense cluster of rich, high-CO₂ countries, so their local density is normal (LOF close to 1). LOF-only countries are unusual for their income, including low emitters the one-sided IQR rule cannot see.
- **Validation:** IQR fences use the course quartile rule k(N+1)/4; the LOF threshold 1.5 is a convention, so k = 5–20 is reported as a sensitivity check.
- **Code:** `src/co2ads/results.py::outliers_lof`

## M6 Time Series

### Components & classical decomposition

- **Variables:** World CO₂ 1950–2024 (annual)
- **Formula:** $Y = T + S + R \ \text{or}\ Y = T\times S\times R;\ T = \text{centred MA}$
- **Computed output:**
    - Seasonal component: not applicable (annual data, no within-year seasons)
    - Trend: centred 5-year moving average
    - Validated on: course quarterly-sales example (indices −10.44, −11.69, 6.06, 16.06)
- **Interpretation:** Seasonality needs sub-annual data. On annual CO₂ only the trend and irregular parts exist; the years in which world emissions fell match known shocks: 1974–75 and 1980–82 (oil crises), 1992 (post-Soviet collapse), 2009 (financial crisis) and 2020 (COVID, −5.2%).
- **Validation:** Additive and multiplicative decomposition reproduce the slide example exactly (tests).
- **Code:** `src/co2ads/timeseries.py::classical_decomposition, centred_moving_average`

### Average, moving-average & naïve forecasts; least-squares trend

- **Variables:** World CO₂; train ≤ 2019, test 2020–2024
- **Formula:** $F_t=\tfrac{1}{t-1}\Sigma_{i<t} y_i;\ \ F_t=\tfrac1k\Sigma_{i=t-k}^{t-1}y_i;\ \ \hat y = a\,t + b$
- **Computed output:**
    - Average method: MAE 16,286 Mt, MAPE 43.69%, MASE 31.52
    - Moving average (k=3): MAE 1,163 Mt, MAPE 3.13%, MASE 2.25
    - Naïve (last value): MAE 1,074 Mt, MAPE 2.93%, MASE 2.08
    - Linear trend (2000–2019): MAE 2,805 Mt, MAPE 7.57%, MASE 5.43
- **Interpretation:** Best on the hold-out by MASE: Naïve (last value). The 2000–2019 linear trend overshoots because emissions growth slowed after 2013 and dropped in 2020. The average method is useless for a trending series, exactly as the slides warn.
- **Validation:** Reproduces slide metrics (MAE 2.32, MASE 1.16 / 2.67, 1.14).
- **Code:** `src/co2ads/timeseries.py::holdout_comparison`

### Forecast evaluation: MAE, RMSE, MAPE, MASE

- **Variables:** same
- **Formula:** $MASE = \frac{MAE}{\frac{1}{n-1}\sum_{j=2}^{n}|y_j-y_{j-1}|}$
- **Computed output:**
    - MASE < 1 means: better than the naïve forecast
    - Scaling series: training period 1950–2019
- **Interpretation:** MASE above 1 for every method shows that, over 2020–2024, simply carrying forward last year's value is hard to beat for annual emissions.
- **Validation:** Hand formulas; slide examples reproduced.
- **Code:** `src/co2ads/timeseries.py::evaluate, mase`

### True hold-out (fixed origin) and the 2020 COVID shock

- **Variables:** World CO₂; every forecast made from data up to 2019 only
- **Formula:** $\hat y_{2019+h} = \bar y_{1950..2019}\ \text{(average)},\ \tfrac13(y_{2017}+y_{2018}+y_{2019})\ \text{(MA)},\ y_{2019}\ \text{(naïve)}$
- **Computed output:**
    - Average method: MAE 16,722 Mt, MAPE 44.83%, MASE 32.36
    - Moving average (k=3): MAE 1,227 Mt, MAPE 3.28%, MASE 2.37
    - Naïve (last value): MAE 1,022 Mt, MAPE 2.76%, MASE 1.98
    - Linear trend (2000–2019): MAE 2,805 Mt, MAPE 7.57%, MASE 5.43
    - 2020 drop: -1,928 Mt (-5.2%); back above 2019 in 2022
- **Interpretation:** One-step-ahead forecasts (course definition) see each hold-out year once it has happened; the fixed-origin test does not. Best by MASE: Naïve (last value). No method anticipates the 2020 fall (2020 is one step ahead in both tests, so each method's 2020 error is the same in both), and the trend line, which assumes growth continues, is furthest off. Emissions rebounded to the 2019 level, so a flat forecast from 2019 did well afterwards.
- **Validation:** Forecasts are unchanged when hold-out values are altered (test); trend errors identical in both tables.
- **Code:** `src/co2ads/timeseries.py::fixed_origin_forecasts, shock_errors`
