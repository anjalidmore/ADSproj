# Global CO₂ Emissions Analytics

An Applied Data Science course project that applies every concept in the ADS syllabus (Modules 1–6) to the Our World in Data CO₂ dataset: 50,411 country-year rows × 79 columns, 1750–2024. Results are explored in an interactive Streamlit dashboard.

Every number in the dashboard, report, slides and viva guide is computed from the raw file by code in `src/co2ads/`. Nothing is typed in by hand.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 automatically).

```bash
uv sync                                  # create .venv with pinned dependencies (uv.lock)
uv run pytest                            # 56 tests (course examples, library cross-checks, data checks, every app page)
uv run python scripts/export_results.py  # docs/results.json, docs/figures/*.png, docs/CONCEPT_MATRIX.md
uv run streamlit run app/Home.py         # dashboard at http://localhost:8501
```

Without uv: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -e . pytest`, then the same commands without `uv run`.

## Project layout

```
data/raw/                 original files, never modified (hashes in MANIFEST.json)
data/processed/           generated clean parquet (git-ignored, rebuilt automatically)
src/co2ads/
  config.py               paths, analysis years (2024 / GDP 2022 / base 2005), seed
  data.py                 M1  load, hash-check, classify countries vs aggregates, continent lookup, flags
  quality.py              M1  data-quality report, world-total reconciliation, coverage
  descriptive.py          M2  mean/median/mode, (N+1) quantiles, dispersion, skewness, kurtosis, grouped data
  correlation.py          M2  central data point, covariance, Pearson r (3 methods), probable error
  inference.py            M3  normal/Poisson, CLT, CI, z & t tests, one- and two-way ANOVA, Type I/II simulation
  regression.py           M4  simple & 2-predictor regression, error metrics, F & t tests; M1 hold-out, k-fold, bootstrap
  lof.py                  M5  Local Outlier Factor with Manhattan distance
  timeseries.py           M6  classical decomposition, average / moving-average / naïve / trend forecasts, MASE
  results.py              every analysis applied to the real data (single source of numbers)
  concepts.py             concept-to-implementation registry (Concepts Lab + CONCEPT_MATRIX.md)
app/                      Streamlit dashboard: Home + 8 pages
scripts/export_results.py writes results.json, figures and the concept matrix
tests/                    pytest suite
docs/                     results.json, figures, concept matrix, report, slides, viva guide
```

## Dashboard pages

| Page | Contents |
|---|---|
| Home | Headline metrics, global trend, top emitters, answers to the 9 research questions |
| Dataset Explorer | Filterable raw/clean data with CSV export, column dictionary (units, level of measurement, missing %), quality checks, coverage |
| Descriptive Statistics | All formula-sheet measures for any variable/year/continent, frequency table, histogram + polygon, ogives, box plot & outliers, distribution chart, moments, discrete series, open-end classes, correcting wrong items |
| Classification & Diagrams | Four types of classification; simple, two-way and higher-order tables; multiple, sub-divided, percentage and broken bars; rectangle, square, circle and pictogram diagrams; false baseline; types of correlation (incl. partial and multiple) |
| Country & Time | Rankings (total vs per person), world map, country comparisons, change between years, fuel mix |
| Statistical Analysis | Correlation & scatter matrix, CLT simulator, confidence intervals, paired / independent t, z-test, one- and two-way ANOVA, Type I/II errors, Poisson check |
| Advanced Analytics | Simple & multiple regression with F/t tests and k-fold CV, LOF, time-series hold-out forecasting, decomposition demo |
| ADS Concepts Lab | 53 syllabus concepts: formula, variables, computed output, interpretation, validation, code location |
| Methodology | Findings, provenance, cleaning decisions, assumptions, limitations, reproduction |

## Key methodological decisions

- **Countries only.** 36 entities have no ISO code: regions, income groups, "(GCP)" duplicates, international aviation and shipping. All are aggregates except Kosovo. Analyses use `entity_type == "country"` (219 countries) so nothing is counted twice. The `World` row equals Σ countries + international transport, which is verified in the tests.
- **No imputation.** Missing means "not reported", never zero. Each analysis drops missing values only for its own columns and reports its n.
- **Outliers are kept and flagged.** Extreme per-person emitters (Qatar, Kuwait, …) are real.
- **Course conventions:** quartiles use the k(N+1)/4 rule; σ divides by N on the formula sheet and s by n−1 in t-tests; ANOVA uses the n-weighted grand mean because group sizes are unequal.
- **Logs for inference and regression.** Per-person CO₂, GDP and energy are right-skewed; their log₁₀ is close to normal.
- **Annual data has no seasonality**, so classical decomposition is verified on the course's quarterly example and only its trend step is applied to CO₂.

## Documents (`docs/`)

| File | What |
|---|---|
| `REPORT.docx` / `REPORT.md` | Academic report (~5,600 words, 10 figures), generated from `results.json` by `scripts/build_report.py` |
| `viva_guide.html` | Viva study guide: architecture, file map, methods by module, question bank, numbers cheat-sheet |
| `CONCEPT_MATRIX.md` | Every syllabus concept → formula, computed output, interpretation, validation, code location |
| `results.json` | Every computed number used anywhere |
| `figures/` | Static charts used in the report and slides |

Rebuilding the Word report needs pandoc: `brew install pandoc` (or set `PYPANDOC_PANDOC` to a pandoc binary), then `uv run --with pypandoc python scripts/build_report.py`. Without pandoc the script still writes `REPORT.md`.

## Data & licence

Our World in Data, *CO₂ and Greenhouse Gas Emissions* (https://catalog.ourworldindata.org/emissions/owid_co2/), downloaded 2026-10-21, licensed CC BY 4.0. The underlying data come from the Global Carbon Project and the sources listed in `data/raw/owid_co2.codebook.csv`. Continent lookup: OWID "Continents according to Our World in Data". Re-downloading will give a newer revision with different numbers; the hash check in the Dataset Explorer shows which snapshot is loaded.
