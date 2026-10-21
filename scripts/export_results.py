"""Run every analysis and write docs/results.json + docs/figures/*.png.

The report, slides and viva guide quote numbers ONLY from results.json, so re-running this
script after a data refresh keeps every deliverable consistent with the code.

    uv run python scripts/export_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from co2ads import concepts, config, data, results as R  # noqa: E402

FIG = config.DOCS_DIR / "figures"
DROP = {"data", "fitted", "residuals", "means", "forecasts", "paired_data", "anova_data", "forecast", "distribution"}

GREEN, LIME, CHAR, GREY = "#1f4d3a", "#8fb339", "#2b2b2b", "#9aa39c"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150})


def to_jsonable(o):
    if isinstance(o, dict):
        # DROP names raw arrays/tables (fitted values, bootstrap draws…); nested result dicts are always kept.
        return {str(k): to_jsonable(v) for k, v in o.items()
                if not callable(v) and (k not in DROP or isinstance(v, dict))}
    if isinstance(o, pd.DataFrame):
        d = o.reset_index() if not isinstance(o.index, pd.RangeIndex) else o
        return json.loads(d.round(6).to_json(orient="records"))
    if isinstance(o, pd.Series):
        return {str(k): to_jsonable(v) for k, v in o.round(6).items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [to_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 6)
    return o


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def figures(df, res):
    FIG.mkdir(parents=True, exist_ok=True)
    g = res["global_trend"]

    w = g["series"].loc[1850:] / 1000
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(w.index, w.values, color=GREEN, alpha=0.15)
    ax.plot(w.index, w.values, color=GREEN, lw=2)
    ax.set_title("World fossil & industry CO₂ emissions, 1850–2024")
    ax.set_ylabel("Billion tonnes CO₂ (Gt)")
    ax.annotate(f"2024: {g['co2_latest']/1000:.1f} Gt", (2024, g["co2_latest"] / 1000), xytext=(1960, 36),
                arrowprops={"arrowstyle": "->", "color": CHAR})
    save(fig, "01_global_trend")

    t = res["top"]["table"].iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    axes[0].barh(t["country"], t["co2"] / 1000, color=GREEN)
    axes[0].set_title("Total CO₂, 2024 (Gt)")
    axes[1].barh(t["country"], t["co2_per_capita"], color=LIME)
    axes[1].set_title("CO₂ per person, 2024 (t)")
    save(fig, "02_total_vs_per_capita")

    p = res["distribution"]
    x = p["data"]["co2_per_capita"]
    ft = p["freq_table"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [3, 1.2]})
    axes[0].bar(ft["mid"], ft["f"], width=ft.attrs["width"] * 0.95, color=GREEN)
    axes[0].plot(ft["mid"], ft["f"], color=LIME, marker="o", lw=1.5, label="frequency polygon")
    for v, lab, c in [(p["summary"]["mean"], "mean", "#c0392b"), (p["summary"]["median"], "median", CHAR)]:
        axes[0].axvline(v, color=c, ls="--", lw=1.2, label=f"{lab} {v:.2f}")
    axes[0].set_title(f"Histogram: CO₂ per person, {config.LATEST_YEAR} (n={len(x)}, width {ft.attrs['width']:g} t)")
    axes[0].set_xlabel("tonnes per person")
    axes[0].set_ylabel("number of countries")
    axes[0].legend(frameon=False)
    axes[1].boxplot(x, orientation="vertical", widths=0.5, patch_artist=True, boxprops={"facecolor": "#e6efe9"},
                    medianprops={"color": GREEN})
    axes[1].set_title("Box plot (1.5×IQR)")
    axes[1].set_xticks([])
    save(fig, "03_distribution")

    rel = res["relationships"]["data"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = dict(zip(sorted(rel["continent"].dropna().unique()), ["#1f4d3a", "#8fb339", "#d98c3f", "#3d7ea6", "#9b5fa5", "#c0392b"]))
    for cont, d in rel.groupby("continent"):
        ax.scatter(d["gdp_per_capita"], d["co2_per_capita"], s=np.sqrt(d["population"]) / 60, alpha=0.6,
                   color=colors[cont], label=cont, edgecolor="white", lw=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    r = res["relationships"]["gdp_log"]["r"]
    ax.set_title(f"Bubble chart: GDP vs CO₂ per person, {config.GDP_YEAR} (log–log r = {r:.3f})")
    ax.set_xlabel("GDP per person (int-$, 2011 prices, log scale)")
    ax.set_ylabel("CO₂ per person (t, log scale)")
    ax.legend(frameon=False, fontsize=8, markerscale=0.5)
    save(fig, "04_bubble_gdp_co2")

    inf = res["inference"]
    ad = res["inference"]["anova_data"]
    order = ad.groupby("continent")["log_pc"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.boxplot([10 ** ad.loc[ad["continent"] == c, "log_pc"] for c in order], tick_labels=order, patch_artist=True,
               boxprops={"facecolor": "#e6efe9"}, medianprops={"color": GREEN})
    ax.set_yscale("log")
    ax.set_title(f"CO₂ per person by continent, {config.LATEST_YEAR} — one-way ANOVA F = {inf['anova']['F']:.1f}")
    ax.set_ylabel("t per person (log)")
    save(fig, "05_anova_continents")

    clt = res["inference"]["clt"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(x, bins=30, color=GREY)
    axes[0].set_title("Population: country CO₂ per person (skewed)")
    axes[1].hist(clt["means"], bins=50, color=GREEN, density=True)
    xs = np.linspace(clt["means"].min(), clt["means"].max(), 200)
    axes[1].plot(xs, np.exp(-(xs - clt["population_mean"]) ** 2 / (2 * clt["theoretical_se"] ** 2))
                 / np.sqrt(2 * np.pi * clt["theoretical_se"] ** 2), color=LIME, lw=2, label="Normal(μ, σ/√n)")
    axes[1].set_title("10,000 sample means, n = 30 (CLT)")
    axes[1].legend(frameon=False)
    save(fig, "06_clt")

    reg = res["regression"]
    rd = reg["data"]
    s = reg["simple"]
    fig, ax = plt.subplots(figsize=(7, 4))
    lx, ly = np.log10(rd["gdp_per_capita"]), np.log10(rd["co2_per_capita"])
    ax.scatter(lx, ly, s=14, color=GREEN, alpha=0.6)
    xs = np.linspace(lx.min(), lx.max(), 50)
    ax.plot(xs, s["slope"] * xs + s["intercept"], color=LIME, lw=2.5,
            label=f"log CO₂pc = {s['slope']:.3f}·log GDPpc {s['intercept']:+.3f}  (R² = {s['metrics']['R2']:.3f})")
    ax.set_xlabel("log₁₀ GDP per person")
    ax.set_ylabel("log₁₀ CO₂ per person")
    ax.set_title(f"Simple linear regression, {config.GDP_YEAR} (n = {s['n']})")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "07_regression")

    lt = res["lof"]["table"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sc = ax.scatter(lt["log_gdp_pc"], lt["log_co2_pc"], c=lt["lof"], cmap="YlGn", s=20 + 40 * (lt["lof"] - 1).clip(0),
                    edgecolor=CHAR, lw=0.3)
    for _, row in lt.head(6).iterrows():
        ax.annotate(row["country"], (row["log_gdp_pc"], row["log_co2_pc"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    fig.colorbar(sc, label="LOF score")
    ax.set_xlabel("log₁₀ GDP per person")
    ax.set_ylabel("log₁₀ CO₂ per person")
    ax.set_title(f"Local Outlier Factor (Manhattan, k = {res['lof']['k']})")
    save(fig, "08_lof")

    tsr = res["time_series"]
    d = tsr["detail"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sr = tsr["series"].loc[2000:] / 1000
    ax.plot(sr.index, sr.values, color=CHAR, lw=2, label="Actual")
    ax.plot(tsr["cma5"].loc[2000:].index, tsr["cma5"].loc[2000:] / 1000, color=GREY, ls=":", label="Centred 5-yr MA (trend)")
    styles = {"Moving average (k=3)": (LIME, "-"), "Naïve (last value)": ("#3d7ea6", "--"),
              "Linear trend (2000–2019)": ("#c0392b", "-.")}
    for col, (c, ls) in styles.items():
        ax.plot(d["year"], d[col] / 1000, color=c, ls=ls, marker="o", ms=3, label=col)
    ax.axvline(2019.5, color=GREY, lw=0.8)
    ax.text(2019.7, sr.min() + 0.5, "hold-out →", fontsize=8, color=GREY)
    ax.set_title("World CO₂: forecasts on the 2020–2024 hold-out")
    ax.set_ylabel("Gt CO₂")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "09_forecasts")

    c = data.countries(df)
    yrs = [1990, 2000, 2010, 2020, 2024]
    fuel = c[c["year"].isin(yrs)].groupby("year")[config.FUEL_COLUMNS].sum() / 1000
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bottom = np.zeros(len(yrs))
    palette = [CHAR, GREEN, LIME, "#d98c3f", "#3d7ea6", GREY]
    for col, colr in zip(config.FUEL_COLUMNS, palette):
        ax.bar([str(y) for y in yrs], fuel[col], bottom=bottom, color=colr, label=col.replace("_co2", ""))
        bottom += fuel[col].to_numpy()
    ax.set_title("Sub-divided bar: CO₂ by source (sum of countries)")
    ax.set_ylabel("Gt CO₂")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    save(fig, "10_fuel_mix")

    ux = res["univariate_extras"]
    ft, cum = ux["freq_table"], ux["cumulative"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(np.r_[ft["lower"].iloc[0], ft["upper"]], np.r_[0, cum["less_than_cf"]], color=GREEN, marker="o", ms=3, label="'less than' ogive")
    ax.plot(np.r_[ft["lower"], ft["upper"].iloc[-1]], np.r_[cum["more_than_cf"], 0], color=LIME, marker="o", ms=3, label="'more than' ogive")
    ax.plot([ux["ogive_median"]], [ft["f"].sum() / 2], "x", color="#c0392b", ms=10, mew=2.5, label=f"median = {ux['ogive_median']:.2f} t")
    ax.set_title(f"Ogives: CO₂ per person, {config.LATEST_YEAR}")
    ax.set_xlabel("tonnes per person")
    ax.set_ylabel("cumulative countries")
    ax.legend(frameon=False)
    save(fig, "11_ogive")

    dg = res["diagrams"]
    fp = dg["fuel_pct"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    left = np.zeros(len(fp))
    for col, colr in zip(fp.columns, [CHAR, GREEN, LIME, "#d98c3f", "#3d7ea6", GREY]):
        ax.barh(fp.index, fp[col], left=left, color=colr, label=col)
        left += fp[col].to_numpy()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of continent's CO₂")
    ax.set_title(f"Percentage bar diagram: CO₂ by source, {config.LATEST_YEAR}")
    ax.legend(frameon=False, ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "12_percentage_bar")

    rc = dg["rectangles"].sort_values("co2_per_capita", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x0 = 0.0
    for i, (_, row) in enumerate(rc.iterrows()):
        w_ = row["population"] / 1e6
        ax.bar(x0, row["co2_per_capita"], width=w_ * 0.98, align="edge", color=GREEN if i % 2 == 0 else LIME)
        if w_ > 120:
            ax.text(x0 + w_ / 2, row["co2_per_capita"] + 0.3, row["country"], ha="center", fontsize=8)
        x0 += w_
    ax.set_xlabel("population (millions, cumulative)")
    ax.set_ylabel("t CO₂ per person")
    ax.set_title("Rectangular diagram: width = population, height = per person, area = total CO₂ (top 10)")
    save(fig, "13_rectangles")


def main():
    df = data.load_clean(rebuild=True)
    res = R.compute_all(df)
    figures(df, res)
    entries = concepts.build(res)
    out = to_jsonable(res)
    out["global_trend"]["series"] = to_jsonable(res["global_trend"]["series"].loc[1950:])
    out["concepts"] = entries
    (config.DOCS_DIR / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (config.DOCS_DIR / "CONCEPT_MATRIX.md").write_text(concepts.to_markdown(entries))
    print("wrote results.json, CONCEPT_MATRIX.md (", len(entries), "concepts ) and",
          len(list(FIG.glob("*.png"))), "figures")


if __name__ == "__main__":
    main()
