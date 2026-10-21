"""Module 1 — data exploration / data-quality report (computed, never assumed)."""

from __future__ import annotations

import pandas as pd

from . import config, data


def missing_table(df: pd.DataFrame, codebook: pd.DataFrame | None = None) -> pd.DataFrame:
    t = pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "dtype": df.dtypes.astype(str),
    })
    if codebook is not None:
        cb = codebook.set_index("column")
        t["unit"] = cb["unit"].reindex(t.index)
        t["description"] = cb["title"].reindex(t.index)
    return t.sort_values("missing_pct")


def measurement_level(col: str) -> str:
    """Module 2.1 levels of measurement for the dataset's variables."""
    if col in {"country", "iso_code", "continent", "entity_type"}:
        return "Nominal (categorical)"
    if col == "year":
        return "Interval (numeric, no true zero)"
    if "temperature" in col or "growth" in col or col.startswith("trade_co2"):
        return "Interval (numeric, can be negative)"
    if col.startswith("flag_"):
        return "Nominal (boolean flag)"
    return "Ratio (numeric, true zero)"


def world_reconciliation(df: pd.DataFrame, years=(1950, 1990, 2005, 2022, 2024)) -> pd.DataFrame:
    """World total vs sum of countries + international transport (double-counting check)."""
    c = data.countries(df).groupby("year")["co2"].sum(min_count=1)
    intl = df[df["entity_type"] == "international"].groupby("year")["co2"].sum(min_count=1)
    special = df[df["country"] == "Kuwaiti Oil Fires"].set_index("year")["co2"]
    world = data.world_series(df)
    t = pd.DataFrame({"world": world, "sum_countries": c, "international": intl, "kuwaiti_fires": special})
    t = t.reindex(list(years))
    t["countries_plus_intl"] = t["sum_countries"] + t["international"].fillna(0) + t["kuwaiti_fires"].fillna(0)
    t["residual"] = t["world"] - t["countries_plus_intl"]
    t["residual_pct"] = t["residual"] / t["world"] * 100
    return t


def coverage_by_year(df: pd.DataFrame, cols=("co2", "co2_per_capita", "gdp", "primary_energy_consumption")) -> pd.DataFrame:
    return data.countries(df).groupby("year")[list(cols)].count()


def report(df_raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Headline data-quality facts for the dashboard, report and viva."""
    types = df["entity_type"].value_counts().to_dict()
    no_iso = sorted(df_raw.loc[df_raw["iso_code"].isna(), "country"].unique())
    zero = df[df["flag_zero_co2"]]
    return {
        "rows": int(df_raw.shape[0]),
        "columns": int(df_raw.shape[1]),
        "year_min": int(df_raw["year"].min()),
        "year_max": int(df_raw["year"].max()),
        "entities": int(df_raw["country"].nunique()),
        "countries": int(data.countries(df)["country"].nunique()),
        "entity_type_rows": {k: int(v) for k, v in types.items()},
        "entities_without_iso": no_iso,
        "duplicate_rows": int(df_raw.duplicated().sum()),
        "duplicate_keys": int(df_raw.duplicated(["country", "year"]).sum()),
        "numeric_columns": int(df_raw.select_dtypes("number").shape[1]),
        "text_columns": int(df_raw.select_dtypes(exclude="number").shape[1]),
        "zero_co2_country_rows": int(len(zero)),
        "zero_co2_after_1950": int((zero["year"] >= 1950).sum()),
        "negative_values_in_nonnegative_columns": int(df["flag_negative_nonneg_col"].sum()),
        "max_fuel_sum_gap_mt": float(df["fuel_sum_gap"].abs().max()),
        "gdp_last_year": int(df.loc[df["gdp"].notna(), "year"].max()),
        "energy_first_year": int(df.loc[df["primary_energy_consumption"].notna(), "year"].min()),
        "continent_missing": sorted(data.countries(df).loc[lambda d: d["continent"].isna(), "country"].unique()),
        "raw_hashes_ok": all(v["ok"] for v in data.verify_raw().values()),
        "latest_year": config.LATEST_YEAR,
    }
