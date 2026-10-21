"""Module 1 — data loading and preparation.

Three layers are kept distinct:
    raw            data/raw/*.csv, never modified (hash-checked against MANIFEST.json)
    clean          every raw row kept, plus entity_type, continent, derived columns and flags
    analysis-ready country-only cross-sections / series returned by the helper functions
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import config


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw() -> dict:
    """Compare the raw files on disk with the hashes recorded at download time."""
    manifest = json.loads(config.MANIFEST.read_text())
    out = {}
    for name, meta in manifest["files"].items():
        actual = sha256(config.RAW_DIR / name)
        out[name] = {"expected": meta["sha256"], "actual": actual, "ok": actual == meta["sha256"]}
    return out


def load_raw() -> pd.DataFrame:
    return pd.read_csv(config.RAW_CO2)


def load_codebook() -> pd.DataFrame:
    return pd.read_csv(config.RAW_CODEBOOK)


def load_continents() -> pd.DataFrame:
    return pd.read_csv(config.RAW_CONTINENTS)


def classify_entity(country: str, iso_code) -> str:
    """country | world | international | special | aggregate.

    OWID gives every country an ISO code except Kosovo; every entity without one is a
    region, income group or special source. Summing aggregates together with countries
    would double-count, so analyses use entity_type == "country" only.
    """
    if country == "World":
        return "world"
    if country in config.INTERNATIONAL:
        return "international"
    if country in config.SPECIAL:
        return "special"
    if isinstance(iso_code, str) or country in config.COUNTRIES_WITHOUT_ISO:
        return "country"
    return "aggregate"


def attach_continent(df: pd.DataFrame, continents: pd.DataFrame) -> pd.Series:
    """Map countries to OWID continents by ISO code; Kosovo (no ISO code) by name."""
    by_code = continents.dropna(subset=["code"]).set_index("code")["owid_region"]
    by_name = continents.set_index("entity")["owid_region"]
    cont = df["iso_code"].map(by_code)
    no_iso = df["iso_code"].isna() & df["country"].isin(config.COUNTRIES_WITHOUT_ISO)
    cont[no_iso] = df.loc[no_iso, "country"].map(by_name)
    cont[df["entity_type"] != "country"] = np.nan
    return cont


def clean(raw: pd.DataFrame, continents: pd.DataFrame) -> pd.DataFrame:
    """Build the clean layer. No rows are deleted and no values are imputed.

    Missing values stay missing (a blank is "not reported", never zero emissions).
    Suspicious values are flagged, not altered, so they can be investigated.
    """
    df = raw.copy()
    df["entity_type"] = [classify_entity(c, i) for c, i in zip(df["country"], df["iso_code"])]
    df["continent"] = attach_continent(df, continents)

    # Derived indicator: GDP per person (international-$, 2011 prices).
    df["gdp_per_capita"] = df["gdp"] / df["population"]

    # Flags (kept as columns so the Dataset Explorer can show them).
    neg = np.zeros(len(df), dtype=bool)
    for col in config.NON_NEGATIVE:
        neg |= df[col].lt(0).fillna(False).to_numpy()
    df["flag_negative_nonneg_col"] = neg
    df["flag_zero_co2"] = df["co2"].eq(0) & df["entity_type"].eq("country")
    # Fuel components should sum to the total (allowing for rounding).
    parts = df[config.FUEL_COLUMNS].sum(axis=1, min_count=1)
    df["fuel_sum_gap"] = df["co2"] - parts
    return df


def build_clean() -> pd.DataFrame:
    return clean(load_raw(), load_continents())


def load_clean(rebuild: bool = False) -> pd.DataFrame:
    """Read the processed parquet if present, otherwise build (and save) it."""
    if config.CLEAN_PARQUET.exists() and not rebuild:
        return pd.read_parquet(config.CLEAN_PARQUET)
    df = build_clean()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.CLEAN_PARQUET, index=False)
    return df


# ---------------------------------------------------------------- analysis-ready views

def countries(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["entity_type"] == "country"]


def cross_section(df: pd.DataFrame, year: int, cols: list[str]) -> pd.DataFrame:
    """Country rows for one year with all requested columns present (listwise for these cols only)."""
    keep = ["country", "iso_code", "continent", "year", "population"]
    sub = countries(df)
    sub = sub.loc[sub["year"] == year, list(dict.fromkeys(keep + cols))]
    return sub.dropna(subset=cols).reset_index(drop=True)


def world_series(df: pd.DataFrame, col: str = "co2", start: int | None = None) -> pd.Series:
    s = df.loc[df["country"] == "World"].set_index("year")[col].dropna()
    return s if start is None else s.loc[start:]


def paired_years(df: pd.DataFrame, col: str, y0: int, y1: int) -> pd.DataFrame:
    """Countries observed in both years for `col` (same units, for paired comparisons)."""
    c = countries(df)
    a = c.loc[c["year"] == y0, ["country", "continent", col]].rename(columns={col: f"{col}_{y0}"})
    b = c.loc[c["year"] == y1, ["country", col]].rename(columns={col: f"{col}_{y1}"})
    return a.merge(b, on="country").dropna().reset_index(drop=True)
