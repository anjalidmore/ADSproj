"""Paths and analysis constants shared by the pipeline, app and tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"

RAW_CO2 = RAW_DIR / "owid_co2.csv"
RAW_CODEBOOK = RAW_DIR / "owid_co2.codebook.csv"
RAW_CONTINENTS = RAW_DIR / "owid_continents.csv"
MANIFEST = RAW_DIR / "MANIFEST.json"

CLEAN_PARQUET = PROCESSED_DIR / "co2_clean.parquet"

# Latest year in the snapshot, and latest year with GDP (GDP series ends in 2022).
LATEST_YEAR = 2024
GDP_YEAR = 2022
# Baseline for change comparisons: 2005 (Kyoto Protocol entered into force).
BASE_YEAR = 2005

SEED = 42
ALPHA = 0.05

# Entities without an ISO code that are genuine countries.
COUNTRIES_WITHOUT_ISO = {"Kosovo"}
# Emission sources that the "World" total includes but no country owns.
INTERNATIONAL = {"International aviation", "International shipping"}
# Entities that are neither countries nor regional aggregates.
SPECIAL = {"Kuwaiti Oil Fires", "Kuwaiti Oil Fires (GCP)", "Ryukyu Islands", "Ryukyu Islands (GCP)"}

# Fossil/industry CO2 columns that cannot be negative by definition.
NON_NEGATIVE = [
    "co2", "coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2",
    "other_industry_co2", "population", "gdp", "primary_energy_consumption",
]

FUEL_COLUMNS = ["coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2", "other_industry_co2"]
