"""Data-pipeline checks against the raw snapshot."""

import pytest

from co2ads import config, data, quality


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


@pytest.fixture(scope="module")
def clean(raw):
    return data.clean(raw, data.load_continents())


def test_raw_snapshot_unchanged():
    assert all(v["ok"] for v in data.verify_raw().values())


def test_shape_and_keys(raw):
    assert raw.shape == (50411, 79)
    assert raw.duplicated(["country", "year"]).sum() == 0


def test_clean_keeps_every_row_and_value(raw, clean):
    assert len(clean) == len(raw)
    assert clean["co2"].isna().sum() == raw["co2"].isna().sum()   # nothing imputed


def test_entity_classification(clean):
    types = clean.groupby("country")["entity_type"].first()
    assert types["Kosovo"] == "country"
    assert types["World"] == "world"
    assert types["Europe"] == "aggregate"
    assert types["Asia (GCP)"] == "aggregate"
    assert types["International shipping"] == "international"
    assert types["China"] == "country"


def test_every_country_but_antarctica_has_continent(clean):
    missing = set(data.countries(clean).loc[lambda d: d["continent"].isna(), "country"])
    assert missing == {"Antarctica"}
    assert clean.loc[clean["country"] == "Kosovo", "continent"].iloc[0] == "Europe"


def test_world_equals_countries_plus_international(clean):
    t = quality.world_reconciliation(clean, years=(1990, 2005, 2024))
    assert (t["residual_pct"].abs() < 0.5).all()


def test_cross_section_is_country_only(clean):
    cs = data.cross_section(clean, config.LATEST_YEAR, ["co2_per_capita"])
    assert cs["country"].is_unique
    assert "World" not in set(cs["country"])
    assert cs["co2_per_capita"].notna().all()
