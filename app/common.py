"""Shared helpers for the Streamlit pages: cached data, palette, formatting, chart style."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from co2ads import concepts, config, data, quality  # noqa: E402
from co2ads import results as R  # noqa: E402

GREEN, LIME, CHAR, CREAM, GREY = "#1f4d3a", "#8fb339", "#1d211e", "#f7f5ef", "#9aa39c"
CONTINENT_COLORS = {"Africa": "#d98c3f", "Asia": "#c0392b", "Europe": "#3d7ea6",
                    "North America": "#1f4d3a", "Oceania": "#9b5fa5", "South America": "#8fb339"}
SEQ = ["#f1f5e9", "#cfe1a8", "#8fb339", "#3f7a4a", "#1f4d3a"]

pio.templates["co2"] = go.layout.Template(layout=go.Layout(
    font={"family": "Inter, Helvetica, Arial, sans-serif", "color": CHAR, "size": 13},
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[GREEN, LIME, "#d98c3f", "#3d7ea6", "#9b5fa5", "#c0392b", GREY],
    xaxis={"gridcolor": "#e2e6df", "zeroline": False}, yaxis={"gridcolor": "#e2e6df", "zeroline": False},
    margin={"l": 10, "r": 10, "t": 50, "b": 10}, hoverlabel={"bgcolor": "white"},
))
pio.templates.default = "co2"

NUMERIC_VARS = {
    "co2_per_capita": "CO₂ per person (t)",
    "co2": "Annual CO₂ (Mt)",
    "gdp_per_capita": "GDP per person (int-$ 2011)",
    "energy_per_capita": "Energy per person (kWh)",
    "co2_per_gdp": "CO₂ per $ of GDP (kg/$)",
    "co2_per_unit_energy": "CO₂ per unit energy (kg/kWh)",
    "coal_co2_per_capita": "Coal CO₂ per person (t)",
    "cumulative_co2": "Cumulative CO₂ (Mt)",
    "population": "Population",
    "consumption_co2_per_capita": "Consumption-based CO₂ per person (t)",
}

CSS = f"""
<style>
h1, h2, h3 {{ font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.01em; }}
h1 {{ font-size: 2.6rem !important; color: {GREEN}; }}
.kpi {{ border-top: 3px solid {GREEN}; padding: 0.6rem 0 0.2rem 0; }}
.kpi .v {{ font-family: Georgia, serif; font-size: 2.3rem; line-height: 1.1; color: {CHAR}; }}
.kpi .l {{ font-size: 0.85rem; color: #4a524c; }}
.kpi .s {{ font-size: 0.78rem; color: #6b746d; }}
.note {{ background: #e8ede6; border-left: 4px solid {LIME}; padding: 0.7rem 1rem; border-radius: 4px; font-size: 0.92rem; }}
.small {{ font-size: 0.82rem; color: #5b635d; }}
</style>
"""


def page(title: str, subtitle: str = ""):
    st.set_page_config(page_title=f"{title} · CO₂ Analytics", page_icon="🌍", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f"<p class='small'>{subtitle}</p>", unsafe_allow_html=True)


def kpi(col, value: str, label: str, sub: str = ""):
    col.markdown(f"<div class='kpi'><div class='v'>{value}</div><div class='l'>{label}</div>"
                 f"<div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def note(text: str):
    st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading and cleaning the OWID dataset…")
def clean_df() -> pd.DataFrame:
    return data.load_clean()


@st.cache_resource(show_spinner=False)
def raw_df() -> pd.DataFrame:
    return data.load_raw()


@st.cache_resource(show_spinner=False)
def codebook() -> pd.DataFrame:
    return data.load_codebook()


@st.cache_resource(show_spinner="Running all analyses…")
def all_results() -> dict:
    return R.compute_all(clean_df())


@st.cache_resource(show_spinner=False)
def concept_entries() -> list[dict]:
    return concepts.build(all_results())


def countries_df() -> pd.DataFrame:
    return data.countries(clean_df())


def label(col: str) -> str:
    return NUMERIC_VARS.get(col, col)


def fmt(x, d=2) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:,.{d}f}"


def csv_button(df: pd.DataFrame, name: str, text: str = "Download CSV"):
    st.download_button(text, df.to_csv(index=False).encode(), file_name=name, mime="text/csv")


def year_options(col: str, min_n: int = 30) -> list[int]:
    n = countries_df().groupby("year")[col].count()
    return sorted(n[n >= min_n].index.tolist())


__all__ = ["config", "data", "quality", "R", "concepts"]
