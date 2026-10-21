import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/common.py importable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (CONTINENT_COLORS, GREEN, LIME, all_results, clean_df, config, countries_df, csv_button, data, fmt,
                    label, note, page)
from co2ads import lof as lofm
from co2ads import regression as rg
from co2ads import timeseries as ts

page("Advanced Analytics", "Module 4 regression · Module 1 model evaluation · Module 5 local outlier factor · Module 6 time series")

df = clean_df()
t_reg, t_lof, t_ts, t_dec = st.tabs(["Regression", "Local Outlier Factor", "Time-series forecasting", "Decomposition (course example)"])

PRED = ["gdp_per_capita", "energy_per_capita", "co2_per_unit_energy", "population"]

with t_reg:
    c1, c2, c3 = st.columns(3)
    year = c1.selectbox("Year (GDP ends 2022)", list(range(config.GDP_YEAR, 1989, -1)), key="ry")
    x1 = c2.selectbox("X₁", PRED, format_func=label)
    x2 = c3.selectbox("X₂ (multiple regression)", [p for p in PRED if p != x1], format_func=label)
    st.caption("Target y = log₁₀ CO₂ per person; predictors are log₁₀-transformed (all variables are right-skewed). "
               "Coefficients on log–log scales are elasticities.")
    cs = data.cross_section(df, year, ["co2_per_capita", x1, x2])
    cs = cs[(cs[["co2_per_capita", x1, x2]] > 0).all(axis=1)]
    if len(cs) < 15:
        st.warning("Too few countries with all three variables.")
    else:
        lx1, lx2, ly = np.log10(cs[x1]).to_numpy(), np.log10(cs[x2]).to_numpy(), np.log10(cs["co2_per_capita"]).to_numpy()
        s = rg.simple_regression(lx1, ly)
        m = rg.multiple_regression_2(lx1, lx2, ly)
        st.markdown(f"#### Simple regression (n = {s['n']})")
        st.latex(rf"\widehat{{\log y}} = {s['slope']:.4f}\,\log x_1 {s['intercept']:+.4f}\qquad R^2 = {s['metrics']['R2']:.4f},\ t_{{slope}} = {s['t_slope']:.2f}")
        fig = px.scatter(x=lx1, y=ly, hover_name=cs["country"], color=cs["continent"], color_discrete_map=CONTINENT_COLORS,
                         labels={"x": f"log₁₀ {label(x1)}", "y": "log₁₀ CO₂ per person"})
        xs = np.linspace(lx1.min(), lx1.max(), 50)
        fig.add_scatter(x=xs, y=s["slope"] * xs + s["intercept"], mode="lines", line={"color": "#1d211e", "width": 3}, name="regression line")
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")
        met = pd.DataFrame({"simple": s["metrics"], "multiple": m["metrics"]}).T
        met["MAPE (back in tonnes)"] = [rg.mape(cs["co2_per_capita"], 10 ** s["fitted"]), rg.mape(cs["co2_per_capita"], 10 ** m["fitted"])]
        st.markdown("#### Multiple regression (two predictors, course deviation-sum formulas)")
        st.latex(rf"\widehat{{\log y}} = {m['b0']:.4f} {m['b1']:+.4f}\,\log X_1 {m['b2']:+.4f}\,\log X_2")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("R²", fmt(m["r2"], 4))
        k2.metric("Adjusted R²", fmt(m["adj_r2"], 4))
        k3.metric("F (overall)", fmt(m["F"], 1), f"crit {m['F_critical']:.2f} → {'significant' if m['F'] >= m['F_critical'] else 'not significant'}", delta_color="off")
        k4.metric("r₁₂ between predictors", fmt(m["r12"], 3))
        an = m["anova"]
        st.dataframe(pd.DataFrame({"SS": [an["ss_regression"], an["ss_error"], an["ss_total"]],
                                   "df": [an["df_regression"], an["df_error"], an["df_regression"] + an["df_error"]],
                                   "MS": [an["ms_regression"], an["ms_error"], np.nan]},
                                  index=["Regression", "Error", "Total"]).round(4), width="stretch")
        coef = pd.DataFrame(m["coefficients"]).assign(term=[f"b1 ({label(x1)})", f"b2 ({label(x2)})"])
        st.dataframe(coef.round(4), hide_index=True, width="stretch")
        st.caption(f"t critical (α = 0.05, df = {an['df_error']}) = {m['t_critical']:.3f}. Test individual coefficients only after the F-test rejects H₀.")
        st.markdown("**Error measures** (MAE/MSE/RMSE on the log₁₀ scale)")
        st.dataframe(met.round(4), width="stretch")
        if abs(m["r12"]) > 0.8:
            note(f"Predictors are highly correlated (r₁₂ = {m['r12']:.2f}): <b>multicollinearity</b>. Each partial coefficient "
                 "then measures an effect 'holding the other constant' for combinations that rarely occur, so standard errors "
                 "grow and a predictor can look insignificant even though it is strongly related to y on its own.")
        st.markdown("#### Model evaluation (Module 1): hold-out and k-fold cross-validation")
        k = st.slider("k folds", 3, 10, 5)
        X2 = np.column_stack([lx1, lx2])
        cv1, cv2 = rg.kfold_cv(lx1.reshape(-1, 1), ly, k=k, seed=config.SEED), rg.kfold_cv(X2, ly, k=k, seed=config.SEED)
        ho = rg.holdout(X2, ly, seed=config.SEED)
        cvt = pd.DataFrame({"simple (mean of folds)": cv1["mean"], "multiple (mean of folds)": cv2["mean"],
                            "multiple: hold-out test (⅓)": {kk: ho["test"][kk] for kk in cv2["mean"]}}).T
        st.dataframe(cvt.round(4), width="stretch")
        st.dataframe(pd.DataFrame(cv2["folds"]).round(4), hide_index=True, width="stretch")
        st.caption(f"Hold-out: {ho['n_train']} countries train, {ho['n_test']} test, seed {config.SEED}. CV R² close to training R² → no overfitting.")
        fig = px.scatter(x=m["fitted"], y=m["residuals"], hover_name=cs["country"], labels={"x": "fitted log₁₀ y", "y": "residual"},
                         color_discrete_sequence=[GREEN])
        fig.add_hline(0, line_color="#1d211e")
        fig.update_layout(height=340, title="Residuals vs fitted (multiple model): no strong pattern = linear form adequate")
        st.plotly_chart(fig, width="stretch")

with t_lof:
    c1, c2, c3 = st.columns(3)
    year = c1.selectbox("Year", list(range(config.GDP_YEAR, 1989, -1)), key="ly")
    k = c2.slider("k (neighbours)", 2, 30, 10)
    thr = c3.slider("Flag if LOF >", 1.1, 3.0, 1.5, step=0.1)
    cs = data.cross_section(df, year, ["co2_per_capita", "gdp_per_capita"])
    cs = cs.assign(log_co2_pc=np.log10(cs["co2_per_capita"]), log_gdp_pc=np.log10(cs["gdp_per_capita"]))
    if len(cs) <= k:
        st.warning("k must be smaller than the number of countries.")
    else:
        tab = lofm.lof_table(cs, ["log_gdp_pc", "log_co2_pc"], k=k).merge(cs[["country", "continent", "co2_per_capita", "gdp_per_capita"]], on="country")
        tab["flag"] = tab["lof"] > thr
        fig = px.scatter(tab, x="log_gdp_pc", y="log_co2_pc", color="lof", size=(tab["lof"] - 0.8).clip(0.2), hover_name="country",
                         color_continuous_scale=["#f1f5e9", LIME, GREEN, "#0f2a1f"],
                         labels={"log_gdp_pc": "log₁₀ GDP per person", "log_co2_pc": "log₁₀ CO₂ per person", "lof": "LOF"})
        for _, r in tab[tab["flag"]].iterrows():
            fig.add_annotation(x=r["log_gdp_pc"], y=r["log_co2_pc"], text=r["country"], showarrow=False, yshift=12, font={"size": 10})
        fig.update_layout(height=520, title=f"LOF with Manhattan distance on min–max scaled values, k = {k}: {int(tab['flag'].sum())} flagged")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(tab.head(20)[["country", "continent", "co2_per_capita", "gdp_per_capita", "k_distance", "lrd", "lof"]].round(4),
                     hide_index=True, width="stretch")
        csv_button(tab, f"lof_{year}_k{k}.csv")
        note("Steps (Module 5): Manhattan distance → k-distance and k-neighbourhood → reachability distance "
             "max(k-dist(o), d(p,o)) → local reachability density lrd = |N_k| / Σ reach-dist → LOF = mean lrd of neighbours / own lrd. "
             "LOF ≈ 1 means as dense as the neighbours; well above 1 means isolated. Unlike the IQR rule it is two-dimensional and local, "
             "so it also finds countries that emit unusually little for their income.")

with t_ts:
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
    who = c1.selectbox("Series", ["World"] + sorted(countries_df()["country"].unique()), key="ts_c")
    train_end = c2.slider("Train up to", 2005, config.LATEST_YEAR - 2, 2019)
    kma = c3.slider("Moving-average k", 2, 10, 3)
    tstart = c4.slider("Trend fitted from", 1950, train_end - 5, 2000)
    s = df[df["country"] == who].set_index("year")["co2"].dropna().loc[1950:]
    if len(s) < 20 or s.index.max() <= train_end:
        st.warning("Not enough data for this series.")
    else:
        comp = ts.holdout_comparison(s, train_end=train_end, k=kma, trend_start=tstart)
        st.dataframe(comp["table"].round(3), width="stretch")
        best = comp["table"]["MASE"].idxmin()
        st.caption(f"Evaluated on {s.index.max() - train_end} hold-out years. MASE < 1 = better than the naïve 'same as last year' forecast "
                   f"(scaled on the training period). Best here: **{best}**.")
        fig = go.Figure()
        fig.add_scatter(x=s.index, y=s.values, name="actual", line={"color": "#1d211e", "width": 2.5})
        cma = ts.centred_moving_average(s.to_numpy(), 5)
        fig.add_scatter(x=s.index, y=cma, name="trend: centred 5-yr MA", line={"color": "#9aa39c", "dash": "dot"})
        palette = ["#c0392b", LIME, "#3d7ea6", GREEN]
        test = comp["years"] > train_end
        for (name, f), col in zip(comp["forecasts"].items(), palette):
            fig.add_scatter(x=comp["years"][test], y=f[test], name=name, mode="lines+markers", line={"color": col})
        fig.add_vline(train_end + 0.5, line_color="#9aa39c")
        fig.update_layout(height=460, title=f"{who}: one-step-ahead forecasts vs actual (Mt CO₂)", yaxis_title="Mt CO₂")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(comp["detail"].round(1), hide_index=True, width="stretch")
        tf = comp["trend_fit"]
        st.latex(rf"\text{{Least-squares trend ({tstart}–{train_end})}}:\ \hat y = {tf['slope']:.2f}\,t {tf['intercept']:+,.0f},\ R^2 = {tf['r2']:.3f}")
        note("Average and moving-average forecasts follow the course definition: each forecast uses the actual values up to the "
             "previous year. The trend line is extrapolated several years ahead, so it cannot react to shocks such as 2020. "
             "Annual data has no seasonality, so seasonal indices are not estimated (see the next tab).")

with t_dec:
    st.markdown("Classical decomposition needs a repeating within-year pattern. OWID data is **annual**, so the method is demonstrated "
                "and verified on the course's quarterly-sales example, and only its trend step is used on CO₂.")
    y = [18, 20, 34, 44, 22, 20, 38, 48, 18, 20, 42, 56]
    model = st.radio("Model", ["additive", "multiplicative"], horizontal=True)
    d = ts.classical_decomposition(y, 4, model)
    lab = [f"Y{i // 4 + 1}Q{i % 4 + 1}" for i in range(12)]
    st.dataframe(pd.DataFrame({"t": lab, "Y": y, "T (2×4 centred MA)": d["trend"],
                               ("Y − T" if model == "additive" else "Y / T"): d["detrended"],
                               "S": d["seasonal_indices"][np.arange(12) % 4], "R": d["residual"]}).round(4),
                 hide_index=True, width="stretch")
    st.write(f"Seasonal indices Q1–Q4: {', '.join(f'{v:.4f}' for v in d['seasonal_indices'])}; trend per quarter = {d['trend_per_period']:.2f}.")
    st.write(f"Forecast Year 4 Q2 = **{d['forecast'](13):.4f}**, Year 5 Q3 = **{d['forecast'](18):.4f}** (matches the slides).")
