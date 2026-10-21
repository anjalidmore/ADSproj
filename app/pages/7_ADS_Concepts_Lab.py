import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/common.py importable

import pandas as pd
import streamlit as st

from common import concept_entries, csv_button, note, page

page("ADS Concepts Lab", "Every syllabus concept applied to the real dataset: formula · variables · computed result · interpretation · validation · code")

entries = concept_entries()
modules = list(dict.fromkeys(e["module"] for e in entries))
c1, c2 = st.columns([2, 3])
pick = c1.multiselect("Modules", modules, default=modules)
search = c2.text_input("Search concepts", placeholder="e.g. kurtosis, ANOVA, LOF")

shown = [e for e in entries if e["module"] in pick and (not search or search.lower() in (e["concept"] + e["module"]).lower())]
st.caption(f"{len(shown)} of {len(entries)} concepts. Values are computed when the app starts; the same registry generates docs/CONCEPT_MATRIX.md.")

for mod in [m for m in modules if m in pick]:
    group = [e for e in shown if e["module"] == mod]
    if not group:
        continue
    st.header(mod)
    for e in group:
        with st.expander(e["concept"], expanded=False):
            st.latex(e["formula"])
            st.markdown(f"**Variables / data:** {e['variables']}")
            if e["objective"]:
                st.markdown(f"**Why:** {e['objective']}")
            out = pd.DataFrame(list(e["output"].items()), columns=["computed output", "value"])
            st.dataframe(out, hide_index=True, width="stretch")
            st.markdown(f"**Interpretation:** {e['interpretation']}")
            st.markdown(f"**Validation:** {e['validation']}")
            st.markdown(f"**Code:** `{e['code']}`")

flat = pd.DataFrame([{**{k: v for k, v in e.items() if k != "output"},
                      "output": "; ".join(f"{k}: {v}" for k, v in e["output"].items())} for e in entries])
csv_button(flat, "concept_matrix.csv", "Download the full concept matrix (CSV)")
note("Concepts that do not fit this dataset are stated as such rather than forced: seasonal indices (annual data) and the "
     "Poisson model (the counts are over-dispersed). Their formulas are still implemented and verified on the course examples.")
