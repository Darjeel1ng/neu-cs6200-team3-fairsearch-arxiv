import streamlit as st

from dashboard import data_loader
from dashboard.tabs import (
    fairness_metrics,
    lambda_tradeoff,
    query_explorer,
    synthesis_placeholder,
)

st.set_page_config(page_title="FairSearch-arXiv Dashboard", layout="wide")

st.title("FairSearch-arXiv: Interactive Fairness Dashboard")
st.caption(
    "Retrieval, MMR re-ranking, and fairness metrics from the "
    "150-query bias-audit set."
)

if not data_loader.data_root_exists():
    st.error(
        f"Could not find data/update2_output under {data_loader.DATA_ROOT}. "
        "Confirm the data/ volume is mounted (see docker-compose.yml)."
    )
    st.stop()

tab_explorer, tab_fairness, tab_lambda, tab_synthesis = st.tabs(
    [
        "Query Explorer",
        "Fairness Metrics",
        "Lambda Tradeoff",
        "Synthesis Faithfulness (coming soon)",
    ]
)

with tab_explorer:
    query_explorer.render()

with tab_fairness:
    fairness_metrics.render()

with tab_lambda:
    lambda_tradeoff.render()

with tab_synthesis:
    synthesis_placeholder.render()
