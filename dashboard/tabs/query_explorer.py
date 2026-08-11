import os

import pandas as pd
import streamlit as st

from dashboard import data_loader, live_retrieval

# geo_group belongs here alongside privilege_label: the audit runs on both
# dimensions, and geo_group is the one with a measurable retrieval gap
# (emerging SRR 0.857) while the institution tier sits near parity. Without it
# this table cannot show the dimension the findings actually rest on.
DOC_COLUMNS = [
    "rank",
    "document_id",
    "title",
    "score",
    "privilege_label",
    "geo_group",
    "region",
    "country_code",
    "institution",
]


def _docs_for_query(df: pd.DataFrame, query_id: int) -> pd.DataFrame:
    subset = df[df["query_id"] == query_id].sort_values("rank")
    cols = [c for c in DOC_COLUMNS if c in subset.columns]
    return subset[cols].reset_index(drop=True)


def render() -> None:
    st.header("Query Explorer")
    st.caption(
        "Compare naive retrieval against every Phase 9 MMR lambda config for "
        "a single query."
    )

    if live_retrieval.is_available():
        query_text = st.text_input(
            "Free-text query (live retrieval against chroma_db)", ""
        )
        if query_text:
            with st.spinner("Retrieving..."):
                results = live_retrieval.run_query(query_text)
            st.subheader("Live retrieval results")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            return
        st.info("Live retrieval is active — enter a query above, or pick a precomputed one below.")
    elif os.path.isdir(live_retrieval.CHROMA_DB_PATH):
        # chroma_db is present but the index would not open -- that is a real
        # failure and worth surfacing. Its plain absence is not: replaying the
        # committed 150 queries is the intended default path, and announcing it
        # in a blue banner on the first tab reads like something is broken.
        st.warning(live_retrieval.unavailable_reason())

    queries = data_loader.get_queries()
    query_labels = {
        q["query_id"]: f"[{q['query_id']}] ({q['query_type']}) {q['query'][:80]}"
        for q in queries
    }
    selected_id = st.selectbox(
        "Precomputed query",
        options=list(query_labels.keys()),
        format_func=lambda qid: query_labels[qid],
    )
    selected_query = next(q for q in queries if q["query_id"] == selected_id)
    st.markdown(f"**Full query:** {selected_query['query']}")
    st.markdown(
        f"**Category:** {selected_query['primary_category']} &nbsp;|&nbsp; "
        f"**Query paper privilege:** {selected_query['privilege_label']}"
    )

    naive_df = data_loader.get_naive_results()
    mmr_results = data_loader.get_mmr_results()

    naive_docs = _docs_for_query(naive_df, selected_id)
    st.subheader(f"Naive baseline (top-{len(naive_docs)})")
    st.caption(
        "This is the Phase 6 over-fetch (fetch_k=20) before MMR re-ranks "
        "down to top-10 — not the final naive result set."
    )
    st.dataframe(naive_docs, use_container_width=True)

    st.subheader("MMR re-ranked configs (top-10 each)")
    config_tabs = st.tabs(list(mmr_results.keys()))
    for tab, (config_name, config_df) in zip(config_tabs, mmr_results.items()):
        with tab:
            st.dataframe(_docs_for_query(config_df, selected_id), use_container_width=True)
