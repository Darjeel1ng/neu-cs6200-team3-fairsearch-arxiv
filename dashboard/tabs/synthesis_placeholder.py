import streamlit as st


def render() -> None:
    st.header("Synthesis Faithfulness (coming soon)")
    st.info(
        "This tab will surface Phase 10 metrics once that phase lands: "
        "citation faithfulness (LLM-as-judge / RAGAS), stance classification "
        "of retrieved docs, standard vs. perspective-balanced prompting "
        "comparison, and citation rate by privilege/region.\n\n"
        "Expected inputs: `synthesis_eval_report.json`, `ragas_scores.csv`."
    )
