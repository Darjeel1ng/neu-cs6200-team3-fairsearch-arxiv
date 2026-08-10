import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data_loader

# Mirrors the Fairness Metrics tab's dimension switch, against the report keys
# Phase 10 writes per citation dimension.
DIMENSIONS = {
    "Institution tier (privilege)": {
        "retrieved": "retrieved_privilege_mix",
        "cited": "cited_privilege_mix_by_variant",
        "rate": "citation_rate_by_privilege",
    },
    "Geo group (income proxy)": {
        "retrieved": "retrieved_geo_group_mix",
        "cited": "cited_geo_group_mix_by_variant",
        "rate": "citation_rate_by_geo_group",
    },
    "Region": {
        "retrieved": "retrieved_region_mix",
        "cited": "cited_region_mix_by_variant",
        "rate": "citation_rate_by_region",
    },
}

RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]


def _mix_vs_cited_bar(retrieved: dict, cited_by_variant: dict, title: str):
    groups = list(retrieved.keys())
    rows = [{"group": g, "share": retrieved[g], "source": "retrieved (available)"} for g in groups]
    for variant, mix in cited_by_variant.items():
        for g in groups:
            rows.append({"group": g, "share": mix.get(g, 0.0), "source": f"cited ({variant})"})
    fig = px.bar(pd.DataFrame(rows), x="group", y="share", color="source",
                 barmode="group", title=title)
    st.plotly_chart(fig, use_container_width=True)


def _rate_table(rate_by_variant: dict):
    df = pd.DataFrame(rate_by_variant).T
    df.index.name = "prompting_variant"
    st.dataframe(df.reset_index(), use_container_width=True, hide_index=True)
    st.caption(
        "citation_rate = cited-share / retrieved-share. 1.0 = cited in "
        "proportion to what was retrieved; >1 over-cited, <1 under-cited."
    )


def render() -> None:
    st.header("Synthesis Faithfulness (Phase 10)")

    if not data_loader.synthesis_report_exists():
        st.info("synthesis_eval_report.json not found yet — run Phase 10.")
        return

    report = data_loader.get_synthesis_report()
    st.caption(
        "LLM synthesis faithfulness (RAGAS), stance mix of retrieved docs, "
        "standard vs. perspective-balanced prompting, and citation bias vs. "
        "what was retrieved. Sample: "
        f"{report.get('sample_size', '?')} queries "
        f"({report.get('effective_sample_size', '?')} with a usable synthesis). "
        f"Generator: {report.get('generation_model', '?')}, "
        f"judge: {report.get('judge_model', '?')}."
    )

    ragas_summary = report.get("ragas_summary") or {
        "faithfulness": report.get("faithfulness_summary", {})
    }
    for metric in [m for m in RAGAS_METRICS if m in ragas_summary]:
        stats_by_variant = ragas_summary[metric]
        cols = st.columns(max(len(stats_by_variant), 1))
        for col, (variant, stats) in zip(cols, stats_by_variant.items()):
            col.metric(f"{metric.replace('_', ' ').title()} ({variant})",
                       f"{stats['mean']:.3f}",
                       help=f"median {stats['median']:.3f}, std {stats['std']:.3f}")

    c1, c2 = st.columns(2)
    c1.metric("Hallucinated citations", report.get("hallucinated_citations", 0))
    c2.metric("Failed synthesis variants", len(report.get("failed_synthesis_variants", [])))

    st.subheader("RAGAS score distributions")
    try:
        ragas = data_loader.get_ragas_scores()
        metric = st.selectbox("Metric", [m for m in RAGAS_METRICS if m in ragas.columns])
        fig = px.histogram(ragas, x=metric, color="prompting_variant",
                           barmode="overlay", nbins=12,
                           title=f"RAGAS {metric} by prompting variant")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.caption("ragas_scores.csv not available.")

    st.subheader("Stance distribution of retrieved documents")
    stance = report.get("stance_distribution", {})
    if stance:
        fig = px.bar(pd.DataFrame([{"stance": k, "share": v} for k, v in stance.items()]),
                     x="stance", y="share", title="Stance mix (pro-consensus / dissenting / neutral)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Citation bias")
    available = [name for name, cfg in DIMENSIONS.items() if cfg["retrieved"] in report]
    dim_name = st.radio("Group dimension", available, horizontal=True, key="synthesis_dim")
    cfg = DIMENSIONS[dim_name]
    _mix_vs_cited_bar(report[cfg["retrieved"]], report[cfg["cited"]],
                      f"{dim_name}: retrieved vs cited")
    if cfg["rate"] in report:
        _rate_table(report[cfg["rate"]])

    if report.get("comparability_note"):
        st.info(report["comparability_note"])

    with st.expander("Raw synthesis_eval_report.json"):
        st.json(report)
