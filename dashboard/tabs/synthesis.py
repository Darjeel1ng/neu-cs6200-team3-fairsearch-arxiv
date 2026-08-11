import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data_loader

# Mirrors the Fairness Metrics tab's dimension switch, against the report keys
# Phase 10 writes per citation dimension. "tests" points at the matching
# variant_shift_tests sub-key so each rate can be shown with its p-value.
DIMENSIONS = {
    "Institution tier (privilege)": {
        "retrieved": "retrieved_privilege_mix",
        "cited": "cited_privilege_mix_by_variant",
        "rate": "citation_rate_by_privilege",
        "tests": "privilege_label",
        "note": "QS Top-50 CS institution tier. Cited close to its retrieved "
        "share under both prompts (1.05x / 1.08x).",
    },
    "Geo group (income proxy)": {
        "retrieved": "retrieved_geo_group_mix",
        "cited": "cited_geo_group_mix_by_variant",
        "rate": "citation_rate_by_geo_group",
        "tests": "geo_group",
        "note": "World Bank high-income vs emerging economies. The "
        "data-supported bias dimension at the retrieval stage, and the "
        "largest (still non-significant) synthesis shift.",
    },
    "Region": {
        "retrieved": "retrieved_region_mix",
        "cited": "cited_region_mix_by_variant",
        "rate": "citation_rate_by_region",
        "tests": "region",
        "note": "Continental region.",
    },
}

RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]

# The report's JSON objects are keyed by variant in arbitrary order; pin the
# display order so "standard" always reads as the control column.
VARIANT_ORDER = ["standard", "balanced"]

ALPHA = 0.05


def _ordered(by_variant: dict) -> list[tuple[str, dict]]:
    known = [(v, by_variant[v]) for v in VARIANT_ORDER if v in by_variant]
    rest = [(v, d) for v, d in by_variant.items() if v not in VARIANT_ORDER]
    return known + rest


def _mix_vs_cited_bar(retrieved: dict, cited_by_variant: dict, title: str):
    groups = list(retrieved.keys())
    rows = [{"group": g, "share": retrieved[g], "source": "retrieved (available)"} for g in groups]
    for variant, mix in cited_by_variant.items():
        for g in groups:
            rows.append({"group": g, "share": mix.get(g, 0.0), "source": f"cited ({variant})"})
    fig = px.bar(pd.DataFrame(rows), x="group", y="share", color="source",
                 barmode="group", title=title)
    st.plotly_chart(fig, use_container_width=True)


def _rate_table(rate_by_variant: dict, tests: dict | None):
    """Citation rate per group, with the standard-vs-balanced significance test.

    The rates alone invite the reading that the prompt moved the citation mix.
    None of these shifts is significant, so the p-value belongs next to the
    number rather than buried in the raw JSON.
    """
    groups = list(dict.fromkeys(g for m in rate_by_variant.values() for g in m))
    rows = []
    for group in groups:
        row: dict = {"group": group}
        for variant, mix in _ordered(rate_by_variant):
            row[f"rate ({variant})"] = mix.get(group)
        test = (tests or {}).get(group)
        if test:
            row["shift (pp)"] = round(
                (test["share_balanced"] - test["share_standard"]) * 100, 2
            )
            row["p"] = round(test["p_value"], 3)
            row["significant"] = "yes" if test["p_value"] < ALPHA else "no"
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "citation_rate = cited-share / retrieved-share. 1.0 = cited in "
        "proportion to what was retrieved; >1 over-cited, <1 under-cited. "
        "`shift` is the balanced-minus-standard change in each group's share "
        "of all citations, with a two-sided two-proportion z-test."
    )

    if tests:
        if all(t["p_value"] >= ALPHA for t in tests.values()):
            st.info(
                f"No group shift on this dimension is statistically significant "
                f"(all p >= {ALPHA}). On this benchmark the prompting variant "
                "does not measurably change **which** documents get cited — "
                "see Citation volume above for what it does change."
            )
        else:
            significant = [g for g, t in tests.items() if t["p_value"] < ALPHA]
            st.warning(f"Significant shift on: {', '.join(significant)}.")


def _citation_volume(volume_by_variant: dict):
    """The result that did hold: balanced prompting widens the evidence base."""
    st.subheader("Citation volume")
    st.caption(
        "How much of the retrieved context each answer actually draws on. "
        "This is where the prompting variants differ — in breadth of coverage "
        "rather than in the group composition of what is cited."
    )

    cols = st.columns(len(volume_by_variant))
    for col, (variant, stats) in zip(cols, _ordered(volume_by_variant)):
        col.metric(
            f"Citations / query ({variant})",
            f"{stats['citations_per_query']:.2f}",
            help=(
                f"{stats['total_citations']} citations total; "
                f"{stats['unique_docs_cited_per_query']:.2f} distinct documents "
                "cited per query"
            ),
        )

    rows = [
        {"variant": variant, "measure": measure, "value": stats[key]}
        for variant, stats in _ordered(volume_by_variant)
        for measure, key in (
            ("citations / query", "citations_per_query"),
            ("distinct docs / query", "unique_docs_cited_per_query"),
        )
    ]
    fig = px.bar(pd.DataFrame(rows), x="measure", y="value", color="variant",
                 barmode="group", title="Citation breadth by prompting variant")
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.header("Synthesis Faithfulness")

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
    st.markdown(
        "**Headline result.** Perspective-balanced prompting roughly doubles "
        "how many documents an answer cites, at a small cost to faithfulness, "
        "but does **not** significantly change the institutional or geographic "
        "composition of those citations. Prompt design moves coverage, not "
        "selection."
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

    if report.get("citation_volume_by_variant"):
        _citation_volume(report["citation_volume_by_variant"])

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
    if cfg.get("note"):
        st.caption(cfg["note"])
    _mix_vs_cited_bar(report[cfg["retrieved"]], report[cfg["cited"]],
                      f"{dim_name}: retrieved vs cited")
    if cfg["rate"] in report:
        _rate_table(
            report[cfg["rate"]],
            report.get("variant_shift_tests", {}).get(cfg["tests"]),
        )

    if report.get("comparability_note"):
        st.info(report["comparability_note"])
