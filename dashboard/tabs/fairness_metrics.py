import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data_loader


def _distribution_bar(baseline: dict, observed: dict, title: str):
    keys = list(baseline.keys())
    df = pd.DataFrame(
        {
            "group": keys * 2,
            "share": [baseline[k] for k in keys] + [observed[k] for k in keys],
            "source": ["baseline"] * len(keys) + ["observed (retrieved)"] * len(keys),
        }
    )
    fig = px.bar(df, x="group", y="share", color="source", barmode="group", title=title)
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.header("Fairness Metrics")
    st.caption(
        "Statistical Parity Difference (SPD) and Selection Rate Ratio (SRR) "
        "of naive retrieval (150 queries) against the Phase 4 corpus priors."
    )

    report = data_loader.get_retrieval_parity_report()

    col1, col2 = st.columns(2)
    col1.metric("Queries evaluated", report["queries"])
    col2.metric("Documents retrieved", report["retrieved_documents"])

    st.subheader("Privilege distribution: baseline vs. observed")
    _distribution_bar(
        report["baseline_privilege"], report["observed_privilege"], "Privilege share"
    )
    spd_df = pd.DataFrame(
        [{"group": k, "SPD": v} for k, v in report["spd_privilege"].items()]
    )
    srr_df = pd.DataFrame(
        [{"group": k, "SRR": v} for k, v in report["srr_privilege"].items()]
    )
    c1, c2 = st.columns(2)
    c1.dataframe(spd_df, use_container_width=True, hide_index=True)
    c2.dataframe(srr_df, use_container_width=True, hide_index=True)

    st.subheader("Region distribution: baseline vs. observed")
    _distribution_bar(
        report["baseline_region"], report["observed_region"], "Region share"
    )

    st.subheader("SPD / SRR by query category")
    rows = []
    for category, stats in report["category_breakdown"].items():
        rows.append(
            {
                "category": category,
                "n_retrieved": stats["n_retrieved"],
                "spd_underrepresented": stats["spd"]["underrepresented"],
                "spd_privileged": stats["spd"]["privileged"],
                "srr_underrepresented": stats["srr"]["underrepresented"],
                "srr_privileged": stats["srr"]["privileged"],
            }
        )
    category_df = pd.DataFrame(rows).sort_values("category")
    st.dataframe(category_df, use_container_width=True, hide_index=True)

    with st.expander("Corpus-level fairness priors (Phase 4)"):
        st.json(data_loader.get_fairness_baseline_priors())
