import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data_loader

# Each dimension maps to the report keys produced by Phase 8.
DIMENSIONS = {
    "Institution tier (privilege)": {
        "baseline": "baseline_privilege",
        "observed": "observed_privilege",
        "spd": "spd_privilege",
        "srr": "srr_privilege",
        "note": "QS Top-50 CS institution tier. Near corpus-balanced under "
        "retrieval (SPD_privileged ~ -0.006).",
    },
    "Geo group (income proxy)": {
        "baseline": "baseline_geo_group",
        "observed": "observed_geo_group",
        "spd": "spd_geo_group",
        "srr": "srr_geo_group",
        "note": "World Bank high-income vs emerging economies. This is the "
        "data-supported bias dimension (high_resource is over-retrieved).",
    },
    "Region": {
        "baseline": "baseline_region",
        "observed": "observed_region",
        "spd": "spd_region",
        "srr": "srr_region",
        "note": "Continental region. Europe is mildly over-retrieved, Asia "
        "mildly under-retrieved.",
    },
}


def _distribution_bar(baseline: dict, observed: dict, title: str):
    keys = list(baseline.keys())
    df = pd.DataFrame(
        {
            "group": keys * 2,
            "share": [baseline[k] for k in keys] + [observed.get(k, 0.0) for k in keys],
            "source": ["baseline"] * len(keys) + ["observed (retrieved)"] * len(keys),
        }
    )
    fig = px.bar(df, x="group", y="share", color="source", barmode="group", title=title)
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.header("Fairness Metrics")
    st.caption(
        "Statistical Parity Difference (SPD) and Selection Rate Ratio (SRR) "
        "of naive retrieval (150 queries) against the Phase 4 corpus priors. "
        "The audit runs on three group dimensions: institution tier, geo group, "
        "and region."
    )

    report = data_loader.get_retrieval_parity_report()

    col1, col2 = st.columns(2)
    col1.metric("Queries evaluated", report["queries"])
    col2.metric("Documents retrieved", report["retrieved_documents"])

    available = [
        name for name, cfg in DIMENSIONS.items() if cfg["baseline"] in report
    ]
    dim_name = st.radio("Group dimension", available, horizontal=True)
    cfg = DIMENSIONS[dim_name]
    st.caption(cfg["note"])

    baseline = report[cfg["baseline"]]
    observed = report[cfg["observed"]]
    spd = report[cfg["spd"]]
    srr = report[cfg["srr"]]

    st.subheader(f"{dim_name}: baseline vs. observed")
    _distribution_bar(baseline, observed, f"{dim_name} share")

    spd_df = pd.DataFrame([{"group": k, "SPD": v} for k, v in spd.items()])
    srr_df = pd.DataFrame([{"group": k, "SRR": v} for k, v in srr.items()])
    c1, c2 = st.columns(2)
    c1.dataframe(spd_df, use_container_width=True, hide_index=True)
    c2.dataframe(srr_df, use_container_width=True, hide_index=True)

    # Equal opportunity is only computed for the institution-tier dimension.
    # Reported in the paper (Methodology / RQ1) but previously not surfaced here.
    if cfg["spd"] == "spd_privilege" and "group_tpr" in report:
        st.subheader("Equal opportunity (institution tier)")
        st.caption(
            "Share of known-item queries whose ground-truth document is "
            "retrieved, per group. This is the TPR / equal-opportunity "
            "component only: the benchmark has one relevant document per query "
            "and no exhaustive negative judgements, so a full TPR/FPR "
            "Equalized Odds decomposition is not computable."
        )
        # Drop empty label buckets -- privilege_label has no unknowns in this
        # corpus, so the "null" group carries 0 queries and would otherwise
        # render as a 0.0 TPR, reading like a total failure for a real group.
        tpr = {g: v for g, v in report["group_tpr"].items()
               if g in (baseline or {}) and g != "null"}
        dropped = [g for g in report["group_tpr"] if g not in tpr]
        cols = st.columns(len(tpr) + 1)
        for col, (group, value) in zip(cols, tpr.items()):
            col.metric(f"TPR ({group})", f"{value:.4f}")
        if "gap" in report:
            cols[-1].metric("Gap", f"{report['gap']:.4f}",
                            help="Difference in TPR between the two groups; "
                                 "0 means equal opportunity.")
        if dropped:
            st.caption(f"Empty label bucket(s) omitted: {', '.join(dropped)}.")

    # Category breakdown is only computed for the institution-tier dimension.
    if cfg["spd"] == "spd_privilege" and "category_breakdown" in report:
        st.subheader("SPD / SRR by query category (institution tier)")
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

    # The raw Phase 4 priors JSON used to be dumped in an expander here. The
    # same numbers are already on screen as the "baseline" series of the
    # distribution chart above, which is the readable form of them.
