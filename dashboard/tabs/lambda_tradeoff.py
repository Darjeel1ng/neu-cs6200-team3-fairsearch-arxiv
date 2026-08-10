import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data_loader

# Utility columns, most-headline-first. Recall@10 is the known-item hit rate
# and is flat across every config; nDCG@10 and MRR are rank-position signals.
UTILITY_COLS = ["Recall@10", "nDCG@10", "MRR"]

# The two fairness families have different parity points: SPD is a difference
# of proportions, so parity is 0, while SRR is a ratio, so parity is 1.0.
# Distance-from-parity has to account for that -- |SRR| would treat 0.79 as a
# smaller deviation than 1.03, which is backwards.
PARITY_POINT = {"SPD": 0.0, "SRR": 1.0}


def _parity_gap(metric: str, value: float) -> float:
    return abs(value - PARITY_POINT.get(metric.split("_")[0], 0.0))


def _indexed_tradeoff(df: pd.DataFrame, fairness_metric: str) -> None:
    """Utility and fairness on one axis, each as a percentage of the baseline.

    The raw scatter below plots utility against fairness directly, but utility
    barely moves (Recall@10 is literally constant), so the points stack into a
    near-vertical line that reads as a broken chart rather than as the result.
    Indexing every series to baseline = 100% puts "fairness moves a lot, utility
    barely moves" on a single readable axis.
    """
    baseline = df[df["config_type"] == "baseline"]
    if baseline.empty:
        st.info(
            "No baseline row in lambda_ablation.csv, so there is nothing to "
            "index against. Showing the raw comparison below instead."
        )
        return
    base = baseline.iloc[0]

    order = list(df["config"])
    rows = []
    for col in [c for c in UTILITY_COLS if c in df.columns]:
        if not base[col]:
            continue
        for _, r in df.iterrows():
            rows.append({"config": r["config"], "series": col,
                         "pct": r[col] / base[col] * 100, "kind": "utility"})

    base_gap = _parity_gap(fairness_metric, base[fairness_metric])
    gap_label = f"{fairness_metric} distance from parity"
    if base_gap:
        for _, r in df.iterrows():
            rows.append({"config": r["config"], "series": gap_label,
                         "pct": _parity_gap(fairness_metric, r[fairness_metric]) / base_gap * 100,
                         "kind": "fairness"})

    fig = px.line(
        pd.DataFrame(rows), x="config", y="pct", color="series",
        line_dash="kind", markers=True,
        title="Utility vs. fairness, indexed to the naive baseline (= 100%)",
        labels={"pct": "% of baseline", "config": ""},
    )
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.add_hline(y=100, line_dash="dot", opacity=0.4)
    st.plotly_chart(fig, use_container_width=True)

    if not base_gap:
        st.caption(
            f"{fairness_metric} is already exactly at parity in the baseline, "
            "so the gap cannot be expressed as a percentage of it; utility is "
            "still shown."
        )
        return

    strongest = df.loc[df["lambda_fair"].idxmax()]
    gap_now = _parity_gap(fairness_metric, strongest[fairness_metric])
    gap_change = (gap_now - base_gap) / base_gap * 100
    ndcg_change = (
        (strongest["nDCG@10"] - base["nDCG@10"]) / base["nDCG@10"] * 100
        if "nDCG@10" in df.columns and base["nDCG@10"] else 0.0
    )

    parity = PARITY_POINT.get(fairness_metric.split("_")[0], 0.0)
    # Name the full weight triple, not just lambda_fair. Several configs can
    # share a lambda_fair value -- a diversity-only and a fairness-only arm at
    # the same weight would tie, and idxmax picks one of them silently -- so
    # "fair=0.3" alone would not say which row is on screen.
    strongest_label = strongest.get("config", f"fair={strongest['lambda_fair']}")
    c1, c2 = st.columns(2)
    c1.metric(
        f"Distance from parity ({strongest_label})",
        f"{gap_now:.4f}",
        f"{gap_change:+.0f}% vs baseline",
        delta_color="inverse",
        help=(
            f"{fairness_metric} moved from {base[fairness_metric]:.4f} at the "
            f"baseline to {strongest[fairness_metric]:.4f} at {strongest_label}; "
            f"parity for this metric is {parity:.1f}, so the distance from "
            f"parity went {base_gap:.4f} -> {gap_now:.4f}."
        ),
    )
    c2.metric(
        "nDCG@10 cost", f"{strongest['nDCG@10']:.4f}", f"{ndcg_change:+.1f}%",
        help=f"Recall@10 stays at {strongest.get('Recall@10', float('nan')):.2f} "
             "across every configuration.",
    )

    if gap_change < 0:
        st.success(
            f"At {strongest_label}, reranking closes {abs(gap_change):.0f}% of "
            f"the {fairness_metric} gap for {abs(ndcg_change):.1f}% of nDCG@10, "
            "with Recall@10 unchanged — fairness here is close to free."
        )
    else:
        st.warning(
            f"At {strongest_label} the distance from parity **grows** "
            f"{gap_change:+.0f}%. It already starts near parity, so raising the "
            "fairness weight over-corrects and pushes the retrieved mix further "
            "from the corpus distribution. Growing distance from parity means "
            "departure from the corpus mix, not increasing unfairness."
        )


def render() -> None:
    st.header("Lambda Tradeoff")
    st.caption(
        "Phase 9 lambda ablation: relevance/diversity/fairness weighting vs. "
        "utility (Recall@10 known-item, nDCG@10, MRR) and fairness (SPD/SRR "
        "on institution tier and geo group). Includes joint MMR plus "
        "diversity-only and fairness-only configs for RQ3 mechanism isolation; "
        "the naive baseline (no MMR) is shown as a reference point."
    )

    df = data_loader.get_lambda_ablation()
    df = df.copy()

    if "config_type" not in df.columns:
        df["config_type"] = "mmr"
    df["config"] = df.apply(
        lambda r: (
            "naive baseline"
            if r["config_type"] == "baseline"
            else f"rel={r['lambda_rel']} div={r['lambda_div']} fair={r['lambda_fair']}"
        ),
        axis=1,
    )

    utility_options = [c for c in ["Recall@10", "nDCG@10", "P@10", "MRR"] if c in df.columns]
    fairness_options = [
        c
        for c in [
            "SPD_underrepresented",
            "SPD_privileged",
            "SPD_high_resource",
            "SPD_emerging",
            "SRR_underrepresented",
            "SRR_privileged",
            "SRR_high_resource",
            "SRR_emerging",
        ]
        if c in df.columns
    ]

    # Default to the geo dimension: the institution tier starts near parity, so
    # it is the one dimension where raising lambda_fair over-corrects rather
    # than helps, which makes it a misleading first impression.
    default_fairness = (
        fairness_options.index("SPD_high_resource")
        if "SPD_high_resource" in fairness_options
        else 0
    )
    fairness_metric = st.selectbox(
        "Fairness metric", options=fairness_options, index=default_fairness
    )

    _indexed_tradeoff(df, fairness_metric)

    st.subheader("All configurations")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Utility vs. fairness, unindexed")
    st.caption(
        "The same data plotted directly. Utility spans a very narrow range, so "
        "the points cluster along one axis — that flatness is the result, not a "
        "rendering problem."
    )
    metric = st.selectbox("Utility metric", options=utility_options)

    fig = px.scatter(
        df,
        x=fairness_metric,
        y=metric,
        text="config",
        color="config_type",
        symbol="config_type",
        title=f"{metric} vs. {fairness_metric} across lambda configs",
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Diversity by config")
    st.caption(
        "Grouped by config rather than plotted against lambda_fair alone: "
        "two configs share lambda_fair=0.1 and two share lambda_fair=0.2 "
        "while lambda_rel/lambda_div still differ between them, so a single "
        "fairness-weight axis can't represent these points as a function."
    )
    diversity_col1, diversity_col2 = st.columns(2)
    with diversity_col1:
        institutions_fig = px.bar(
            df, x="config", y="Unique Institutions", color="config_type",
            title="Unique institutions by config"
        )
        st.plotly_chart(institutions_fig, use_container_width=True)
    with diversity_col2:
        countries_fig = px.bar(
            df, x="config", y="Unique Countries", color="config_type",
            title="Unique countries by config"
        )
        st.plotly_chart(countries_fig, use_container_width=True)
