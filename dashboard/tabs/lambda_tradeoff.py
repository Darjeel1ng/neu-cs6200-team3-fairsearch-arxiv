import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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


def _dedupe_fairness_options(df: pd.DataFrame, candidates: list[str]):
    """Drop fairness metrics that plot an identical distance-from-parity line.

    A two-group SPD is symmetric -- SPD_privileged == -SPD_underrepresented
    exactly -- so both give the same |distance from parity| and the selector
    offered two entries that drew the same chart. Computed rather than
    hard-coded so it stays correct if new group columns are added.

    Note this only holds for the absolute view. The earlier indexed version
    divided by the baseline gap, which cancelled each group's corpus share and
    made the SRR variants collapse onto the SPD ones too; those are genuinely
    distinct here and are kept.
    """
    seen: dict[tuple, str] = {}
    options: list[str] = []
    aliases: dict[str, list[str]] = {}
    for metric in candidates:
        signature = tuple(round(_parity_gap(metric, v), 12) for v in df[metric])
        if signature in seen:
            aliases.setdefault(seen[signature], []).append(metric)
        else:
            seen[signature] = metric
            options.append(metric)
    return options, aliases


def _pick_comparison_row(df: pd.DataFrame) -> pd.Series:
    """Which config the summary tiles compare against the baseline.

    This used to be `idxmax(lambda_fair)`, which was unique only while the
    sweep varied diversity and fairness together. With the RQ3 isolation arms
    added, a fairness-only config ties the joint config at lambda_fair=0.3 and
    idxmax silently returned whichever came first -- headlining the joint
    config, the one whose effect is *not* attributable to a single mechanism.
    Let the reader choose instead, defaulting to the strongest fairness-only
    arm when one exists, since that is the attributable result.
    """
    candidates = df[df["config_type"] != "baseline"]
    if candidates.empty:
        candidates = df

    isolated = candidates[candidates["config_type"] == "fairness_only"]
    preferred = (isolated if not isolated.empty else candidates)
    default = preferred.loc[preferred["lambda_fair"].idxmax(), "config"]

    options = list(candidates["config"])
    choice = st.selectbox(
        "Compare against baseline", options=options,
        index=options.index(default),
        help="Configs sharing a lambda_fair value are distinguished by their "
             "full weight triple; the default is the strongest fairness-only "
             "arm, whose effect is attributable to the fairness term alone.",
    )
    return candidates[candidates["config"] == choice].iloc[0]


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

    # Utility is indexed to the baseline: the metrics have different natural
    # scales (Recall@10 0.90 vs nDCG@10 0.83) and the point is how little each
    # one moves, which a shared percentage axis shows directly.
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for col in [c for c in UTILITY_COLS if c in df.columns and base[c]]:
        fig.add_trace(
            go.Scatter(x=df["config"], y=df[col] / base[col] * 100,
                       name=col, mode="lines+markers"),
            secondary_y=False,
        )

    # Fairness is NOT indexed. Dividing by the baseline gap is meaningless when
    # that gap is already ~0: on the institution tier the baseline sits at
    # parity, so the ratio ran to 2175% and flattened every utility line into
    # the axis. Absolute distance from parity on its own axis keeps both
    # readable and keeps "0 = parity" interpretable.
    fig.add_trace(
        go.Scatter(x=df["config"],
                   y=[_parity_gap(fairness_metric, v) for v in df[fairness_metric]],
                   name=f"|{fairness_metric} − parity|", mode="lines+markers",
                   line=dict(dash="dash")),
        secondary_y=True,
    )

    fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_yaxes(title_text="utility, % of baseline", secondary_y=False)
    fig.update_yaxes(title_text="distance from parity (0 = parity)",
                     rangemode="tozero", secondary_y=True)
    fig.update_layout(
        title="Utility (left, % of baseline) vs. fairness gap (right, absolute)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=90),
    )
    fig.add_hline(y=100, line_dash="dot", opacity=0.4, secondary_y=False)
    st.plotly_chart(fig, use_container_width=True)

    # The chart above is absolute, so it renders fine at any baseline. The
    # tiles below are still expressed as a percentage change against the
    # baseline gap, which is undefined when that gap is exactly zero.
    base_gap = _parity_gap(fairness_metric, base[fairness_metric])
    if not base_gap:
        st.caption(
            f"{fairness_metric} is already exactly at parity in the baseline, "
            "so the change cannot be expressed as a percentage of it. The chart "
            "above shows the absolute distance from parity."
        )
        return

    strongest = _pick_comparison_row(df)
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

    fairness_options, fairness_aliases = _dedupe_fairness_options(df, fairness_options)

    # Default to the geo dimension: the institution tier starts near parity, so
    # it is the one dimension where raising lambda_fair over-corrects rather
    # than helps, which makes it a misleading first impression.
    default_fairness = (
        fairness_options.index("SPD_high_resource")
        if "SPD_high_resource" in fairness_options
        else 0
    )
    fairness_metric = st.selectbox(
        "Fairness metric", options=fairness_options, index=default_fairness,
        format_func=lambda m: (
            f"{m}  (identical to {', '.join(fairness_aliases[m])})"
            if m in fairness_aliases else m
        ),
    )
    if fairness_aliases:
        merged = "; ".join(
            f"{kept} = {', '.join(dropped)}" for kept, dropped in fairness_aliases.items()
        )
        st.caption(
            f"Some metrics are omitted from this list because they plot the "
            f"identical line: {merged}. A two-group SPD is symmetric, so both "
            "groups are the same distance from parity in opposite directions."
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
