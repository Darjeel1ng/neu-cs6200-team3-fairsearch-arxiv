import plotly.express as px
import streamlit as st

from dashboard import data_loader


def render() -> None:
    st.header("Lambda Tradeoff")
    st.caption(
        "Phase 9 lambda ablation: relevance/diversity/fairness weighting vs. "
        "utility (nDCG@10, P@10, MRR) and fairness (SPD, SRR)."
    )

    df = data_loader.get_lambda_ablation()
    df = df.copy()
    df["config"] = df.apply(
        lambda r: f"rel={r['lambda_rel']} div={r['lambda_div']} fair={r['lambda_fair']}",
        axis=1,
    )

    st.dataframe(df, use_container_width=True, hide_index=True)

    metric = st.selectbox(
        "Utility metric",
        options=["nDCG@10", "P@10", "MRR"],
    )
    fairness_metric = st.selectbox(
        "Fairness metric",
        options=[
            "SPD_underrepresented",
            "SPD_privileged",
            "SRR_underrepresented",
            "SRR_privileged",
        ],
    )

    fig = px.scatter(
        df,
        x=fairness_metric,
        y=metric,
        text="config",
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
            df, x="config", y="Unique Institutions", title="Unique institutions by config"
        )
        st.plotly_chart(institutions_fig, use_container_width=True)
    with diversity_col2:
        countries_fig = px.bar(
            df, x="config", y="Unique Countries", title="Unique countries by config"
        )
        st.plotly_chart(countries_fig, use_container_width=True)
