"""Regenerate the dual-label (institution + geo) fairness artifacts.

This script implements the plan's "re-join labels by document_id" path: it does
NOT re-ingest ChromaDB or call any LLM. It:

  1. adds `geo_group` to data/final_50k_labeled.parquet (Phase 3 extension),
  2. refreshes fairness_baseline_priors.json + geo label distribution (Phase 4),
  3. recomputes retrieval_parity_report.json with the geo dimension (Phase 8),
  4. rebuilds lambda_ablation.csv: P@10 -> Recall@10 (known-item HitRate) plus
     geo SPD/SRR columns and a naive-baseline row (Phase 9),
  5. backfills citation-rate-by-group into synthesis_eval_report.json for
     pre-DeepSeek reports that lack it (Phase 10 now emits it directly),
  6. adds known-item difficulty (Easy/Medium/Hard) and SPD classification
     (Biased/Fair/Over-corrected) to query_benchmark_report.json (Phase 7),
  7. renders the geo figures under data/{figures,update2_output/figures}.

Utility metrics (Recall@10 / nDCG@10 / MRR) are known-item and are averaged over
the 100 neutral queries that carry a ground-truth document. Fairness metrics are
averaged over all 150 queries.
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geo_labels import GEO_GROUP_ORDER, geo_group

DATA = "data"
U2 = os.path.join(DATA, "update2_output")
FIG4 = os.path.join(DATA, "figures")
FIG = os.path.join(U2, "figures")
LABELED = os.path.join(DATA, "final_50k_labeled.parquet")

os.makedirs(FIG, exist_ok=True)
os.makedirs(FIG4, exist_ok=True)


def _dist(series) -> dict:
    vc = series.value_counts(dropna=False)
    total = vc.sum()
    return {str(k): float(v) / total for k, v in vc.items()}


def _repr_vs_uniform(prior: dict) -> dict:
    g = len(prior)
    uniform = 1.0 / g if g else 0.0
    return {k: (v / uniform if uniform else 0.0) for k, v in prior.items()}


# ---------------------------------------------------------------------------
# 1. Phase 3 extension: add geo_group to the labeled corpus
# ---------------------------------------------------------------------------
print("[1/7] Adding geo_group to labeled corpus ...")
df = pd.read_parquet(LABELED)
df["geo_group"] = df["country_code"].map(geo_group)
df.to_parquet(LABELED, index=False)
geo_counts = df["geo_group"].value_counts()
print("      geo_group counts:", geo_counts.to_dict())

# ---------------------------------------------------------------------------
# 2. Phase 4 extension: priors + geo distribution
# ---------------------------------------------------------------------------
print("[2/7] Refreshing baseline priors + geo distribution ...")
priors_path = os.path.join(DATA, "fairness_baseline_priors.json")
with open(priors_path, encoding="utf-8") as f:
    priors = json.load(f)

geo_prior = _dist(df["geo_group"])
priors["geo_group_prior"] = geo_prior
priors["geo_group_representation_vs_uniform"] = _repr_vs_uniform(geo_prior)
with open(priors_path, "w", encoding="utf-8") as f:
    json.dump(priors, f, indent=2)

geo_dist_rows = [
    {"geo_group": k, "paper_count": int(v), "ratio": round(v / len(df), 5)}
    for k, v in df["geo_group"].value_counts().items()
]
pd.DataFrame(geo_dist_rows).to_csv(
    os.path.join(DATA, "geo_label_distribution.csv"), index=False
)
print("      geo_group prior:", {k: round(v, 4) for k, v in geo_prior.items()})

# ---------------------------------------------------------------------------
# 3. Phase 8 extension: parity report geo dimension
# ---------------------------------------------------------------------------
print("[3/7] Recomputing retrieval parity report (geo) ...")
with open(os.path.join(U2, "naive_retrieval_results.json"), encoding="utf-8") as f:
    naive = pd.DataFrame(json.load(f))
naive["geo_group"] = naive["country_code"].map(geo_group)

parity_path = os.path.join(U2, "retrieval_parity_report.json")
with open(parity_path, encoding="utf-8") as f:
    parity = json.load(f)

observed_geo = _dist(naive["geo_group"])
baseline_geo = geo_prior
spd_geo = {g: round(observed_geo.get(g, 0.0) - baseline_geo.get(g, 0.0), 6) for g in baseline_geo}
srr_geo = {
    g: round(observed_geo.get(g, 0.0) / baseline_geo[g], 6) if baseline_geo[g] else None
    for g in baseline_geo
}
parity["baseline_geo_group"] = baseline_geo
parity["observed_geo_group"] = observed_geo
parity["spd_geo_group"] = spd_geo
parity["srr_geo_group"] = srr_geo
with open(parity_path, "w", encoding="utf-8") as f:
    json.dump(parity, f, indent=2)
print("      geo SPD:", spd_geo)

# ---------------------------------------------------------------------------
# 4. Phase 9 extension: lambda ablation (Recall@10 + geo SPD/SRR + baseline)
# ---------------------------------------------------------------------------
print("[4/7] Rebuilding lambda ablation ...")
with open(os.path.join(U2, "queries.json"), encoding="utf-8") as f:
    queries = json.load(f)
# known-item ground truth: only the neutral queries carry a document_id
ground_truth = {
    q["query_id"]: q["document_id"]
    for q in queries
    if q.get("document_id")
}
known_item_ids = set(ground_truth)
print(f"      known-item queries: {len(known_item_ids)} (of {len(queries)})")

priv_prior = priors["privilege_prior"]
geo_prior_full = priors["geo_group_prior"]


def _per_query_metrics(group10: pd.DataFrame):
    """Per-query fairness (SPD/SRR privilege+geo) + diversity, over its <=10 docs."""
    obs_priv = group10["privilege_label"].value_counts(normalize=True).to_dict()
    obs_geo = group10["geo_group"].value_counts(normalize=True).to_dict()
    out = {}
    for g, exp in priv_prior.items():
        o = obs_priv.get(g, 0.0)
        out[f"spd_priv_{g}"] = o - exp
        out[f"srr_priv_{g}"] = (o / exp) if exp else 0.0
    for g, exp in geo_prior_full.items():
        o = obs_geo.get(g, 0.0)
        out[f"spd_geo_{g}"] = o - exp
        out[f"srr_geo_{g}"] = (o / exp) if exp else 0.0
    out["uniq_inst"] = group10["institution"].replace("", np.nan).dropna().nunique()
    out["uniq_country"] = group10["country_code"].nunique()
    return out


def _summarize(config_label, per_config_df, lam=(None, None, None), config_type="mmr"):
    per_config_df = per_config_df.copy()
    per_config_df["geo_group"] = per_config_df["country_code"].map(geo_group)
    recall, ndcg, mrr = [], [], []
    fair_rows = []
    for qid, grp in per_config_df.groupby("query_id"):
        top10 = grp.sort_values("rank").head(10)
        fair_rows.append(_per_query_metrics(top10))
        if qid in known_item_ids:
            gt = ground_truth[qid]
            hit = top10[top10["document_id"] == gt]
            recall.append(1.0 if len(hit) else 0.0)
            if len(hit):
                r = int(hit.iloc[0]["rank"])
                ndcg.append(1.0 / math.log2(r + 1))
                mrr.append(1.0 / r)
            else:
                ndcg.append(0.0)
                mrr.append(0.0)
    fair = pd.DataFrame(fair_rows).mean()
    return {
        "lambda_rel": lam[0],
        "lambda_div": lam[1],
        "lambda_fair": lam[2],
        "config_type": config_type,
        "nDCG@10": float(np.mean(ndcg)),
        "Recall@10": float(np.mean(recall)),
        "MRR": float(np.mean(mrr)),
        "SPD_underrepresented": fair["spd_priv_underrepresented"],
        "SPD_privileged": fair["spd_priv_privileged"],
        "SRR_underrepresented": fair["srr_priv_underrepresented"],
        "SRR_privileged": fair["srr_priv_privileged"],
        "SPD_high_resource": fair["spd_geo_high_resource"],
        "SPD_emerging": fair["spd_geo_emerging"],
        "SRR_high_resource": fair["srr_geo_high_resource"],
        "SRR_emerging": fair["srr_geo_emerging"],
        "Unique Institutions": fair["uniq_inst"],
        "Unique Countries": fair["uniq_country"],
    }


with open(os.path.join(U2, "mmr_reranked_results.json"), encoding="utf-8") as f:
    mmr = json.load(f)

rows = []
# naive baseline = pure relevance top-10 (rel=1.0)
rows.append(_summarize("naive", naive, lam=(1.0, 0.0, 0.0), config_type="baseline"))
for key, records in mmr.items():
    parts = key.split("_")  # rel_0.80_div_0.10_fair_0.10
    lam = (float(parts[1]), float(parts[3]), float(parts[5]))
    rows.append(_summarize(key, pd.DataFrame(records), lam=lam, config_type="mmr"))

lambda_df = pd.DataFrame(rows)
lambda_df.to_csv(os.path.join(U2, "lambda_ablation.csv"), index=False)
print(lambda_df[["config_type", "lambda_rel", "Recall@10", "nDCG@10",
                 "SPD_privileged", "SPD_high_resource"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Phase 10 extension: citation rate by group (privilege + region)
# ---------------------------------------------------------------------------
print("[5/7] Backfilling citation-rate-by-group into synthesis report ...")
synth_path = os.path.join(U2, "synthesis_eval_report.json")
with open(synth_path, encoding="utf-8") as f:
    synth = json.load(f)


def _citation_rate(cited: dict, retrieved: dict) -> dict:
    """cited share / retrieved share; >1 = over-cited relative to availability."""
    return {
        k: round(cited.get(k, 0.0) / retrieved[k], 4) if retrieved.get(k) else None
        for k in retrieved
    }


# Backfill only. Phase 10 now emits citation_rate_by_* itself (including the
# geo dimension, which needs per-citation country codes this aggregate doesn't
# carry), so recomputing here would just overwrite fresher numbers with
# whatever the report already holds.
CITATION_RATE_DIMS = {
    "citation_rate_by_privilege": ("retrieved_privilege_mix", "cited_privilege_mix_by_variant"),
    "citation_rate_by_region": ("retrieved_region_mix", "cited_region_mix_by_variant"),
    "citation_rate_by_geo_group": ("retrieved_geo_group_mix", "cited_geo_group_mix_by_variant"),
}

backfilled = []
for rate_key, (retrieved_key, cited_key) in CITATION_RATE_DIMS.items():
    if rate_key in synth or retrieved_key not in synth:
        continue
    synth[rate_key] = {
        v: _citation_rate(synth[cited_key][v], synth[retrieved_key])
        for v in synth[cited_key]
    }
    backfilled.append(rate_key)

synth.setdefault(
    "citation_rate_note",
    "citation_rate = cited-share / retrieved-share per group and prompting "
    "variant. 1.0 = cited in proportion to retrieval; >1 over-cited, <1 "
    "under-cited.",
)
with open(synth_path, "w", encoding="utf-8") as f:
    json.dump(synth, f, indent=2)
print("      backfilled:", backfilled or "nothing (Phase 10 report already complete)")

# ---------------------------------------------------------------------------
# 6. Phase 7 extension: query difficulty + SPD classification
# ---------------------------------------------------------------------------
print("[6/7] Classifying query difficulty + SPD ...")
qbr_path = os.path.join(U2, "query_benchmark_report.json")
with open(qbr_path, encoding="utf-8") as f:
    qbr = json.load(f)

difficulty = {"Easy": 0, "Medium": 0, "Hard": 0}
spd_class = {"Biased": 0, "Fair": 0, "Over-corrected": 0}
per_query = {}
for qid, grp in naive.groupby("query_id"):
    top10 = grp.sort_values("rank").head(10)
    # difficulty (known-item only)
    if qid in known_item_ids:
        gt = ground_truth[qid]
        hit_full = grp[grp["document_id"] == gt]
        if len(hit_full):
            r = int(hit_full.iloc[0]["rank"])
            bucket = "Easy" if r <= 3 else ("Medium" if r <= 10 else "Hard")
        else:
            bucket = "Hard"
        difficulty[bucket] += 1
    else:
        bucket = None
    # SPD classification on privilege (per-query top-10)
    obs = top10["privilege_label"].value_counts(normalize=True).to_dict()
    spd_priv = obs.get("privileged", 0.0) - priv_prior["privileged"]
    if spd_priv > 0.05:
        sclass = "Biased"          # privileged over-represented
    elif spd_priv < -0.05:
        sclass = "Over-corrected"  # privileged strongly under-represented
    else:
        sclass = "Fair"
    spd_class[sclass] += 1
    per_query[str(qid)] = {"difficulty": bucket, "spd_privileged": round(spd_priv, 4),
                           "spd_class": sclass}

qbr["difficulty_thresholds"] = {"Easy": "gt rank <= 3", "Medium": "gt rank 4-10",
                                "Hard": "gt not in top-10 (known-item only)"}
qbr["difficulty_distribution"] = difficulty
qbr["spd_classification_thresholds"] = {
    "Biased": "SPD_privileged > 0.05", "Fair": "|SPD_privileged| <= 0.05",
    "Over-corrected": "SPD_privileged < -0.05"}
qbr["spd_classification_distribution"] = spd_class
qbr["per_query_classification"] = per_query
with open(qbr_path, "w", encoding="utf-8") as f:
    json.dump(qbr, f, indent=2)
print("      difficulty:", difficulty, "| spd_class:", spd_class)

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
print("[7/7] Rendering figures ...")
plt.rcParams.update({"figure.dpi": 110})

# 7a. corpus geo_group distribution (Phase 4)
fig, ax = plt.subplots(figsize=(5, 3.2))
order = [g for g in GEO_GROUP_ORDER if g in geo_prior]
ax.bar(order, [geo_prior[g] for g in order], color=["#2b8cbe", "#fdae61", "#999999"])
ax.set_title("Corpus geo_group distribution")
ax.set_ylabel("share")
fig.tight_layout()
fig.savefig(os.path.join(FIG4, "geo_group.png"))
plt.close(fig)

# 7b. geo baseline vs retrieved (Phase 8)
fig, ax = plt.subplots(figsize=(6, 3.6))
x = np.arange(len(order))
w = 0.38
ax.bar(x - w / 2, [baseline_geo.get(g, 0) for g in order], w, label="baseline")
ax.bar(x + w / 2, [observed_geo.get(g, 0) for g in order], w, label="observed (retrieved)")
ax.set_xticks(x)
ax.set_xticklabels(order)
ax.set_ylabel("share")
ax.set_title("geo_group: baseline vs retrieved")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "geo_baseline_vs_retrieved.png"))
plt.close(fig)

# 7c. geo SPD / SRR (Phase 8)
for metric, data, ref in [("SPD", spd_geo, 0.0), ("SRR", srr_geo, 1.0)]:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    vals = [data[g] if data[g] is not None else 0 for g in order]
    ax.bar(order, vals, color="#2b8cbe")
    ax.axhline(ref, color="k", lw=0.8, ls="--")
    ax.set_title(f"geo_group {metric} (retrieval vs corpus)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"geo_{metric}.png"))
    plt.close(fig)

# 7d. lambda geo tradeoff (Phase 9)
mmr_only = lambda_df[lambda_df["config_type"] == "mmr"]
base = lambda_df[lambda_df["config_type"] == "baseline"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(mmr_only["SPD_high_resource"], mmr_only["Recall@10"], c="#2b8cbe", label="MMR configs")
ax.scatter(base["SPD_high_resource"], base["Recall@10"], c="#d7301f", marker="*", s=180, label="naive baseline")
for _, r in lambda_df.iterrows():
    ax.annotate(f"{r['lambda_rel']:.2f}/{r['lambda_fair']:.2f}",
                (r["SPD_high_resource"], r["Recall@10"]), fontsize=7,
                textcoords="offset points", xytext=(4, 4))
ax.set_xlabel("SPD_high_resource")
ax.set_ylabel("Recall@10 (known-item)")
ax.set_title("geo fairness vs utility tradeoff")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "SPD_geo_fairness_utility_tradeoff.png"))
plt.close(fig)

# 7e. citation rate per dimension (Phase 10, rendered as the rate ratio so the
# "cited in proportion to availability" line sits at a fixed 1.0 everywhere)
for rate_key in CITATION_RATE_DIMS:
    if rate_key not in synth:
        continue
    dim = rate_key.removeprefix("citation_rate_by_")
    rates = synth[rate_key]
    variants = list(rates)
    groups = list(rates[variants[0]])
    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, v in enumerate(variants):
        vals = [rates[v].get(g) or 0 for g in groups]
        ax.bar(x + (i - 0.5) * w, vals, w, label=v)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("cited / retrieved share")
    ax.set_title(f"Citation rate by {dim}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"citation_rate_by_{dim}.png"))
    plt.close(fig)

print("Done. All dual-label artifacts regenerated.")
