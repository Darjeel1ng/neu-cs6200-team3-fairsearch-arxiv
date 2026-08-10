"""Offline Phase 9 lambda ablation (RQ3), including mechanism-isolating configs.

Re-ranks existing Phase 8 naive retrieval results; no API calls.
Mirrors work_notebook.ipynb Phase 9 logic and writes:
  - data/update2_output/mmr_reranked_results.json
  - data/update2_output/lambda_ablation.csv
  - data/update2_output/figures/*tradeoff*.png
"""

from __future__ import annotations

import json
import math
import os
import time

import chromadb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geo_labels import geo_group

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "update2_output")
CHROMA_DIR = os.path.join(ROOT, "chroma_db")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

FAIRNESS_DIMENSION = "institution"  # {"institution", "geo"}

LAMBDA_CONFIGS = [
    # joint
    (0.80, 0.10, 0.10),
    (0.70, 0.20, 0.10),
    (0.60, 0.20, 0.20),
    (0.50, 0.30, 0.20),
    (0.40, 0.30, 0.30),
    # diversity-only
    (0.90, 0.10, 0.00),
    (0.70, 0.30, 0.00),
    # fairness-only
    (0.90, 0.00, 0.10),
    (0.70, 0.00, 0.30),
]


def config_type(lambda_rel, lambda_div, lambda_fair):
    if lambda_fair == 0.0 and lambda_div > 0.0:
        return "diversity_only"
    if lambda_div == 0.0 and lambda_fair > 0.0:
        return "fairness_only"
    return "mmr"


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_embedding_lookup(collection):
    print("Loading document embeddings from chroma_db ...")
    t0 = time.time()
    batch_size = 1000
    lookup = {}
    total = collection.count()
    for offset in range(0, total, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["embeddings", "metadatas"],
        )
        for emb, meta in zip(batch["embeddings"], batch["metadatas"]):
            vec = np.asarray(emb, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            lookup[meta["arxiv_id"]] = vec
        if (offset // batch_size) % 10 == 0:
            print(f"  loaded {min(offset + batch_size, total):,}/{total:,}")
    print(f"Embeddings loaded: {len(lookup):,} in {time.time() - t0:.1f}s")
    return lookup


def diversity_penalty(candidate_id, selected_ids, embedding_lookup):
    if not selected_ids or candidate_id not in embedding_lookup:
        return 0.0
    cand = embedding_lookup[candidate_id]
    max_sim = 0.0
    for doc_id in selected_ids:
        emb = embedding_lookup.get(doc_id)
        if emb is None:
            continue
        sim = float(np.dot(cand, emb))
        if sim > max_sim:
            max_sim = sim
    return max_sim


def mmr_rerank(query_df, lambda_rel, lambda_div, lambda_fair, embedding_lookup, top_k=10):
    remaining = query_df.copy()
    selected = []
    selected_ids = []

    while len(selected) < top_k and len(remaining) > 0:
        best_idx = None
        best_score = -1e9
        need_div = lambda_div > 0.0 and len(selected_ids) > 0

        for idx, row in remaining.iterrows():
            rel = row["relevance_score"]
            fair = row["fairness_score"]
            div = (
                diversity_penalty(row["document_id"], selected_ids, embedding_lookup)
                if need_div
                else 0.0
            )
            mmr = lambda_rel * rel - lambda_div * div + lambda_fair * fair
            if mmr > best_score:
                best_score = mmr
                best_idx = idx

        chosen = remaining.loc[best_idx]
        selected.append(chosen)
        selected_ids.append(chosen["document_id"])
        remaining = remaining.drop(best_idx)

    reranked = pd.DataFrame(selected).reset_index(drop=True)
    reranked["rank"] = np.arange(1, len(reranked) + 1)
    return reranked


def recall_at_10(result_df, relevant_doc):
    top10 = result_df[result_df["rank"] <= 10]
    return 1.0 if (top10["document_id"] == relevant_doc).any() else 0.0


def reciprocal_rank(result_df, relevant_doc):
    hit = result_df[result_df["document_id"] == relevant_doc]
    if len(hit) == 0:
        return 0.0
    return 1.0 / float(hit.iloc[0]["rank"])


def ndcg_at_10(result_df, relevant_doc):
    hit = result_df[result_df["document_id"] == relevant_doc]
    if len(hit) == 0:
        return 0.0
    rank = float(hit.iloc[0]["rank"])
    return (1.0 / math.log2(rank + 1)) / 1.0


def make_fairness_metrics(privilege_prior, region_prior, geo_group_prior):
    def fairness_metrics(result_df):
        unique_institutions = (
            result_df["institution"].loc[lambda s: s.ne("")].dropna().nunique()
        )
        unique_countries = result_df["country_code"].nunique()
        observed_privilege = (
            result_df["privilege_label"].value_counts(normalize=True).round(6).to_dict()
        )
        observed_region = (
            result_df["region"].value_counts(normalize=True).round(6).to_dict()
        )
        observed_geo = (
            result_df["geo_group"].value_counts(normalize=True).round(6).to_dict()
        )

        spd_privilege = {
            g: round(observed_privilege.get(g, 0) - exp, 6)
            for g, exp in privilege_prior.items()
        }
        srr_privilege = {
            g: round(observed_privilege.get(g, 0) / exp, 6) if exp > 0 else 0
            for g, exp in privilege_prior.items()
        }
        spd_region = {
            r: round(observed_region.get(r, 0) - exp, 6)
            for r, exp in region_prior.items()
        }
        srr_region = {
            r: round(observed_region.get(r, 0) / exp, 6) if exp > 0 else 0
            for r, exp in region_prior.items()
        }
        spd_geo = {
            g: round(observed_geo.get(g, 0) - exp, 6)
            for g, exp in geo_group_prior.items()
        }
        srr_geo = {
            g: round(observed_geo.get(g, 0) / exp, 6) if exp > 0 else 0
            for g, exp in geo_group_prior.items()
        }
        return {
            "unique_institutions": unique_institutions,
            "unique_countries": unique_countries,
            "spd_privilege": spd_privilege,
            "spd_region": spd_region,
            "spd_geo_group": spd_geo,
            "srr_privilege": srr_privilege,
            "srr_region": srr_region,
            "srr_geo_group": srr_geo,
        }

    return fairness_metrics


def summarize_config(records_metrics):
    (
        ndcg_scores,
        recall_scores,
        mrr_scores,
        spd_under_scores,
        srr_under_scores,
        spd_priv_scores,
        srr_priv_scores,
        spd_geo_high_scores,
        spd_geo_emerging_scores,
        srr_geo_high_scores,
        srr_geo_emerging_scores,
        institution_counts,
        country_counts,
    ) = records_metrics
    return {
        "nDCG@10": float(np.mean(ndcg_scores)),
        "Recall@10": float(np.mean(recall_scores)),
        "MRR": float(np.mean(mrr_scores)),
        "SPD_underrepresented": float(np.mean(spd_under_scores)),
        "SPD_privileged": float(np.mean(spd_priv_scores)),
        "SRR_underrepresented": float(np.mean(srr_under_scores)),
        "SRR_privileged": float(np.mean(srr_priv_scores)),
        "SPD_high_resource": float(np.mean(spd_geo_high_scores)),
        "SPD_emerging": float(np.mean(spd_geo_emerging_scores)),
        "SRR_high_resource": float(np.mean(srr_geo_high_scores)),
        "SRR_emerging": float(np.mean(srr_geo_emerging_scores)),
        "Unique Institutions": float(np.mean(institution_counts)),
        "Unique Countries": float(np.mean(country_counts)),
    }


def eval_reranked(reranked, query_id, ground_truth, fairness_metrics):
    ndcg = recall = mrr = None
    if query_id in ground_truth:
        gt = ground_truth[query_id]
        ndcg = ndcg_at_10(reranked, gt)
        recall = recall_at_10(reranked, gt)
        mrr = reciprocal_rank(reranked, gt)
    fm = fairness_metrics(reranked)
    return ndcg, recall, mrr, fm


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)

    retrieval_df = pd.read_json(os.path.join(OUTPUT_DIR, "naive_retrieval_results.json"))
    queries_df = pd.read_json(os.path.join(OUTPUT_DIR, "queries.json"))
    with open(os.path.join(DATA_DIR, "fairness_baseline_priors.json"), encoding="utf-8") as f:
        fairness_priors = json.load(f)

    if "geo_group" not in retrieval_df.columns:
        retrieval_df["geo_group"] = retrieval_df["country_code"].map(geo_group)

    privilege_prior = fairness_priors["privilege_prior"]
    region_prior = fairness_priors["region_prior"]
    geo_group_prior = fairness_priors["geo_group_prior"]
    fairness_metrics = make_fairness_metrics(
        privilege_prior, region_prior, geo_group_prior
    )

    region_inverse = {r: 1 / p for r, p in region_prior.items()}
    max_inverse = max(region_inverse.values())
    region_reward = {r: v / max_inverse for r, v in region_inverse.items()}

    def fairness_score(row):
        if FAIRNESS_DIMENSION == "geo":
            return 1.0 if row.get("geo_group") == "emerging" else 0.0
        privilege_component = 1.0 if row["privilege_label"] == "underrepresented" else 0.0
        region_component = region_reward.get(row["region"], 0.0)
        institution_component = 0.0 if row["is_top50_institution"] else 1.0
        return (privilege_component + region_component + institution_component) / 3.0

    retrieval_df["fairness_score"] = retrieval_df.apply(fairness_score, axis=1)
    retrieval_df["relevance_score"] = (
        retrieval_df.groupby("query_id")["score"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
        )
    )

    ground_truth = {
        qid: doc
        for qid, doc in zip(queries_df["query_id"], queries_df["document_id"])
        if isinstance(doc, str) and doc
    }
    print(f"Queries: {len(queries_df)} | known-item: {len(ground_truth)}")
    print(f"Retrieved rows: {len(retrieval_df):,}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("cs50k_collection")
    embedding_lookup = load_embedding_lookup(collection)

    results_summary = []
    all_reranked_results = {}
    t_ablate = time.time()

    for lambda_rel, lambda_div, lambda_fair in LAMBDA_CONFIGS:
        ctype = config_type(lambda_rel, lambda_div, lambda_fair)
        print(
            f"Running λ=({lambda_rel:.2f}, {lambda_div:.2f}, {lambda_fair:.2f}) "
            f"[{ctype}] ..."
        )
        t0 = time.time()
        reranked_records = []
        ndcg_scores, recall_scores, mrr_scores = [], [], []
        spd_under_scores, srr_under_scores = [], []
        spd_priv_scores, srr_priv_scores = [], []
        spd_geo_high_scores, spd_geo_emerging_scores = [], []
        srr_geo_high_scores, srr_geo_emerging_scores = [], []
        institution_counts, country_counts = [], []

        for query_id, group in retrieval_df.groupby("query_id"):
            reranked = mmr_rerank(
                group,
                lambda_rel=lambda_rel,
                lambda_div=lambda_div,
                lambda_fair=lambda_fair,
                embedding_lookup=embedding_lookup,
                top_k=10,
            )
            reranked_records.extend(reranked.to_dict("records"))
            ndcg, recall, mrr, fm = eval_reranked(
                reranked, query_id, ground_truth, fairness_metrics
            )
            if ndcg is not None:
                ndcg_scores.append(ndcg)
                recall_scores.append(recall)
                mrr_scores.append(mrr)
            spd_under_scores.append(fm["spd_privilege"]["underrepresented"])
            srr_under_scores.append(fm["srr_privilege"]["underrepresented"])
            spd_priv_scores.append(fm["spd_privilege"]["privileged"])
            srr_priv_scores.append(fm["srr_privilege"]["privileged"])
            spd_geo_high_scores.append(fm["spd_geo_group"].get("high_resource", 0))
            spd_geo_emerging_scores.append(fm["spd_geo_group"].get("emerging", 0))
            srr_geo_high_scores.append(fm["srr_geo_group"].get("high_resource", 0))
            srr_geo_emerging_scores.append(fm["srr_geo_group"].get("emerging", 0))
            institution_counts.append(fm["unique_institutions"])
            country_counts.append(fm["unique_countries"])

        key = f"rel_{lambda_rel:.2f}_div_{lambda_div:.2f}_fair_{lambda_fair:.2f}"
        all_reranked_results[key] = reranked_records
        row = {
            "lambda_rel": lambda_rel,
            "lambda_div": lambda_div,
            "lambda_fair": lambda_fair,
            "config_type": ctype,
            **summarize_config(
                (
                    ndcg_scores,
                    recall_scores,
                    mrr_scores,
                    spd_under_scores,
                    srr_under_scores,
                    spd_priv_scores,
                    srr_priv_scores,
                    spd_geo_high_scores,
                    spd_geo_emerging_scores,
                    srr_geo_high_scores,
                    srr_geo_emerging_scores,
                    institution_counts,
                    country_counts,
                )
            ),
        }
        results_summary.append(row)
        print(
            f"  done in {time.time() - t0:.1f}s | "
            f"nDCG={row['nDCG@10']:.4f} SPD_priv={row['SPD_privileged']:.4f}"
        )

    print(f"All configs finished in {time.time() - t_ablate:.1f}s")

    # Naive baseline
    _b_recall, _b_ndcg, _b_mrr = [], [], []
    _b_spd_u, _b_srr_u, _b_spd_p, _b_srr_p = [], [], [], []
    _b_spd_gh, _b_spd_ge, _b_srr_gh, _b_srr_ge = [], [], [], []
    _b_inst, _b_ctry = [], []
    for query_id, group in retrieval_df.groupby("query_id"):
        top10 = group[group["rank"] <= 10]
        if query_id in ground_truth:
            gt = ground_truth[query_id]
            _b_recall.append(recall_at_10(top10, gt))
            _b_ndcg.append(ndcg_at_10(top10, gt))
            _b_mrr.append(reciprocal_rank(top10, gt))
        fm = fairness_metrics(top10)
        _b_spd_u.append(fm["spd_privilege"]["underrepresented"])
        _b_srr_u.append(fm["srr_privilege"]["underrepresented"])
        _b_spd_p.append(fm["spd_privilege"]["privileged"])
        _b_srr_p.append(fm["srr_privilege"]["privileged"])
        _b_spd_gh.append(fm["spd_geo_group"].get("high_resource", 0))
        _b_spd_ge.append(fm["spd_geo_group"].get("emerging", 0))
        _b_srr_gh.append(fm["srr_geo_group"].get("high_resource", 0))
        _b_srr_ge.append(fm["srr_geo_group"].get("emerging", 0))
        _b_inst.append(fm["unique_institutions"])
        _b_ctry.append(fm["unique_countries"])

    baseline_row = {
        "lambda_rel": 1.0,
        "lambda_div": 0.0,
        "lambda_fair": 0.0,
        "config_type": "baseline",
        "nDCG@10": float(np.mean(_b_ndcg)),
        "Recall@10": float(np.mean(_b_recall)),
        "MRR": float(np.mean(_b_mrr)),
        "SPD_underrepresented": float(np.mean(_b_spd_u)),
        "SPD_privileged": float(np.mean(_b_spd_p)),
        "SRR_underrepresented": float(np.mean(_b_srr_u)),
        "SRR_privileged": float(np.mean(_b_srr_p)),
        "SPD_high_resource": float(np.mean(_b_spd_gh)),
        "SPD_emerging": float(np.mean(_b_spd_ge)),
        "SRR_high_resource": float(np.mean(_b_srr_gh)),
        "SRR_emerging": float(np.mean(_b_srr_ge)),
        "Unique Institutions": float(np.mean(_b_inst)),
        "Unique Countries": float(np.mean(_b_ctry)),
    }

    lambda_df = pd.concat(
        [pd.DataFrame([baseline_row]), pd.DataFrame(results_summary)],
        ignore_index=True,
    )

    mmr_path = os.path.join(OUTPUT_DIR, "mmr_reranked_results.json")
    with open(mmr_path, "w", encoding="utf-8") as f:
        json.dump(all_reranked_results, f, indent=2, default=_json_default)

    csv_path = os.path.join(OUTPUT_DIR, "lambda_ablation.csv")
    lambda_df.to_csv(csv_path, index=False)

    print("Saved:", mmr_path)
    print("Saved:", csv_path)
    print(
        lambda_df[
            [
                "config_type",
                "lambda_rel",
                "lambda_div",
                "lambda_fair",
                "nDCG@10",
                "Recall@10",
                "SPD_privileged",
                "SPD_high_resource",
                "Unique Institutions",
            ]
        ].to_string(index=False)
    )

    # Figures (same set as notebook)
    plt.figure(figsize=(8, 6))
    plt.plot(
        lambda_df["SPD_underrepresented"],
        lambda_df["nDCG@10"],
        marker="o",
        linewidth=2,
        label="nDCG@10",
    )
    plt.plot(
        lambda_df["SPD_underrepresented"],
        lambda_df["MRR"],
        marker="s",
        linewidth=2,
        label="MRR",
    )
    plt.xlabel("Statistical Parity Difference (SPD Underrepresented)")
    plt.ylabel("Utility")
    plt.title("SPD Fairness-Utility Tradeoff")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(FIGURE_DIR, "SPD_fairness_utility_tradeoff.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(
        lambda_df["SRR_underrepresented"],
        lambda_df["nDCG@10"],
        marker="o",
        linewidth=2,
        label="nDCG@10",
    )
    plt.plot(
        lambda_df["SRR_underrepresented"],
        lambda_df["MRR"],
        marker="s",
        linewidth=2,
        label="MRR",
    )
    plt.xlabel("SRR Underrepresented")
    plt.ylabel("Utility")
    plt.title("SRR Fairness-Utility Tradeoff")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(FIGURE_DIR, "SRR_fairness_utility_tradeoff.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(
        lambda_df["lambda_fair"],
        lambda_df["Unique Institutions"],
        marker="o",
        linewidth=2,
        label="Institutions",
    )
    plt.plot(
        lambda_df["lambda_fair"],
        lambda_df["Unique Countries"],
        marker="s",
        linewidth=2,
        label="Countries",
    )
    plt.xlabel("Fairness Weight (λ_fair)")
    plt.ylabel("Average Unique Count")
    plt.title("Representation Diversity vs Fairness Weight")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(FIGURE_DIR, "representation_diversity.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    _base = lambda_df[lambda_df["config_type"] == "baseline"]
    _style = {
        "mmr": ("#2b8cbe", "o", "joint MMR"),
        "diversity_only": ("#31a354", "s", "diversity-only"),
        "fairness_only": ("#756bb1", "D", "fairness-only"),
    }
    plt.figure(figsize=(8, 6))
    for ctype, (color, marker, label) in _style.items():
        sub = lambda_df[lambda_df["config_type"] == ctype]
        if len(sub) == 0:
            continue
        plt.scatter(
            sub["SPD_high_resource"],
            sub["Recall@10"],
            c=color,
            marker=marker,
            s=80,
            label=label,
        )
    plt.scatter(
        _base["SPD_high_resource"],
        _base["Recall@10"],
        c="#d7301f",
        marker="*",
        s=250,
        label="naive baseline",
    )
    plt.xlabel("SPD (high_resource)")
    plt.ylabel("Recall@10 (known-item)")
    plt.title("geo_group Fairness-Utility Tradeoff")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(FIGURE_DIR, "SPD_geo_fairness_utility_tradeoff.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Figures saved in", FIGURE_DIR)


if __name__ == "__main__":
    main()
