# Paper revision checklist

The final report's editable source is not in this repo (only PDFs live in the
parent folder), so this is the hand-off list: every number, claim, and method
description in the paper that the current pipeline contradicts or supersedes.
Numbers below are the authoritative ones — all are reproducible from
`work_notebook.ipynb` and the artifacts in `data/`.

Everything is measured on the same 150-query bias-audit benchmark (100 neutral
known-item queries + 50 debate queries) over the 50,000-document corpus.

---

## 1. Method corrections (write these before touching the numbers)

| Paper says | Reality | Source |
|---|---|---|
| Fairness is audited on one group dimension | **Two independent dimensions**: `privilege_label` (QS Top-50 CS institution tier) and `geo_group` (World Bank income proxy on `country_code`) | `data/fairness_labeling_rules.md`, `geo_labels.py` |
| MMR re-ranking via LlamaIndex's built-in single-λ MMR | **Custom three-signal re-ranker** (relevance / diversity / fairness) with an independent λ per signal, swept over 5 configs | Phase 9 cells |
| `P@10` | **Recall@10 (HitRate@10)**, a known-item metric — each query has exactly one ground-truth document, so the old P@10 was capped at 0.1 (this is why it was pinned at 0.06, it was not a bug) | `data/update2_output/lambda_ablation.csv` |
| nDCG@10 as a graded metric | Single relevant document, so IDCG = 1; report it as such | Phase 9 cells |
| Corpus = arXiv CS sample | Kaggle `Cornell-University/arxiv` **v289**, filtered to `cs.*`, then **OpenAlex-gated** so every retained paper has a resolvable primary institution (this is what makes the fairness labels meaningful, and it biases the corpus toward institutionally-indexed work — state it as a limitation) | Phase 1 |
| Generator / judge = Claude | **`deepseek-v4-flash` for all three LLM roles** (synthesis generation, stance classification, RAGAS judge) via DeepSeek's OpenAI-compatible API | Phase 10 cells |
| Faithfulness only | RAGAS **faithfulness + answer relevancy + context precision**, one shared judge so the three are comparable; answer relevancy uses the same all-MiniLM-L6-v2 embeddings the index was built on | Phase 10 cells |

Also worth a sentence in limitations: `geo_group` is an **income proxy**, not a
Global North/South geography. It was chosen over a raw region split because Asia
mixes high-income (JP/KR/SG/HK/IL) with non-high-income (CN/IN) economies. The
region-based alternative is recorded but unused in
`data/fairness_labeling_rules.md`.

---

## 2. Numbers to replace

### Corpus (Phase 3-4)

- Institution tier: privileged **17.1%** (8,553) / underrepresented **82.9%**
  (41,447) / unknown 0%.
- Geo group: high_resource **81.1%** / emerging **18.6%** / unknown **0.3%**.
- Region: Europe 43.1% / North America 27.0% / Asia 25.3% / Oceania 2.3% /
  South America 1.5% / Africa 0.5% / Unknown 0.3%.
- Concentration: Institution Gini **0.7525**, Country Gini **0.8654**.

### RQ1 — retrieval parity (Phase 8)

| dimension | group | baseline | retrieved | SPD | SRR |
|---|---|---|---|---|---|
| institution | privileged | 0.1711 | 0.1650 | **-0.0061** | **0.965** |
| geo | high_resource | 0.8110 | 0.8373 | **+0.0264** | **1.032** |
| geo | emerging | 0.1860 | 0.1593 | **-0.0267** | **0.857** |
| region | Europe | 0.4306 | 0.4630 | +0.0324 | 1.075 |
| region | Asia | 0.2527 | 0.2233 | -0.0293 | 0.884 |

Equalized odds on the institution dimension: `group_tpr` privileged 0.900 vs
underrepresented 0.913, **gap 0.0125**.

### RQ1 diagnostic — is it the embeddings or the volume? (new, Phase 8)

PCA + a 5-fold linear probe on 12,000 document vectors read directly out of the
Phase 5 Chroma index:

| dimension | linear-probe ROC-AUC | silhouette (cosine) | centroid dist. / within-group spread |
|---|---|---|---|
| `privilege_label` | 0.641 | -0.002 | 0.066 |
| `geo_group` | 0.722 | -0.000 | 0.112 |

Silhouette ≈ 0 and centroid separation of only 7-11% of the within-group spread
mean the groups are **intermixed**, not occupying separate regions of the space.
The above-chance probe AUC reflects topical differences between groups, not a
separable region. Conclusion for the paper: the parity gap is a **corpus-volume**
effect, not embedding geometry.

### RQ3 — MMR fairness/utility tradeoff (Phase 9)

Utility is flat across every configuration: Recall@10 = **0.90** and nDCG@10 ≈
**0.83** everywhere, including the naive baseline (now an explicit row in
`lambda_ablation.csv`). Fairness moves, utility does not:

| λ_rel / λ_div / λ_fair | SPD_privileged | SPD_high_resource |
|---|---|---|
| 1.0 / 0.0 / 0.0 (naive baseline) | -0.006 | +0.040 |
| 0.8 / 0.1 / 0.1 | -0.049 | +0.035 |
| 0.6 / 0.2 / 0.2 | -0.096 | +0.021 |
| 0.4 / 0.3 / 0.3 | -0.139 | +0.015 |

Two things to say about this: (a) because the institution dimension starts near
parity, pushing λ_fair **over-corrects** it (SPD_privileged goes to -0.14, i.e.
privileged papers become under-represented); (b) the same institution-targeted
objective **also** shrinks the geo gap (+0.040 → +0.015), because non-Top50 and
emerging-economy documents overlap heavily.

### RQ2 — synthesis faithfulness and citation bias (Phase 10)

Full 150-query run, both prompting variants, `deepseek-v4-flash` throughout.
**Zero refusals** (150/150 usable syntheses per variant), 300/300 rows scored on
every RAGAS metric.

| RAGAS metric (mean) | standard | balanced |
|---|---|---|
| faithfulness | 0.924 | 0.907 |
| answer relevancy | 0.674 | 0.667 |
| context precision | 0.803 | 0.796 |

- Citation volume: standard **4.50** citations / **3.60** distinct docs per
  query; balanced **7.66** / **7.17**. The perspective-balanced prompt roughly
  doubles citation breadth.
- Citation rate (cited-share ÷ retrieved-share): privileged **1.05×**
  (standard) / **1.08×** (balanced); emerging **0.92×** / **1.07×**.
- **None of these shifts is statistically significant** (two-proportion z-test,
  standard vs balanced: privileged p=0.80, emerging p=0.19, high_resource
  p=0.22).
- Stance mix of retrieved documents: neutral **67.2%** / pro-consensus
  **28.9%** / dissenting **3.9%**.
- 5 of 1,811 citations referenced a `doc_N` label outside the provided context.

---

## 3. Claims to retract or rewrite

1. **"Perspective-balanced prompting reduces privilege over-citation from 1.25×
   to 1.09×."** Does not replicate. That came from a 40-query Claude-generated /
   Claude-judged run. At 150 queries with DeepSeek the privileged citation rate
   is ~1.05-1.08× under *both* prompts and the difference is indistinguishable
   from noise (p=0.80). What the balanced prompt demonstrably does is cite
   **more** documents, not **different** ones — rewrite the claim as a
   coverage/breadth effect.

2. **Any framing of RQ1 as "retrieval amplifies institutional prestige."** It
   does not, in this corpus: SPD_privileged = -0.006, SRR = 0.96. Report the
   near-parity as the finding, and move the bias narrative to the geo dimension
   (emerging SRR 0.857), which is where the data actually supports it.

3. **Anything treating P@10 ≈ 0.06 as a weak-retrieval result.** It was a metric
   ceiling artifact. The honest number is Recall@10 = 0.90.

4. **Any RQ2 claim resting on the 40-query sample.** The sample is now 150 and
   the models changed; do not mix the two runs' numbers in one table or compare
   them across the two generators.

---

## 4. New limitations to add

- Judge and generator are the same model (`deepseek-v4-flash`), so RAGAS scores
  carry a self-evaluation bias; they are usable for the standard-vs-balanced
  comparison but not as absolute quality figures.
- The OpenAlex institution gate excludes papers with no resolvable affiliation,
  which plausibly skews the corpus toward well-indexed institutions before any
  retrieval happens.
- Utility is known-item: one ground-truth document per neutral query, and the
  50 debate queries have none, so Recall@10 / nDCG@10 / MRR are averaged over
  100 queries while fairness metrics use all 150.
- `geo_group` is an income proxy; it classifies high-income Asian economies as
  `high_resource`, which is intentional but differs from a Global North/South
  reading.
- Retrieval-stage debate coverage is thin: only 3.9% of retrieved documents are
  dissenting, which limits what RQ2 can say about viewpoint suppression.
