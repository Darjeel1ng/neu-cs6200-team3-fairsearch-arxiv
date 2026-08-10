# Paper revision notes

Hand-off list for whoever edits the report. Section references below point at
`CS_6200__Team_3_.pdf` (the version in the parent folder as of Aug 6). The
report's editable source is not in this repo, so this file is the bridge
between the pipeline and the write-up.

Everything is measured on the same 150-query benchmark (100 neutral known-item
queries + 50 debate queries) over the 50,000-document corpus.

---

## Where to look up any number

All results live in `data/update2_output/`. They are reproducible: two
consecutive notebook runs produce byte-identical artifacts, and the notebook
cell outputs match the files on disk.

| What you need | File |
|---|---|
| Retrieval parity: SPD / SRR / equalized odds, both group dimensions | `retrieval_parity_report.json` |
| Utility-fairness tradeoff across λ configs, incl. a real baseline row | `lambda_ablation.csv` |
| Embedding geometry diagnostic (PCA, linear probe, silhouette) | `embedding_pca_report.json` |
| Synthesis: RAGAS scores, citation rates, significance tests | `synthesis_eval_report.json` |
| Per-query RAGAS scores | `ragas_scores.csv` |
| Raw LLM syntheses and stance labels | `synthesis_outputs.json`, `stance_outputs.json` |
| Corpus priors used as the fairness reference | `data/fairness_baseline_priors.json` |
| How the two label schemes are defined | `data/fairness_labeling_rules.md`, `geo_labels.py` |
| All figures | `data/update2_output/figures/` |

Do not pull numbers from older drafts or from README versions before Aug 6;
several region SPD/SRR values there were stale. The interactive view of the same
data is `dashboard/` (`docker compose up`).

---

## 1. Fixes needed, most urgent first

### 1.1 Table 4 numbers are all off by a factor of 1.5

NDCG@10 and MRR in Table 4 were averaged over all 150 queries, but the 50 debate
queries have no ground-truth document and were scored 0. The correct denominator
is the 100 known-item queries, which makes every value exactly 1.5x higher.

| λ_rel / λ_div / λ_fair | NDCG@10 (paper) | NDCG@10 (correct) | MRR (paper) | MRR (correct) |
|---|---|---|---|---|
| baseline | 0.5513 | **0.8305** | 0.5352 | **0.8078** |
| 0.8 / 0.1 / 0.1 | 0.5513 | **0.8269** | 0.5352 | **0.8028** |
| 0.7 / 0.2 / 0.1 | 0.5509 | **0.8263** | 0.5347 | **0.8020** |
| 0.6 / 0.2 / 0.2 | 0.5494 | **0.8241** | 0.5329 | **0.7993** |
| 0.5 / 0.3 / 0.2 | 0.5498 | **0.8248** | 0.5334 | **0.8002** |
| 0.4 / 0.3 / 0.3 | 0.5384 | **0.8076** | 0.5186 | **0.7779** |

Three knock-on edits:

- Section 4 and the Conclusion both say the strongest fairness setting costs
  "approximately a 2.3% reduction" in NDCG@10. On the corrected denominator it
  is **2.8%** (0.8305 → 0.8076). The Conclusion's "0.5513 to 0.5384" becomes
  "0.8305 to 0.8076".
- The current baseline row duplicates the λ=0.8 row and lists SPD 0.0000. There
  is now a genuinely computed baseline: NDCG 0.8305, MRR 0.8078, SPD 0.0064.
- The SPD column is correct as printed and needs no change.

### 1.2 We do not implement Fair Top-k

The report attributes the reranker to "MMR and Fair Top-k fairness constraints"
and cites Zehlike et al. [5] in five places: the Introduction contribution list,
the end of Section 2.1, Table 3, the method paragraph in Section 4, and the
Conclusion.

The implementation is a three-signal weighted MMR,
`score = λ_rel · relevance − λ_div · diversity + λ_fair · fairness`, where the
fairness term averages a privilege, a region, and an institution component.
There is no FA*IR quota or binomial test anywhere in the code.

Either implement FA*IR or reword all five mentions to "fairness-aware MMR
reranking" and demote [5] to related work that inspired the design.

### 1.3 Table 3: Precision@10 → Recall@10 (HitRate@10)

Known-item setup with exactly one relevant document per query, so dividing by 10
capped the score at 0.1. The honest number is **Recall@10 = 0.90**, flat across
every λ configuration. Anything in the text treating the old ~0.06 as evidence of
weak retrieval should go.

### 1.4 Context Recall was promised but not run

Plan for Next lists four RAGAS metrics; we ran three (faithfulness, answer
relevancy, context precision). Either add it or drop it from the list.

### 1.5 Other method descriptions to correct

| Report says | Reality |
|---|---|
| One group dimension | **Two**: `privilege_label` (QS Top-50 CS tier) and `geo_group` (World Bank income proxy on `country_code`) |
| nDCG@10 as a graded metric | One relevant document, so IDCG = 1; report it as a rank-position signal |
| Corpus = arXiv CS sample | Kaggle `Cornell-University/arxiv` **v289**, filtered to `cs.*`, then **OpenAlex-gated** so every retained paper has a resolvable primary institution |
| (generation stage unwritten) | `deepseek-v4-flash` for all three LLM roles: synthesis, stance classification, RAGAS judge, via the OpenAI-compatible API |

Tables 2 and 5 are correct as printed and need no edit.

---

## 2. Authoritative numbers

### Corpus

- Institution tier: privileged **17.1%** (8,553) / underrepresented **82.9%**
  (41,447) / unknown 0%.
- Geo group: high_resource **81.1%** / emerging **18.6%** / unknown **0.3%**.
- Region: Europe 43.1% / North America 27.0% / Asia 25.3% / Oceania 2.3% /
  South America 1.5% / Africa 0.5% / Unknown 0.3%.
- Concentration: Institution Gini **0.7525**, Country Gini **0.8654**.

### Retrieval parity

| dimension | group | corpus | retrieved | SPD | SRR |
|---|---|---|---|---|---|
| institution | privileged | 0.1711 | 0.1650 | **-0.0061** | **0.965** |
| geo | high_resource | 0.8110 | 0.8373 | **+0.0264** | **1.032** |
| geo | emerging | 0.1860 | 0.1593 | **-0.0267** | **0.857** |
| region | Europe | 0.4306 | 0.4630 | +0.0324 | 1.075 |
| region | Asia | 0.2527 | 0.2233 | -0.0293 | 0.884 |

Equalized odds on the institution dimension: privileged 0.900 vs
underrepresented 0.913, **gap 0.0125**.

### Embedding geometry diagnostic

PCA + a 5-fold linear probe on 12,000 vectors read straight out of the Chroma
index:

| dimension | linear-probe ROC-AUC | silhouette (cosine) | centroid dist. / within-group spread |
|---|---|---|---|
| `privilege_label` | 0.641 | -0.002 | 0.066 |
| `geo_group` | 0.722 | -0.000 | 0.112 |

Silhouette near zero and centroid separation of only 7-11% of the within-group
spread mean the groups are **intermixed**, not occupying separate regions. The
above-chance probe AUC reflects topical differences, not a separable region.
Conclusion for the report: the parity gap is a **corpus-volume** effect, not
embedding geometry.

### Fairness-utility tradeoff

Utility is flat everywhere: Recall@10 = 0.90 and nDCG@10 ≈ 0.83 across all
configurations including the baseline. Only fairness moves:

| λ_rel / λ_div / λ_fair | type | SPD_privileged | SPD_high_resource |
|---|---|---|---|
| 1.0 / 0.0 / 0.0 | baseline | -0.006 | +0.040 |
| 0.8 / 0.1 / 0.1 | joint | -0.049 | +0.035 |
| 0.6 / 0.2 / 0.2 | joint | -0.096 | +0.021 |
| 0.4 / 0.3 / 0.3 | joint | -0.139 | +0.015 |
| 0.9 / 0.1 / 0.0 | diversity-only | -0.008 | +0.043 |
| 0.7 / 0.3 / 0.0 | diversity-only | -0.008 | +0.041 |
| 0.9 / 0.0 / 0.1 | fairness-only | -0.046 | +0.036 |
| 0.7 / 0.0 / 0.3 | fairness-only | -0.105 | +0.017 |

**RQ3 mechanism isolation (added to answer the professor's confound):**
diversity-only leaves `SPD_privileged` near the baseline (~-0.008), while
fairness-only at comparable `λ_fair` reproduces the joint SPD shift
(-0.046 / -0.105). So the fairness movement in the joint sweep is attributable
to the fairness term, not diversity. Re-run offline with
`python run_phase9_ablation.py` (no API calls); full table in
`lambda_ablation.csv`.

Two further points: because the institution dimension starts near parity,
raising λ_fair **over-corrects** it (privileged papers end up under-represented
at -0.14); and the same institution-targeted objective **also** shrinks the geo
gap (+0.040 → +0.015), because non-Top50 and emerging-economy documents overlap
heavily.

### Synthesis stage

Full 150-query run per variant, `deepseek-v4-flash` throughout. Zero refusals
(150/150 usable syntheses per variant), 300/300 rows scored on every metric.

| RAGAS metric (mean) | standard | balanced |
|---|---|---|
| faithfulness | 0.924 | 0.907 |
| answer relevancy | 0.674 | 0.667 |
| context precision | 0.803 | 0.796 |

- Citation volume: standard **4.50** citations / **3.60** distinct docs per
  query; balanced **7.66** / **7.17**. The balanced prompt roughly doubles
  citation breadth.
- Citation rate (cited share ÷ retrieved share): privileged **1.05x**
  (standard) / **1.08x** (balanced); emerging **0.92x** / **1.07x**.
- **No shift is statistically significant** (two-proportion z-test, standard vs
  balanced: privileged p=0.80, emerging p=0.19, high_resource p=0.22).
- Stance mix of retrieved documents: neutral **67.2%** / pro-consensus **28.9%**
  / dissenting **3.9%**.
- 5 of 1,811 citations referenced a `doc_N` label outside the provided context.

---

## 3. Claims to retract or rewrite

1. **"Perspective-balanced prompting reduces privilege over-citation from 1.25x
   to 1.09x."** Does not replicate. That came from a 40-query Claude-generated,
   Claude-judged run. At 150 queries the privileged citation rate is ~1.05-1.08x
   under *both* prompts and the difference is indistinguishable from noise
   (p=0.80). What the balanced prompt demonstrably does is cite **more**
   documents, not **different** ones. Rewrite it as a coverage effect.

2. **Any framing of retrieval as "amplifying institutional prestige."** It does
   not, in this corpus. The report already says so ("differs from our initial
   hypothesis"), so this is mostly about not reintroducing the stronger claim
   when the new sections get written.

3. **Anything treating P@10 ≈ 0.06 as a weak-retrieval result.** Metric ceiling
   artifact; the honest number is Recall@10 = 0.90.

4. **Any synthesis-stage claim resting on the 40-query sample.** The sample is
   now 150 and the generator changed; do not mix the two runs in one table.

---

## 4. New content to write up

- **The `geo_group` analysis**, as the positive counterpart to the
  institution-tier null result the report already reports. This is where the
  data actually supports a bias narrative (emerging SRR 0.857, Country Gini
  0.865 > Institution Gini 0.752).
- **The embedding geometry diagnostic**, which explains *why* the institution
  tier shows parity.
- **The synthesis stage.** Note the headline result is **negative**: prompt
  design does not measurably change institutional attribution on this benchmark.

---

## 5. Limitations to add

- Judge and generator are the same model (`deepseek-v4-flash`), so RAGAS scores
  carry a self-evaluation bias. Usable for the standard-vs-balanced comparison,
  not as absolute quality figures.
- The OpenAlex institution gate excludes papers with no resolvable affiliation,
  which plausibly skews the corpus toward well-indexed institutions before any
  retrieval happens.
- Utility is known-item: one ground-truth document per neutral query, and the 50
  debate queries have none, so Recall@10 / nDCG@10 / MRR average over 100
  queries while fairness metrics use all 150.
- `geo_group` is an **income proxy**, not a Global North/South geography. It
  classifies high-income Asian economies as `high_resource`, which is
  intentional: a raw region split would lump JP/KR/SG/HK/IL together with CN/IN.
  The region-based alternative is recorded but unused in
  `data/fairness_labeling_rules.md`.
- Retrieval-stage debate coverage is thin: only 3.9% of retrieved documents are
  dissenting, which limits what the synthesis analysis can say about viewpoint
  suppression.
