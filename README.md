# neu-cs6200-team3-fairsearch-arxiv

Working notes for the FairSearch-arXiv project. **Phases 10 and 11 are still working on**, Everything below is a quick reference, not a formal report.

---

## TL;DR for whoever picks this up

- All code lives in `work_notebook.ipynb`, organized as Phase 1 -> Phase 5.
- Phases 1-5 have already been run; the resulting corpus + reports are in `data/`.
- The single most important file is **`data/final_50k_labeled.parquet`** - the
  final 50K corpus with cleaned text + institution + region + fairness labels.
  That is the direct input for Phase 5.
- Large files are stored with **Git LFS**, so after cloning run:

```bash
git lfs install
git lfs pull
```

If you skip `git lfs pull`, the `.parquet` files will look like tiny text
pointer stubs instead of real data.

---

## What's in `data/` (and why)

### Committed - the result corpus (Git LFS)
| File | ~Size | What it is |
|------|-------|-----------|
| `final_50k_labeled.parquet` | 110 MB | **Final result.** Cleaned text + institution/country/region + `privilege_label`. Input for Phase 5. |
| `final_50k_cleaned.parquet` | 110 MB | Phase 2 output (cleaned text, before labels). Input for Phase 3. |
| `final_50k_with_institution.parquet` | 46 MB | Phase 1 output (50K with OpenAlex institution). The reproducibility seed: Phases 2-4 can be regenerated from this offline. |
| `openalex_gate_cache.json` | 66 MB | Cache of all OpenAlex DOI lookups from Phase 1. This is the expensive artifact (~2.5h of API calls); keep it so Phase 1 never has to hit the API again. |

### Committed - small reports / stats / figures (plain git)
- Phase 1: `raw_data_profile.json`, `corpus_selection_report.json`, `openalex_gate_report.json`
- Phase 2: `text_cleaning_report.json`
- Phase 3: `affiliation_match_report.json`, `institution_alias_map.csv`, `label_distribution.csv`
- Phase 4: `final_corpus_statistics.csv`, `fairness_baseline_priors.json`, `corpus_summary.md`, `figures/*.png`
- Phase 5: `chroma_db/`, `chroma_ingestion_report.json`, `sample_retrieval_results.md`, `index_config.yaml`

Update2_output folder:
- Phase 6: `phase6_sample_retrieval.json`, `phase6_retrieval_pipeline_config.json`
- Phase 7: `queries.json`, `query_benchmark_report.json`
- Phase 8: `naive_retrieval_results.json`, `retrieval_parity_report.json` and figures
- Phase 9: `mmr_reranked_results.json`, `lambda_ablation.csv` and figures
- Phase 10:
- Phase 11:  

These let you sanity-check each phase without opening the parquet files.

### NOT committed (see `.gitignore`)
- `cs_candidate_pool.parquet` (~840 MB) - Phase 1 intermediate, regenerable from
  the fixed Kaggle snapshot. Too big for GitHub.
- `final_50k_cs_corpus.parquet`, `final_50k_labeled_enriched.parquet` - stale
  files from older notebook versions, not used by the current pipeline.
- `openalex_gate_cache.json.{bak,broken,corrupt}`, `*.salvaged.json`, `_*.txt` -
  cache backups / debug scratch.
- `chroma_db` folder, 800MB plus, too big to push github, this can easily run in phase 5

---

## How to re-run things

The notebook caches aggressively, so re-running is cheap:
- **( SUGGEST DO NOT RE-RUN THIS PHASE 1 )Re-run Phase 1:** you need to download the Kaggle snapshot (the notebook does this via `kagglehub`) and rebuild the candidate pool. With `openalex_gate_cache.json` present, the OpenAlex gate is served from cache, so the slow part is skipped.
- **Re-run Phase 2->4 from scratch:** keep `final_50k_with_institution.parquet`,
  run the Phase 2, 3, 4 cells in order. No network needed (Phase 3 uses the
  OpenAlex institution already carried in the Phase 1 output; it only
  canonicalizes against the local QS Top-50 list).
- **Re-run phase 5 from scratch** `final_50k_labeled.parquet` is the only file needed in this phase, just run the cell one by one in order.(This is the phase building chroma_db)
- **Re-run phase 6-10 from scratch** `final_50k_labeled.parquet` is the only file needed in phase 6, for all the cells in phase 6 and the other phase 7 - 10 just run the cell one by one in order.


Note: Phase 1 reads an `openalex api.txt` config (mailto / polite-pool email)
from the project parent folder. It's gitignored - set up your own.

---

## Pipeline overview (Phase 1-5)

Fixed data source: Kaggle `Cornell-University/arxiv` **version 289** snapshot
(3,066,190 records). Target corpus size: **50,000**. Seed: 42.

### Phase 1 - CS corpus acquisition + OpenAlex-gated 50K selection
Idea: start from all arXiv, narrow to CS, then only keep papers we can attach a
real institution to (so the fairness labels in Phase 3 are meaningful).

- Filter `categories LIKE 'cs.%'` -> 955,380 CS records.
- Drop missing title/abstract, dedupe on canonical arXiv id (strip `vN` suffix,
  keep latest `update_date`) -> candidate pool of **955,379**.
- "Gate" the pool with batched OpenAlex DOI lookups (50 DOIs/request, retry on
  transient errors). A paper is selectable only if OpenAlex returns a primary
  institution. Resolution strategy priority: published DOI -> arXiv DOI ->
  (optional, disabled) title search.
- Walk the pool until **50,000** papers with institutions are selected.
- Result: `final_50k_with_institution.parquet`. Institution coverage = 100% by
  construction. Cache stats: 50,000 ok, ~308K permanent fails skipped.

### Phase 2 - Text cleaning + metadata normalization
- Clean title/abstract (strip HTML, control chars, normalize whitespace); keep
  both raw and `cleaned_*` fields.
- Normalize: `year` from `update_date`, `primary_category` = first `cs.*` token,
  stable `document_id` = canonical arXiv id (hash fallback).
- Validation: 50,000 rows, `document_id` unique, no empty cleaned fields. Year
  range 2024-2026. Median abstract ~1,193 chars.
- Result: `final_50k_cleaned.parquet`.

### Phase 3 - Institution/geography enrichment + fairness labels
- Carry the Phase 1 OpenAlex institution/country through.
- Canonicalize institution names against the **QS Top-50 CS** list (handles
  OpenAlex name variants via an alias map), map country -> region.
- Assign `privilege_label`:
  - `privileged` = QS Top-50 CS university
  - `underrepresented` = any other resolved institution
  - `unknown` = no resolvable affiliation (≈0 here, since Phase 1 gates on it)
- Coverage: institution 100%, country 99.7%, region 99.66%.
- Result: `final_50k_labeled.parquet` + `institution_alias_map.csv` +
  `label_distribution.csv`.

### Phase 4 - Corpus statistics + fairness baseline priors
- Summarize distributions (year / category / institution / region / country),
  measure concentration (Top-k share, Gini, Lorenz), render figures.
- `fairness_baseline_priors.json` is the **reference distribution** for the later
  retrieval-fairness analysis (Phase 5+).
- Result: `final_corpus_statistics.csv`, `fairness_baseline_priors.json`,
  `corpus_summary.md`, `data/figures/*.png`.

### Phase 5 - ChromaDB Index Construction and Retrieval Smoke Test
- Select sentence-transformers/all-MiniLM-L6-v2 as the embedding model and generate document embeddings from `final_50k_labeled.parquet` dataset.
- Write the cleaned/labeled 50K corpus into ChromaDB and confirm the index is usable for later RAG experiments.
- Perform 20 (k = 5, 10, 20) smoke-test with 4 queries, and confirm top-10 returns plus metadata/filter functionality.
- Manually assess retrieval relevance based on paper titles and abstracts, and compute retrieval metrics including Precision@k and Recall@k for each query.
- Result: `chroma_db/`, `chroma_ingestion_report.json`, `sample_retrieval_results.md`, `index_config.yaml`.
---

## Phase 1-5: Key results (the 50K corpus)

- 50,000 documents, 5,937 distinct institutions, 137 countries, years 2024-2026.

**Fairness labels (`privilege_label`)**
- privileged: **17.1%** (8,553 papers)
- underrepresented: **82.9%** (41,447 papers)
- unknown: 0%
- privileged : underrepresented ≈ 0.21

**Region distribution**
- Europe 43.1% | North America 27.0% | Asia 25.3% | Oceania 2.3% |
  South America 1.5% | Africa 0.5% | Unknown 0.3%

**Concentration / inequality**
- Top-10 institution share 6.4%, Top-50 share 20.8%
- Institution Gini 0.7525, Country Gini 0.8654

**Top categories:** cs.NA, cs.LG, cs.CV, cs.SY, cs.RO, cs.HC, cs.AI, cs.CL...
**Top countries:** US (11,852), CN (5,672), DE (4,215), FR (3,631), GB (2,579)...
**Top institutions:** CNRS (660), TU Munich (353), Tsinghua (316), ETH Zurich (312), CMU (300)...

(See `data/corpus_summary.md` and `data/figures/` for the full picture.)

---

## Update 2: Roadmap (Phase 6-11)

Phases 1-5 build the labeled corpus, fairness priors, and ChromaDB index. The
phases below cover the RAG-fairness pipeline described in the two project reports
(retrieval audit, MMR mitigation, synthesis faithfulness). Each phase lists its
goal, expected inputs, and expected outputs.

### Phase 6 - RAG retrieval pipeline (LlamaIndex + ChromaDB)
- **Goal:** Wrap the Phase 5 index into a query engine / retriever for the audits.
- **Tasks:** Load the Chroma index into a LlamaIndex `VectorStoreIndex`, keep the
  embedding model consistent with the index, expose a baseline top-k retriever
  with over-fetch headroom (e.g. fetch 20, re-rank to 10).
- **In:** `chroma_db/`, `index_config.yaml`.
- **Out:** retrieval pipeline config + sample retrieval output.
- **Out:** retrieval pipeline config `phase6_sample_retrieval.json` + sample retrieval output `phase6_retrieval_pipeline_config.json`.

### Phase 7 - Bias-audit query benchmark
- **Goal:** Build the 150-query evaluation set.
- **Tasks:** Generate 100 neutral queries across CS subfields from stratified
  abstracts, curate 50 contradictory/debate queries, verify query privilege
  share vs. corpus baseline.
- **In:** `final_50k_labeled.parquet`, retrieval pipeline.
- **In:** `final_50k_labeled.parquet`, `phase6_retrieval_pipeline_config.json`.
- **Out:** `queries.json`, `query_benchmark_report.json`.

### Phase 8 - Experiment A: retrieval parity audit (RQ1)
- **Goal:** Measure institutional homophily in baseline retrieval.
- **Tasks:** Run top-k retrieval over the 150 queries, tag results with
  privilege/region metadata, compute SPD and SRR against the Phase 4 priors,
  break down by query category.
- **In:** `queries.json`, retrieval pipeline, `fairness_baseline_priors.json`.
- **Out:** `naive_retrieval_results.json`, `retrieval_parity_report.json`, figures.

### Phase 9 - Fairness-aware MMR re-ranking + lambda ablation (RQ3)
- **Goal:** Mitigate institutional over/under-representation and quantify the
  fairness-utility tradeoff.
- **Tasks:** Implement a three-signal MMR re-ranker (relevance / diversity /
  fairness), sweep lambda configs, re-rank top-20 to top-10, measure nDCG@10 /
  P@10 / MRR plus unique institutions/countries and SPD/SRR.
- **In:** Phase 8 retrieval results, `fairness_baseline_priors.json`.
- **In:** Phase 8 retrieval results `naive_retrieval_results.json`, `fairness_baseline_priors.json`.
- **Out:** `mmr_reranked_results.json`, `lambda_ablation.csv`, tradeoff figures.

### Phase 10 - Experiment B: generative faithfulness + synthesis bias (RQ2)
- **Goal:** Assess whether LLM synthesis faithfully represents retrieved
  viewpoints and whether it amplifies institutional bias.
- **Tasks:** Generate summaries with citations, stance-classify retrieved docs
  (pro-consensus / dissenting / neutral), evaluate faithfulness (LLM-as-judge /
  RAGAS), compare standard vs. perspective-balanced prompting, report citation
  rate by privilege/region.
- **In:** Phase 8/9 results, `queries.json`.
- **Out:** `synthesis_eval_report.json`, `ragas_scores.csv`, figures.

### Phase 11 - Interactive fairness dashboard
- **Goal:** Expose retrieval and fairness metrics interactively.
- **Tasks:** Build a Streamlit app to issue queries, show retrieved docs with
  institution/geo metadata, and visualize SPD / SRR / diversity live.
- **In:** retrieval pipeline + result artifacts.
- **Out:** dashboard app + screenshots.

---

## Update 2: Results (Phase 6-11), based on 150-query evaluation set:

**Fairness labels (`privilege_label`)**
- **Baseline**
  - privileged: **17.1%** (8,553 papers)
  - underrepresented: **82.9%** (41,447 papers)
  - unknown: 0%
  - privileged : underrepresented ≈ 0.21
- **Observed**
  - privileged: **16.5%** (8,150 papers)
  - underrepresented: **83.5%** (41,750 papers)
  - unknown: 0%
  - privileged : underrepresented ≈ 0.198

**Baseline: Region distribution**
- Europe 43.1% | North America 27.0% | Asia 25.3% | Oceania 2.3% |
  South America 1.5% | Africa 0.5% | Unknown 0.3%

**Observed: Region distribution**
- **Europe 46.2%** | North America 26.3% | **Asia 22.3%** | Oceania 2.8% |
  South America 1.3% | Africa 0.5% | Unknown 0.5%

**SPD:**
- Privilege
  - Underrepresented: 0.00606,
  - Privileged": -0.00606,
- Region
  - Europe: 0.031773, ( 0.03 indicates mild overrepresentation relative to the corpus)
  - North America: -0.006807,
  - Asia: -0.02966, ( -0.03 indicates mild underrepresentation relative to the corpus)
  - Oceania: 0.00494,
  - South America: -0.001567,
  - Africa: -0.0024,
  - Unknown: 0.00156

**SRR:**
- Privilege
  - Underrepresented: 1.007311,
  - Privileged": 0.964574,
- Region
  - Europe: 1.073795, ( 1.07 > 1 indicates mild overrepresentation relative to the corpus)
  - North America: 0.974802,
  - Asia: 0.882609, ( 0.88 < 1 indicates mild underrepresentation relative to the corpus)
  - Oceania: 1.214224, ( 1.2 > 1 indicates mild overrepresentation relative to the corpus)
  - South America: 0.894832,
  - Africa: 0.954198,
  - Unknown: 1.453488

(See `data/update2_output` folders for all the outputs and figures.)
