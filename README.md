# neu-cs6200-team3-fairsearch-arxiv

Working notes for the FairSearch-arXiv project. Phases 1-4 (data pipeline) are
done and their outputs are committed here so the next person can start Phase 5
(ChromaDB index + retrieval-fairness analysis) **without re-running the slow
data-prep steps**. Everything below is a quick reference, not a formal report.

---

## TL;DR for whoever picks this up

- All code lives in `work_notebook.ipynb`, organized as Phase 1 -> Phase 5.
- Phases 1-4 have already been run; the resulting corpus + reports are in `data/`.
- Phase 5 onward is **not done yet** and is left for you.
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
- `raw_data_profile.json`, `corpus_selection_report.json`, `openalex_gate_report.json` (Phase 1)
- `text_cleaning_report.json` (Phase 2)
- `affiliation_match_report.json`, `institution_alias_map.csv`, `label_distribution.csv` (Phase 3)
- `final_corpus_statistics.csv`, `fairness_baseline_priors.json`, `corpus_summary.md`, `figures/*.png` (Phase 4)

These let you sanity-check each phase without opening the parquet files.

### NOT committed (see `.gitignore`)
- `cs_candidate_pool.parquet` (~840 MB) - Phase 1 intermediate, regenerable from
  the fixed Kaggle snapshot. Too big for GitHub.
- `final_50k_cs_corpus.parquet`, `final_50k_labeled_enriched.parquet` - stale
  files from older notebook versions, not used by the current pipeline.
- `openalex_gate_cache.json.{bak,broken,corrupt}`, `*.salvaged.json`, `_*.txt` -
  cache backups / debug scratch.
- `chroma_db`, 800MB plus, can easily run in phase 5

---

## How to re-run things

The notebook caches aggressively, so re-running is cheap:

- **Re-run Phase 2->5 from scratch:** keep `final_50k_with_institution.parquet`,
  run the Phase 2, 3, 4, 5cells in order. No network needed (Phase 3 uses the
  OpenAlex institution already carried in the Phase 1 output; it only
  canonicalizes against the local QS Top-50 list).
- **Re-run Phase 1:** you need to download the Kaggle snapshot (the notebook does
  this via `kagglehub`) and rebuild the candidate pool. With
  `openalex_gate_cache.json` present, the OpenAlex gate is served from cache, so
  the slow part is skipped.

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

## Key results (the 50K corpus)

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
