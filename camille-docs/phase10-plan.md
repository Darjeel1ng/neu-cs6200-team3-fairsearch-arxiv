# Phase 10 — Generative Faithfulness + Synthesis Bias (RQ2)

## Context

Phases 1-9 are complete and live entirely in `work_notebook.ipynb`, each as a
markdown header + a sequence of code cells (see Phase 8/9 for the pattern:
load inputs → compute → print → save JSON/figures). Phase 10 continues that
pattern. Its goal per the README: check whether LLM synthesis over retrieved
documents faithfully represents them, and whether the generation step itself
amplifies the institutional bias already measured in Phases 8-9.

Verified before planning:
- `data/update2_output/queries.json` — 150 queries: 100 "neutral" (have a
  source `document_id` + `privilege_label`) and 50 "debate" (topic-only, both
  fields `null` by design — not a data bug).
- `data/update2_output/naive_retrieval_results.json` — 150 queries × 20 docs
  (Phase 6 fetch_k over-fetch), every doc has `privilege_label`/`region`
  populated, no nulls.
- `data/update2_output/mmr_reranked_results.json` — 5 lambda configs × 150
  queries × 10 docs, fully populated.

Decisions locked in with the user:
- **Credentials**: personal `ANTHROPIC_API_KEY` exported in the shell for one
  notebook run, never written to a file or committed. Outputs
  (`synthesis_eval_report.json`, `ragas_scores.csv`) are committed so nobody
  else needs a key to use the results — same pattern as Phase 1's
  `openalex_gate_cache.json`.
- **Scope**: a stratified subsample of the 150 queries (~40), not all 150, to
  bound one-time API cost/runtime. Stratify to preserve the corpus's
  neutral:debate ratio (100:50 → keep ~2:1 in the sample).
- **Model**: `claude-opus-4-8` (skill default) for synthesis generation and
  for stance classification.
- **Faithfulness judge**: RAGAS's `faithfulness` metric (not a hand-rolled
  LLM-as-judge prompt), backed by Claude via `langchain-anthropic`.
- **Retrieval condition held constant**: use the Phase 6 naive top-10 (first
  10 of the 20 over-fetched docs per query) as the retrieved context for
  every synthesis call. Holding retrieval fixed isolates the generation step
  as the only variable when comparing standard vs. perspective-balanced
  prompting — otherwise a bias difference could come from MMR's diversity
  mechanism rather than the LLM's citation behavior, which would confound
  RQ2 (does *synthesis* amplify bias) with RQ3 (already answered in Phase 9).

## Pipeline design

New "Phase 10" section in `work_notebook.ipynb`, following the existing
banner-comment cell style. Sub-steps, each its own cell block:

**1. Setup**
```python
import os, anthropic
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env; raise clearly if unset
```
New deps for this phase only: `%pip install -q anthropic ragas langchain-anthropic`
(mirrors the existing `%pip install -q chromadb sentence-transformers ...`
cell from Phase 5/6 — first time `anthropic`/`ragas` are used in this repo).

**2. Stratified query subsample**
Sample ~27 neutral + ~13 debate queries (proportional to the 100:50 corpus
split) from `queries.json`, fixed `random_state` for reproducibility. Persist
the sampled `query_id` list at the top of `synthesis_eval_report.json` so the
subsample is auditable.

**3. Build retrieval context per sampled query**
For each sampled query, take its 10 highest-ranked docs from
`naive_retrieval_results.json` (already loaded the same way Phase 8/9 does).
Format as a numbered context block: `[doc_1] <title> — <abstract>` etc., and
keep a `{doc_n: {document_id, privilege_label, region}}` lookup for later
citation-rate computation.

**4. Generate synthesis — standard vs. perspective-balanced**
Two prompt variants per sampled query, both constrained to structured output
via `client.messages.parse()` (Pydantic model: `summary: str`,
`citations: list[{doc_id: str, claim: str}]`) so citations are extracted
reliably instead of regex-parsing bracket notation:
- **Standard**: "Synthesize an answer to the query using only the provided
  documents; cite the specific doc for each claim."
- **Perspective-balanced**: same, plus an explicit instruction to draw from
  and cite a representative range of the provided documents rather than
  defaulting to the most prominent/familiar-sounding institutions.

This is the direct RQ2 experiment: same query, same retrieved docs, only the
prompt changes.

**5. Stance classification**
One batched Claude call per query (not per doc) classifying all 10 retrieved
docs as `pro-consensus | dissenting | neutral` relative to the query, via
structured output — cheaper than 10 separate calls.

**6. Faithfulness scoring (RAGAS)**
For each (query, prompting variant) pair, run RAGAS's `faithfulness` metric
with `{question: query, answer: summary, contexts: [doc abstracts]}`, LLM
backend = `ChatAnthropic(model="claude-opus-4-8")` via `langchain-anthropic`.
Produces one faithfulness score per query per prompting variant.

**7. Citation rate by privilege/region**
For each generated `citations` list, look up each cited `doc_id`'s
`privilege_label`/`region` from step 3's lookup. Compute, per prompting
variant: % of citations pointing to privileged vs. underrepresented docs, and
by region — then compare against the *retrieved* privilege mix (what was
available to cite) to see whether generation over/under-cites privileged
sources relative to what was retrieved. This is the core bias-amplification
metric for RQ2, and the standard-vs-balanced comparison shows whether
perspective-balanced prompting measurably closes any gap.

**8. Aggregate + save outputs**
- `data/update2_output/synthesis_eval_report.json` — sampled query IDs,
  citation-rate-by-privilege/region per prompting variant, stance
  distribution, faithfulness score summary (mean/median per variant).
- `data/update2_output/ragas_scores.csv` — one row per (query_id, prompting
  variant): faithfulness score.
- Figures in `data/update2_output/figures/`: citation-rate-by-privilege bar
  chart (standard vs. balanced), faithfulness score distribution — same
  `figure_dir` pattern Phase 8/9 already use.

## Files touched

- `work_notebook.ipynb` — new Phase 10 section only; no changes to earlier
  phases.
- New committed outputs under `data/update2_output/`: `synthesis_eval_report.json`,
  `ragas_scores.csv`, two new figures.
- No dashboard changes in this pass — the Phase 11 "Synthesis Faithfulness
  (coming soon)" placeholder tab stays as-is; wiring it to these new outputs
  is a follow-up once Phase 10 lands, not part of this plan.

## Verification

1. Run the new Phase 10 cells top-to-bottom with `ANTHROPIC_API_KEY` set in
   the shell; confirm no missing-key error and that the stratified sample
   prints ~27/13 neutral/debate.
2. Spot-check 2-3 synthesis outputs manually: does `citations[].doc_id`
   actually match a `doc_id` present in that query's top-10 context (no
   hallucinated citations to documents not in context)?
3. Confirm `ragas_scores.csv` has exactly `2 × sample_size` rows (one per
   query per prompting variant) with scores in `[0, 1]`.
4. Confirm `synthesis_eval_report.json` parses as valid JSON and its
   citation-rate percentages per variant sum sensibly (privileged% +
   underrepresented% ≈ 100% modulo any uncited-but-retrieved docs).
5. Re-run the whole section a second time (fresh kernel) with no code
   changes to confirm it's reproducible given the fixed sampling seed, and
   that a `git diff` on the sample-query-ID list is empty.
