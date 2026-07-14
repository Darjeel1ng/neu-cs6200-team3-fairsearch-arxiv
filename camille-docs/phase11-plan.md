# Phase 11 — Containerized Interactive Fairness Dashboard

## Context

Phase 11 (`Streamlit dashboard exposing retrieval + fairness metrics live`) has
no dependency on Phase 10 per the README spec — its only inputs are Phase
6/8/9 artifacts (retrieval pipeline config, `naive_retrieval_results.json`,
`mmr_reranked_results.json`, `lambda_ablation.csv`, `fairness_baseline_priors.json`),
all already committed. Since a teammate is currently fixing data issues
upstream in Phases 6-9, and the pipeline/data will keep changing on `main`
independent of this work, we containerize the app so:

1. The dashboard has a pinned, reproducible environment (today there's no
   `requirements.txt` at all — everything so far lives only in
   `work_notebook.ipynb`).
2. Getting the latest (fixed) data is just `git pull` + `docker compose up` —
   no image rebuild, no notebook changes, no separate data-download step.
   (Only `data/*.parquet` and `openalex_gate_cache.json` are Git LFS per
   `.gitattributes`; everything the dashboard reads is plain git.)

The container bind-mounts the repo's `data/` directory read-only rather than
baking data into the image, so it always reflects whatever is on disk after a
`git pull`.

## Decisions from scoping discussion

- **Lambda configs:** side-by-side compare — for a selected query, show the
  naive baseline and all 5 MMR configs (`rel_0.80_div_0.10_fair_0.10` ...
  `rel_0.40_div_0.30_fair_0.30`) together, not a single dropdown.
- **Phase 10 tie-in:** add a disabled/placeholder "Synthesis Faithfulness
  (coming soon)" tab now, so the tab layout doesn't need rework later.
- **Query mode:** build for both, matching the README's "In: retrieval
  pipeline + result artifacts." The 150 precomputed queries (replay mode) are
  the default and always-available path. A free-text live-query box is also
  wired in, but only activates if `chroma_db/` is present under the mounted
  `data/` volume and the live-retrieval deps are installed; otherwise it shows
  a disabled state explaining why ("chroma_db not found — showing precomputed
  queries only"). This matches the spec without blocking on chroma_db's
  current unavailability (800MB, gitignored, not present locally, and the
  retrieval pipeline it depends on is mid-fix).

## File layout (new, root of repo)

```
dashboard/
  app.py                # entrypoint: st.set_page_config, sidebar nav, tab wiring
  data_loader.py         # cached loaders for all data/update2_output/*.json|csv + fairness_baseline_priors.json
  live_retrieval.py      # optional: detects chroma_db + deps, exposes run_query() or None
  tabs/
    query_explorer.py    # select a query (or free-text if live mode active) -> naive vs MMR docs table
    fairness_metrics.py  # SPD/SRR charts from retrieval_parity_report.json, baseline vs observed
    lambda_tradeoff.py    # lambda_ablation.csv -> nDCG/P@10/MRR vs SPD/SRR tradeoff charts (reuse existing tradeoff figures' data, not the PNGs)
    synthesis_placeholder.py  # disabled tab, static "coming soon" message
requirements.txt          # streamlit, pandas, plotly, pyyaml
requirements-live.txt      # optional extra: chromadb, llama-index, sentence-transformers (only needed for live-query mode)
Dockerfile
docker-compose.yml
.dockerignore
```

## Data loading

`data_loader.py` reads directly from the mounted path (e.g. `/app/data`,
mapped via compose from host `./data`):
- `update2_output/queries.json` — 150 queries for the picker
- `update2_output/naive_retrieval_results.json` — baseline per-query docs
- `update2_output/mmr_reranked_results.json` — dict keyed by lambda config name, each a list of per-query docs (has `relevance_score`/`fairness_score` per doc)
- `update2_output/lambda_ablation.csv` — nDCG@10/P@10/MRR/SPD/SRR/unique institutions per config, for the tradeoff tab
- `update2_output/retrieval_parity_report.json` — baseline vs observed privilege/region distributions, SPD/SRR, category breakdown, for the fairness metrics tab
- `fairness_baseline_priors.json` — corpus-level reference distribution

Use `st.cache_data` on load functions keyed by file mtime so a `git pull`
that updates these files is picked up without restarting the container.

## Dockerfile / compose

- `Dockerfile`: slim Python base, `pip install -r requirements.txt`, copy
  `dashboard/`, `CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]`.
  Do **not** copy `data/` into the image — it's bind-mounted at runtime.
- `docker-compose.yml`: one `dashboard` service, `volumes: ["./data:/app/data:ro"]`,
  `ports: ["8501:8501"]`. Add a build arg / optional second service or
  `requirements-live.txt` install step, gated so the default `docker compose up`
  stays lightweight; live mode can be a documented opt-in (`docker compose --profile live up`
  or a separate compose override file) if we don't want to force the heavy
  ML deps into the default image.
- Refresh workflow to document in README: `git pull` (host) → `docker compose up`
  (add `--build` only if `requirements.txt` itself changed).

## Verification

1. `docker compose up --build`, open `localhost:8501`.
2. Query Explorer tab: pick a query from the 150, confirm naive + all 5 MMR
   configs render side-by-side with matching `document_id`/title/privilege
   fields from the JSON artifacts.
3. Fairness Metrics tab: confirm baseline vs observed privilege/region and
   SPD/SRR numbers match `retrieval_parity_report.json` values directly (spot
   check a couple of numbers against the raw JSON).
4. Lambda Tradeoff tab: confirm the 5 rows of `lambda_ablation.csv` are all
   plotted and hovering/selecting shows correct nDCG@10/SPD/SRR values.
5. Synthesis Faithfulness tab: confirm it renders as visibly disabled/"coming
   soon", no broken links or errors.
6. Stop the container, edit one value in a local copy of
   `retrieval_parity_report.json` (or simulate a `git pull` by touching the
   file), restart, confirm the dashboard reflects the change — proves the
   bind-mount + no-rebuild refresh workflow actually works.
7. Live-query path: without chroma_db present, confirm the free-text box is
   visibly disabled with the explanatory message rather than erroring.
