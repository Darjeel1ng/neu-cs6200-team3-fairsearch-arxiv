# FairSearch-arXiv Dashboard (Phase 11)

Streamlit app that reads the Phase 6-9 result artifacts from `data/` and
shows retrieval + fairness metrics interactively. Read-only against those
files — it doesn't modify or regenerate anything.

## Prerequisites

- Get the latest data before starting: `git pull` from the repo root. (Only
  `data/*.parquet` and `openalex_gate_cache.json` are Git LFS; everything
  this app reads is plain git, so a normal `git pull` is enough.)
- Docker + Docker Compose (recommended), **or** Python 3.12 if running
  without Docker.

## Run with Docker (recommended)

From the **repo root** (not this `dashboard/` directory):

```bash
docker compose up --build
```

Open http://localhost:8501.

To pick up newer data later, just `git pull` again and re-run
`docker compose up` — no rebuild needed, since `data/` is bind-mounted, not
baked into the image.

### Live free-text querying (optional)

The default image is lightweight and only replays the 150 precomputed
queries. To also enable live free-text retrieval against `data/chroma_db`
(if you have it locally):

```bash
docker compose --profile live up dashboard-live --build
```

If `data/chroma_db` isn't present, the app falls back to precomputed queries
automatically with an explanation in the UI — this isn't an error state.

## Run without Docker

From the **repo root**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

DASHBOARD_DATA_ROOT="$PWD/data" PYTHONPATH="$PWD" streamlit run dashboard/app.py
```

Open http://localhost:8501.

For live querying, also `pip install -r requirements-live.txt` first.

## Layout

| File | Purpose |
|---|---|
| `app.py` | Entry point, tab wiring |
| `data_loader.py` | Cached readers for `data/update2_output/*` + `fairness_baseline_priors.json` |
| `live_retrieval.py` | Optional live query against `chroma_db`; no-ops if unavailable |
| `tabs/query_explorer.py` | Naive vs. all 5 MMR lambda configs, side by side, for a selected query |
| `tabs/fairness_metrics.py` | SPD / SRR, baseline vs. observed distributions |
| `tabs/lambda_tradeoff.py` | Phase 9 lambda ablation: utility vs. fairness tradeoff |
| `tabs/synthesis_placeholder.py` | Placeholder for Phase 10 metrics (not yet implemented) |

## Troubleshooting

- **"Could not find data/update2_output"** — you're missing the `data/`
  volume mount, or ran `streamlit run` from the wrong directory without
  setting `DASHBOARD_DATA_ROOT`. Run from the repo root as shown above.
- **Data looks stale** — `git pull`, then restart the container/process. No
  rebuild is needed for data-only changes.
- **`ModuleNotFoundError: No module named 'dashboard'`** — the image is stale
  relative to the `Dockerfile`. Rebuild with `docker compose up --build`
  (unlike data changes, changes to the `Dockerfile`/`requirements*.txt`
  *do* require a rebuild). If running without Docker, make sure
  `PYTHONPATH` is set to the repo root as shown above, not `dashboard/`.
