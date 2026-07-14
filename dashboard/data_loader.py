"""Cached readers for the Phase 6-9 result artifacts under data/update2_output/.

All loaders are keyed on each file's mtime so a `git pull` that updates the
underlying JSON/CSV is picked up without restarting the Streamlit process.
"""
import json
import os

import pandas as pd
import streamlit as st

DATA_ROOT = os.environ.get("DASHBOARD_DATA_ROOT", "/app/data")
UPDATE2_DIR = os.path.join(DATA_ROOT, "update2_output")


def _mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else -1.0


@st.cache_data
def load_queries(_mtime_key: float) -> list[dict]:
    with open(os.path.join(UPDATE2_DIR, "queries.json")) as f:
        return json.load(f)


@st.cache_data
def load_naive_results(_mtime_key: float) -> pd.DataFrame:
    with open(os.path.join(UPDATE2_DIR, "naive_retrieval_results.json")) as f:
        return pd.DataFrame(json.load(f))


@st.cache_data
def load_mmr_results(_mtime_key: float) -> dict[str, pd.DataFrame]:
    with open(os.path.join(UPDATE2_DIR, "mmr_reranked_results.json")) as f:
        raw = json.load(f)
    return {config: pd.DataFrame(rows) for config, rows in raw.items()}


@st.cache_data
def load_lambda_ablation(_mtime_key: float) -> pd.DataFrame:
    return pd.read_csv(os.path.join(UPDATE2_DIR, "lambda_ablation.csv"))


@st.cache_data
def load_retrieval_parity_report(_mtime_key: float) -> dict:
    with open(os.path.join(UPDATE2_DIR, "retrieval_parity_report.json")) as f:
        return json.load(f)


@st.cache_data
def load_fairness_baseline_priors(_mtime_key: float) -> dict:
    with open(os.path.join(DATA_ROOT, "fairness_baseline_priors.json")) as f:
        return json.load(f)


def get_queries() -> list[dict]:
    return load_queries(_mtime(os.path.join(UPDATE2_DIR, "queries.json")))


def get_naive_results() -> pd.DataFrame:
    path = os.path.join(UPDATE2_DIR, "naive_retrieval_results.json")
    return load_naive_results(_mtime(path))


def get_mmr_results() -> dict[str, pd.DataFrame]:
    path = os.path.join(UPDATE2_DIR, "mmr_reranked_results.json")
    return load_mmr_results(_mtime(path))


def get_lambda_ablation() -> pd.DataFrame:
    path = os.path.join(UPDATE2_DIR, "lambda_ablation.csv")
    return load_lambda_ablation(_mtime(path))


def get_retrieval_parity_report() -> dict:
    path = os.path.join(UPDATE2_DIR, "retrieval_parity_report.json")
    return load_retrieval_parity_report(_mtime(path))


def get_fairness_baseline_priors() -> dict:
    path = os.path.join(DATA_ROOT, "fairness_baseline_priors.json")
    return load_fairness_baseline_priors(_mtime(path))


def data_root_exists() -> bool:
    return os.path.isdir(UPDATE2_DIR)
