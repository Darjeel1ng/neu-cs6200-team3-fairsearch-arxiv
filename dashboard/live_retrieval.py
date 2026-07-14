"""Optional live free-text retrieval against the Phase 5/6 ChromaDB index.

Activates only when both:
  1. chroma_db/ is present under the mounted data volume, and
  2. chromadb + llama-index + sentence-transformers are importable
     (installed via requirements-live.txt).

Everything here degrades to `is_available() == False` otherwise, so the
Query Explorer tab can fall back to the precomputed 150 queries without
erroring.
"""
import json
import os

import yaml

from dashboard.data_loader import DATA_ROOT, UPDATE2_DIR

CHROMA_DB_PATH = os.path.join(DATA_ROOT, "chroma_db")
INDEX_CONFIG_PATH = os.path.join(DATA_ROOT, "index_config.yaml")
PIPELINE_CONFIG_PATH = os.path.join(
    UPDATE2_DIR, "phase6_retrieval_pipeline_config.json"
)

_index = None
_unavailable_reason = None
_default_top_k = 20


def _try_init():
    global _index, _unavailable_reason, _default_top_k

    if not os.path.isdir(CHROMA_DB_PATH):
        _unavailable_reason = f"chroma_db not found at {CHROMA_DB_PATH} — showing precomputed queries only."
        return

    if not os.path.exists(INDEX_CONFIG_PATH):
        _unavailable_reason = f"index_config.yaml not found at {INDEX_CONFIG_PATH}."
        return

    with open(INDEX_CONFIG_PATH) as f:
        index_config = yaml.safe_load(f)
    collection_name = index_config["collection_name"]
    embedding_model = index_config["embedding_model"]

    if os.path.exists(PIPELINE_CONFIG_PATH):
        with open(PIPELINE_CONFIG_PATH) as f:
            _default_top_k = json.load(f)["fetch_k"]

    try:
        import chromadb
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except ImportError as exc:
        _unavailable_reason = (
            "Live retrieval deps not installed "
            f"(pip install -r requirements-live.txt): {exc}"
        )
        return

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        _index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
        _unavailable_reason = f"Failed to open chroma_db collection: {exc}"


_try_init()


def is_available() -> bool:
    return _index is not None


def unavailable_reason() -> str:
    return _unavailable_reason or "Live retrieval unavailable."


def run_query(query_text: str, top_k: int | None = None) -> list[dict]:
    if not is_available():
        raise RuntimeError(unavailable_reason())

    retriever = _index.as_retriever(similarity_top_k=top_k or _default_top_k)
    nodes = retriever.retrieve(query_text)
    return [
        {
            "document_id": node.node.node_id,
            "title": node.node.metadata.get("title", ""),
            "score": node.score,
            "privilege_label": node.node.metadata.get("privilege_label", ""),
            "region": node.node.metadata.get("region", ""),
            "country_code": node.node.metadata.get("country_code", ""),
            "institution": node.node.metadata.get("institution", ""),
        }
        for node in nodes
    ]
