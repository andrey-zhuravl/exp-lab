from __future__ import annotations

import glob
import json
import os
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List

import mlflow
from mlflow.entities import Run
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

__all__ = ["build_index", "query", "query_cli"]

_IDX_DIR: str | None = None
_VEC: TfidfVectorizer | None = None
_DOC_IDS: list[str] = []
_DOC_TEXT: dict[str, str] = {}
_MATRIX: np.ndarray | None = None


def _load_cfg(manifest_path: str) -> dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_mlflow(cfg: dict[str, Any]) -> tuple[dict[str, Any], Run | None]:
    experiment_cfg = cfg.get("experiment", {})
    parent = mlflow.active_run()
    if parent is None:
        mlflow_cfg = experiment_cfg.get("mlflow", {})
        uri = mlflow_cfg.get("uri") or mlflow_cfg.get("tracking_uri")
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "exp-lab/A3"))
    return experiment_cfg, parent


@contextmanager
def _mlflow_run(cfg: dict[str, Any], phase: str) -> Iterable[None]:
    experiment_cfg, parent = _ensure_mlflow(cfg)
    run_name = f"{experiment_cfg.get('id', 'a3-demo')}.{phase}"
    with mlflow.start_run(run_name=run_name, nested=parent is not None):
        mlflow.set_tags(
            {
                "milestone": "A3",
                "agent": "rag",
                "phase": phase,
                "rag": "tfidf",
            }
        )
        yield


def _read_corpus(corpus_dir: str) -> List[tuple[str, str]]:
    docs: List[tuple[str, str]] = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        doc_id = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as handle:
            docs.append((doc_id, handle.read()))
    return docs


def build_index(manifest_path: str) -> str:
    cfg = _load_cfg(manifest_path)
    rag_cfg = cfg["experiment"]["rag"]
    corpus_dir = rag_cfg["corpus_dir"]
    index_dir = Path(rag_cfg["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)

    docs = _read_corpus(corpus_dir)
    if not docs:
        raise RuntimeError(f"No documents found in {corpus_dir!r}")

    texts = [content for _, content in docs]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts).toarray()

    ids = [doc_id for doc_id, _ in docs]
    doc_text_map = {doc_id: content for doc_id, content in docs}

    tfidf_path = index_dir / "tfidf.npz"
    np.savez_compressed(tfidf_path, data=matrix)

    with (index_dir / "vectorizer.pkl").open("wb") as handle:
        pickle.dump(vectorizer, handle)

    (index_dir / "doc_ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    (index_dir / "doc_texts.json").write_text(json.dumps(doc_text_map, indent=2), encoding="utf-8")

    with _mlflow_run(cfg, "rag.index"):
        mlflow.log_param("rag_index_dir", str(index_dir))
        mlflow.log_param("rag_num_docs", len(ids))
        mlflow.log_param("rag_vocab_size", len(vectorizer.vocabulary_))
        mlflow.log_artifacts(str(index_dir), artifact_path="rag/index")

    return str(index_dir)


def _load_index(index_dir: str) -> np.ndarray:
    global _IDX_DIR, _VEC, _DOC_IDS, _DOC_TEXT, _MATRIX

    if _IDX_DIR == index_dir and _VEC is not None and _MATRIX is not None:
        return _MATRIX

    import pickle

    vectorizer_path = Path(index_dir) / "vectorizer.pkl"
    ids_path = Path(index_dir) / "doc_ids.json"
    texts_path = Path(index_dir) / "doc_texts.json"
    matrix_path = Path(index_dir) / "tfidf.npz"

    with vectorizer_path.open("rb") as handle:
        _VEC = pickle.load(handle)

    _DOC_IDS = json.loads(ids_path.read_text(encoding="utf-8"))
    _DOC_TEXT = json.loads(texts_path.read_text(encoding="utf-8"))
    _MATRIX = np.load(matrix_path)["data"]
    _IDX_DIR = index_dir
    return _MATRIX


def _compose_answer(q: str, top_docs: List[Dict[str, Any]]) -> str:
    if not top_docs:
        return ""
    top_doc = top_docs[0]
    text = _DOC_TEXT.get(top_doc["doc_id"], "").strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    if not first_line:
        first_line = text[:200].strip()
    return f"Based on {top_doc['doc_id']}: {first_line}"


def query(index_dir: str, q: str, k: int = 3) -> dict[str, Any]:
    matrix = _load_index(index_dir)
    if _VEC is None:
        raise RuntimeError("Index not loaded")

    query_vec = _VEC.transform([q]).toarray()
    scores = cosine_similarity(query_vec, matrix)[0]
    order = np.argsort(scores)[::-1][:k]

    top_docs = [
        {"doc_id": _DOC_IDS[i], "score": float(scores[i]), "snippet": _DOC_TEXT.get(_DOC_IDS[i], "")[:200]}
        for i in order
    ]

    return {"q": q, "top_k": top_docs, "answer": _compose_answer(q, top_docs)}


def query_cli(manifest_path: str, q: str, k: int = 3) -> None:
    cfg = _load_cfg(manifest_path)
    index_dir = cfg["experiment"]["rag"]["index_dir"]
    result = query(index_dir, q, k)
    print(json.dumps(result, indent=2))
