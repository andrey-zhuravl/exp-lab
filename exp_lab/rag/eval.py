from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import mlflow

from .indexer import _load_cfg, _mlflow_run, _read_corpus, query

__all__ = ["eval_rag"]


def _load_pairs(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _token_set(text: str) -> set[str]:
    return {token.strip().lower() for token in text.split() if token.strip()}


def _lexical_f1(reference: str, candidate: str) -> float:
    ref_tokens = _token_set(reference)
    cand_tokens = _token_set(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    intersection = ref_tokens & cand_tokens
    if not intersection:
        return 0.0
    precision = len(intersection) / len(cand_tokens)
    recall = len(intersection) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def eval_rag(manifest_path: str) -> dict[str, Any]:
    cfg = _load_cfg(manifest_path)
    rag_cfg = cfg["experiment"]["rag"]
    index_dir = rag_cfg["index_dir"]
    pairs_path = rag_cfg["eval_pairs"]
    k = rag_cfg.get("top_k", 3)

    doc_text_map = {doc_id: text for doc_id, text in _read_corpus(rag_cfg["corpus_dir"])}

    total = 0
    hits = 0
    mrr = 0.0
    lexical_scores: list[float] = []
    rows: list[dict[str, Any]] = []

    for pair in _load_pairs(pairs_path):
        total += 1
        question = pair["q"]
        expected_doc = pair["doc_id"]

        result = query(index_dir, question, k)
        top_docs = result["top_k"]
        predicted_ids = [item["doc_id"] for item in top_docs]

        if expected_doc in predicted_ids:
            hits += 1
            rank = predicted_ids.index(expected_doc) + 1
            mrr += 1.0 / rank

        predicted_doc_text = doc_text_map.get(predicted_ids[0], "") if predicted_ids else ""
        lexical_scores.append(_lexical_f1(doc_text_map.get(expected_doc, ""), predicted_doc_text))

        rows.append(
            {
                "q": question,
                "gt": expected_doc,
                "pred": predicted_ids,
                "answer": result["answer"],
            }
        )

    recall = hits / total if total else 0.0
    mrr_k = mrr / total if total else 0.0
    lexical_f1 = sum(lexical_scores) / len(lexical_scores) if lexical_scores else 0.0

    out_dir = Path("out") / "a3"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "rag_report.md"
    metrics = {"recall": recall, "mrr": mrr_k, "lexical_f1": lexical_f1, "k": k, "rows": rows}

    report_lines = [
        "# RAG Report",
        "",
        f"- Recall@{k}: {recall:.3f}",
        f"- MRR@{k}: {mrr_k:.3f}",
        f"- Lexical F1: {lexical_f1:.3f}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    metrics_path = out_dir / "rag_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with _mlflow_run(cfg, "rag.eval"):
        mlflow.log_metric("rag_recall", recall)
        mlflow.log_metric("rag_mrr", mrr_k)
        mlflow.log_metric("rag_lexical_f1", lexical_f1)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(metrics_path)

    print("RAG metrics:", {"recall": recall, "mrr": mrr_k, "lexical_f1": lexical_f1})

    return {"recall": recall, "mrr": mrr_k, "lexical_f1": lexical_f1, "k": k}
