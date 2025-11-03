from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import yaml

from exp_lab.pipeline import run_experiment
from exp_lab.rag.eval import eval_rag
from exp_lab.rag.indexer import build_index

__all__ = ["run_all", "finetune_from"]


def _load_manifest(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _configure_mlflow(experiment_cfg: dict[str, Any]) -> None:
    mlflow_cfg = experiment_cfg.get("mlflow", {})
    uri = mlflow_cfg.get("uri") or mlflow_cfg.get("tracking_uri")
    if uri:
        mlflow.set_tracking_uri(uri)
    experiment_name = mlflow_cfg.get("experiment_name", "exp-lab/A3")
    if mlflow.active_run() is None:
        mlflow.set_experiment(experiment_name)


def run_all(manifest_path: str) -> None:
    """Execute the full A3 agent flow."""

    cfg = _load_manifest(manifest_path)
    experiment_cfg = cfg.get("experiment", {})
    artifacts_dir = Path(experiment_cfg.get("artifacts", {}).get("base_dir", "out/a3/run_001"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _configure_mlflow(experiment_cfg)

    run_name = f"{experiment_cfg.get('id', 'a3-demo')}:A3"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "milestone": "A3",
                "agent": "orchestrator",
                "phase": "pipeline",
                "rag": "tfidf",
            }
        )

        run_experiment(manifest_path)
        build_index(manifest_path)
        rag_metrics = eval_rag(manifest_path)

        report_path = Path("out") / "a3" / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = ["# A3 Report", "", "## Pipeline Summary", "- DataAgent, TrainAgent, EvalAgent completed via `run_experiment`."]
        lines.append("- RAG index built and evaluated via RAGAgent.")

        if rag_metrics:
            top_k = rag_metrics.get("k", 0)
            lines.extend(
                [
                    "",
                    "## RAG Metrics",
                    f"- Recall@{top_k}: {rag_metrics.get('recall', 0.0):.3f}",
                    f"- MRR@{top_k}: {rag_metrics.get('mrr', 0.0):.3f}",
                    f"- Lexical F1: {rag_metrics.get('lexical_f1', 0.0):.3f}",
                ]
            )

            mlflow.log_metrics(
                {
                    f"rag_recall@{top_k}": rag_metrics.get("recall", 0.0),
                    f"rag_mrr@{top_k}": rag_metrics.get("mrr", 0.0),
                    "rag_lexical_f1": rag_metrics.get("lexical_f1", 0.0),
                }
            )

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        mlflow.log_artifact(report_path)


def finetune_from(manifest_path: str, ckpt: str | None) -> None:
    """Create a one-off manifest for incremental fine-tuning and re-run eval."""

    cfg = _load_manifest(manifest_path)
    experiment_cfg = cfg.setdefault("experiment", {})
    model_cfg = experiment_cfg.setdefault("model", {})
    base_dir = experiment_cfg.get("artifacts", {}).get("base_dir", "out/a3/run_001")
    default_ckpt = Path(base_dir) / "model"

    model_cfg["finetune_from"] = ckpt or str(default_ckpt)
    model_cfg["epochs"] = int(model_cfg.get("epochs", 3)) + 1

    tmp_manifest = Path(manifest_path).with_suffix(".finetune.yaml")
    tmp_manifest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    try:
        run_experiment(str(tmp_manifest))
    finally:
        try:
            tmp_manifest.unlink()
        except FileNotFoundError:
            pass

    eval_rag(manifest_path)
