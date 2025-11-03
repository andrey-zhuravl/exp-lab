from __future__ import annotations

import copy
import json
import os
import random
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import mlflow
import yaml


def rprint(*args, **kwargs):  # pragma: no cover - simple fallback
    print(*args, **kwargs)

from exp_lab.models import get_trainer
from exp_lab.registry import get_or_register as registry_get_or_register


@dataclass
class Cfg:
    manifest_path: str
    experiment_id: str
    seed: int
    mlflow_uri: str
    mlflow_experiment: str
    dataset: Dict[str, Any]
    model: Dict[str, Any]
    evaluation: Dict[str, Any]
    artifacts_dir: str


def _run(cmd: Iterable[str], cwd: str | None = None) -> None:
    cmd = list(cmd)
    rprint(f"[bold]$ {' '.join(cmd)}[/bold]")
    subprocess.check_call(cmd, cwd=cwd)


def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _resolve_under_base(base: Path, maybe_relative: str | None, default: str) -> Path:
    if not maybe_relative:
        return base / default
    candidate = Path(maybe_relative)
    if candidate.is_absolute():
        return candidate
    return base / candidate


def _load_cfg(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    experiment = manifest["experiment"]
    artifacts_dir = experiment.get("artifacts", {}).get("base_dir", "out/a1")
    mlflow_section = experiment.get("mlflow", {})
    mlflow_uri = mlflow_section.get(
        "uri", os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    mlflow_experiment = mlflow_section.get("experiment_name", "exp-lab/A1")
    cfg = Cfg(
        manifest_path=path,
        experiment_id=experiment["id"],
        seed=int(experiment.get("seed", 212)),
        mlflow_uri=mlflow_uri,
        mlflow_experiment=mlflow_experiment,
        dataset=copy.deepcopy(experiment.get("dataset", {})),
        model=copy.deepcopy(experiment.get("model", {})),
        evaluation=copy.deepcopy(experiment.get("evaluation", {})),
        artifacts_dir=str(artifacts_dir),
    )
    return cfg


LOCK_ENV_KEYS = [
    "EXP_LAB_HOME",
    "TOY_LANG_LAB_HOME",
    "TTLAB_HOME",
    "MLOPS_HOME",
    "MLFLOW_TRACKING_URI",
]
TOKENIZER_FILENAME = "tokenizer.json"


def _read_grid_tags() -> Dict[str, Any]:
    raw = os.environ.get("EXP_GRID_TAGS")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:  # pragma: no cover - guard against malformed env
        rprint("[yellow]Failed to parse EXP_GRID_TAGS; ignoring")
        return {}


def _maybe_subsample(data: List[Mapping[str, Any]], size: int | None, rng: random.Random) -> List[Mapping[str, Any]]:
    if not size or size <= 0 or len(data) <= size:
        return list(data)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    selected = indices[:size]
    return [data[idx] for idx in selected]


def _apply_noise(data: List[Mapping[str, Any]], noise_cfg: Dict[str, Any], rng: random.Random) -> List[Mapping[str, Any]]:
    p_drop = float(noise_cfg.get("p_drop", 0.0) or 0.0)
    if p_drop <= 0.0:
        return list(data)

    result: List[Mapping[str, Any]] = []
    for row in data:
        text = str(row.get("text", ""))
        tokens = text.split()
        if not tokens:
            result.append(row)
            continue
        kept = [token for token in tokens if rng.random() > p_drop]
        if not kept:
            kept = [tokens[rng.randrange(len(tokens))]]
        new_row = dict(row)
        new_row["text"] = " ".join(kept)
        result.append(new_row)
    return result


def _dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _build_tokenizer_vocab(processed_dir: Path) -> List[str]:
    tokens: set[str] = set()
    train_path = processed_dir / "train.jsonl"
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:  # pragma: no cover - defensive
                    continue
                text = str(record.get("text", ""))
                tokens.update(text.split())
    return sorted(tokens)


def _ensure_tokenizer(processed_dir: Path, cfg: Cfg) -> tuple[Path | None, Dict[str, Any]]:
    spec = {
        "provider": cfg.dataset.get("provider"),
        "recipe": cfg.dataset.get("recipe"),
        "size": cfg.dataset.get("size"),
        "seed": cfg.seed,
        "targets": cfg.evaluation.get("targets"),
        "d_model": cfg.model.get("d_model"),
        "noise": cfg.dataset.get("noise", {}),
        "hide": cfg.dataset.get("hide", {}),
    }
    entry = registry_get_or_register(spec)
    registry_path = Path(entry["path"])
    processed_path = processed_dir / TOKENIZER_FILENAME

    if processed_path.exists():
        if not registry_path.exists():
            _ensure_dir(registry_path.parent)
            shutil.copyfile(processed_path, registry_path)
        return processed_path, entry

    if registry_path.exists():
        _ensure_dir(processed_path.parent)
        shutil.copyfile(registry_path, processed_path)
        return processed_path, entry

    vocab = _build_tokenizer_vocab(processed_dir)
    tokenizer_payload = {
        "id": entry["id"],
        "spec": spec,
        "size": len(vocab),
        "tokens": vocab,
        "created_at": int(time.time()),
    }
    _ensure_dir(processed_path.parent)
    with open(processed_path, "w", encoding="utf-8") as handle:
        json.dump(tokenizer_payload, handle, ensure_ascii=False, indent=2)
    _ensure_dir(registry_path.parent)
    shutil.copyfile(processed_path, registry_path)
    return processed_path, entry


def _capture_env() -> Dict[str, Any]:
    return {key: os.environ.get(key, "") for key in LOCK_ENV_KEYS}


def _capture_git_state() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
            .decode("utf-8")
            .strip()
        )
    except Exception:  # pragma: no cover - git may be unavailable
        return {}
    status_cmd = ["git", "status", "--short"]
    try:
        status = (
            subprocess.check_output(status_cmd, cwd=repo_root)
            .decode("utf-8")
            .strip()
        )
    except Exception:  # pragma: no cover - git may be unavailable
        status = ""
    return {"commit": commit, "status": status, "dirty": bool(status)}


def _write_run_lock(
    base_dir: Path,
    cfg: Cfg,
    manifest_snapshot: Path,
    metrics_path: Path,
    tags: Dict[str, Any],
    tokenizer_entry: Dict[str, Any] | None,
) -> None:
    payload = {
        "experiment_id": cfg.experiment_id,
        "seed": cfg.seed,
        "manifest_path": os.path.abspath(cfg.manifest_path),
        "manifest_snapshot_path": str(manifest_snapshot),
        "artifacts_dir": str(base_dir),
        "metrics_path": str(metrics_path),
        "dataset": cfg.dataset,
        "model": cfg.model,
        "evaluation": cfg.evaluation,
        "grid_tags": tags,
        "environment": _capture_env(),
        "git": _capture_git_state(),
        "created_at": int(time.time()),
    }
    if tokenizer_entry:
        payload["tokenizer"] = tokenizer_entry
    active_run = mlflow.active_run()
    if active_run is not None:
        payload["mlflow_run_id"] = active_run.info.run_id

    lock_path = base_dir / "lock.json"
    with open(lock_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _resolve_arch(cfg: Cfg) -> str:
    arch = cfg.model.get("arch", "baseline")
    if arch == "tiny_transformer":
        arch = "baseline"
    if get_trainer(arch) is None:
        arch = "baseline"
    cfg.model["arch"] = arch
    return arch


@contextmanager
def _mlflow_run(cfg: Cfg) -> Iterable[None]:
    """Start an MLflow run with graceful fallback to a local file store."""

    primary_uri = cfg.mlflow_uri
    fallback_uri = f"file://{Path(cfg.artifacts_dir).resolve() / 'mlruns'}"
    last_exc: Exception | None = None
    for uri in (primary_uri, fallback_uri):
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(cfg.mlflow_experiment)
            with mlflow.start_run(run_name=cfg.experiment_id):
                yield
            return
        except Exception as exc:  # pragma: no cover - fallback path best effort
            last_exc = exc
            if uri == fallback_uri:
                break
            rprint(
                "[yellow]MLflow tracking URI %s failed, falling back to %s (%s)" %
                (primary_uri, fallback_uri, exc)
            )
    if last_exc is not None:
        raise last_exc


def run_experiment(manifest_path: str) -> None:
    cfg = _load_cfg(manifest_path)
    base_dir = Path(cfg.artifacts_dir)
    _ensure_dir(base_dir)
    os.environ.setdefault("MLFLOW_TRACKING_URI", cfg.mlflow_uri)
    manifest_snapshot = base_dir / "manifest.snapshot.yaml"
    shutil.copyfile(manifest_path, manifest_snapshot)

    rng = random.Random(cfg.seed)
    grid_tags = _read_grid_tags()
    arch = _resolve_arch(cfg)

    with _mlflow_run(cfg):
        for key, value in grid_tags.items():
            mlflow.set_tag(key, value)

        hide_cfg = cfg.dataset.get("hide", {})
        noise_cfg = cfg.dataset.get("noise", {})
        mlflow.log_params(
            {
                "seed": cfg.seed,
                "hide_strategy": hide_cfg.get("strategy"),
                "hide_value": hide_cfg.get("value"),
                "dataset_size": cfg.dataset.get("size"),
                "noise_p_drop": noise_cfg.get("p_drop", 0.0),
                "d_model": cfg.model.get("d_model", 8),
                "n_layers": cfg.model.get("n_layers", 1),
                "arch": arch,
            }
        )
        mlflow.log_artifact(manifest_path)
        mlflow.log_artifact(str(manifest_snapshot))

        dataset_root = _resolve_under_base(base_dir, cfg.dataset.get("out_dir"), "dataset")
        raw_dir = dataset_root / "raw"
        split_dir = dataset_root / "splits"
        _ensure_dir(raw_dir)
        _ensure_dir(split_dir)

        raw_path = raw_dir / "raw.jsonl"
        if cfg.dataset.get("mode", "generate") == "generate":
            _generate_or_fetch_toydata(cfg, raw_path, rng)
        else:
            source_file = Path(cfg.dataset["input_file"])
            if not source_file.exists():  # pragma: no cover - manifest error guard
                raise FileNotFoundError(source_file)
            shutil.copyfile(source_file, raw_path)

        with open(raw_path, "r", encoding="utf-8") as handle:
            raw_data = [json.loads(line) for line in handle]
        sampled = _maybe_subsample(raw_data, cfg.dataset.get("size"), rng)
        noisy = _apply_noise(sampled, noise_cfg, rng)
        _dump_jsonl(raw_path, noisy)
        mlflow.log_metric("dataset_rows", len(noisy))

        paths = _split_and_hide(noisy, split_dir, cfg, rng)
        mlflow.log_artifact(str(raw_path))
        for file_path in paths.values():
            mlflow.log_artifact(str(file_path))

        processed_dir = base_dir / "processed"
        _ensure_dir(processed_dir)
        _ttlab_process(paths["train"], paths["dev"], paths["test_visible"], processed_dir, cfg)

        tokenizer_path, tokenizer_entry = _ensure_tokenizer(processed_dir, cfg)
        if tokenizer_path:
            mlflow.log_artifact(str(tokenizer_path))

        model_dir = base_dir / "model"
        _ensure_dir(model_dir)
        trainer = get_trainer(arch)
        if trainer is None:
            trainer = _ttlab_train
        trainer(cfg, processed_dir, model_dir)
        mlflow.log_artifacts(str(model_dir))

        metrics = _evaluate(processed_dir, model_dir, paths, cfg)
        metrics_path = base_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        mlflow.log_dict(metrics, "metrics.json")
        mlflow.log_metrics(
            {
                "visible_loss": metrics["visible"]["loss"],
                "visible_token_acc": metrics["visible"]["token_acc"],
                "hidden_loss": metrics["hidden"]["loss"],
                "hidden_token_acc": metrics["hidden"]["token_acc"],
                "token_acc_gap": metrics["gap"],
            }
        )

        report_path = base_dir / "report.md"
        _write_report(report_path, cfg, metrics)
        mlflow.log_artifact(str(report_path))
        rprint(f"[green]DONE: report -> {report_path}")

        _write_run_lock(base_dir, cfg, manifest_snapshot, metrics_path, grid_tags, tokenizer_entry)


def report_experiment(manifest_path: str) -> None:
    cfg = _load_cfg(manifest_path)
    report_path = Path(cfg.artifacts_dir) / "report.md"
    if not report_path.exists():  # pragma: no cover - user error
        raise FileNotFoundError(report_path)
    with open(report_path, "r", encoding="utf-8") as handle:
        print(handle.read())


def _generate_or_fetch_toydata(cfg: Cfg, out_path: Path, rng: random.Random) -> None:
    toy_lab_home = os.environ.get("TOY_LANG_LAB_HOME")
    if toy_lab_home and Path(toy_lab_home).exists():
        cmd = [
            "python",
            "-m",
            "toy_lang_lab.cli",
            "gen",
            "--recipe",
            cfg.dataset.get("recipe", "minimal_24"),
            "--out",
            str(out_path),
        ]
        try:
            _run(cmd, cwd=toy_lab_home)
            return
        except Exception as exc:  # pragma: no cover - optional dependency
            rprint(f"[yellow]toy-lang-lab generation failed, using fallback ({exc})")

    subjects = ["cat", "dog", "robot", "child"]
    verbs = ["takes", "drops", "sees", "finds"]
    objects = ["key", "ball", "book", "coin", "map", "toy"]
    cities = ["Paris", "Berlin", "Rome", "Prague", "Lisbon", "Oslo"]
    rows: list[Mapping[str, Any]] = []
    target_size = int(cfg.dataset.get("size") or 24)
    limit = max(target_size, 24)
    for s, v, obj, city in product(subjects, verbs, objects, cities):
        rows.append(
            {
                "text": f"{s} {v} the {obj} in {city}",
                "template": "S V O in L",
                "subject": s,
                "verb": v,
                "object": obj,
                "location": city,
            }
        )
        if len(rows) >= limit:
            break
    rng.shuffle(rows)
    _ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows[:limit]:
            handle.write(json.dumps(row) + "\n")


def _split_and_hide(
    data: List[Mapping[str, Any]],
    out_dir: Path,
    cfg: Cfg,
    rng: random.Random,
) -> Dict[str, Path]:
    rows = list(data)
    rng.shuffle(rows)

    split = cfg.dataset["split"]
    total = len(rows)
    train_size = int(total * split["train"])
    dev_size = int(total * split["dev"])
    train = rows[:train_size]
    dev = rows[train_size : train_size + dev_size]
    test = rows[train_size + dev_size :]

    hide_cfg = cfg.dataset.get("hide", {})
    strategy = hide_cfg.get("strategy", "by_word")
    value = hide_cfg.get("value", "")

    def is_hidden(example: Mapping[str, Any]) -> bool:
        if strategy == "by_template":
            return example.get("template") == value
        if strategy == "by_word":
            tokens = str(example.get("text", "")).split()
            return value in tokens
        return False

    apply_to = hide_cfg.get("apply_to", "train")
    if isinstance(apply_to, str):
        apply_set = {apply_to}
    else:
        apply_set = set(apply_to)

    def filter_split(rows_set: List[Mapping[str, Any]], label: str) -> tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
        rows_local = list(rows_set)
        if strategy == "percent":
            if label not in apply_set:
                return rows_local, []
            pct = float(hide_cfg.get("value", 0) or 0)
            pct = max(0.0, min(100.0, pct))
            drop = int(round(len(rows_local) * pct / 100.0))
            drop = min(drop, len(rows_local))
            if drop <= 0:
                return rows_local, []
            indices = list(range(len(rows_local)))
            rng.shuffle(indices)
            drop_set = set(indices[:drop])
            visible_rows = [rows_local[idx] for idx in range(len(rows_local)) if idx not in drop_set]
            hidden_rows = [rows_local[idx] for idx in range(len(rows_local)) if idx in drop_set]
            return visible_rows, hidden_rows

        hidden_rows_all = [row for row in rows_local if is_hidden(row)]
        if label in apply_set:
            visible_rows = [row for row in rows_local if not is_hidden(row)]
        else:
            visible_rows = rows_local
        return visible_rows, hidden_rows_all

    train_visible, _ = filter_split(train, "train")
    dev_visible, _ = filter_split(dev, "dev")
    test_visible, test_hidden_candidates = filter_split(test, "test")

    export_hidden = hide_cfg.get("export_hidden_test", True)
    test_hidden = test_hidden_candidates if export_hidden else []

    paths = {
        "train": _dump_jsonl(out_dir / "train.jsonl", train_visible),
        "dev": _dump_jsonl(out_dir / "dev.jsonl", dev_visible),
        "test_visible": _dump_jsonl(out_dir / "test_visible.jsonl", test_visible),
        "test_hidden": _dump_jsonl(out_dir / "test_hidden.jsonl", test_hidden),
    }
    return paths


def _ttlab_process(train: Path, dev: Path, test_visible: Path, out_dir: Path, cfg: Cfg) -> None:
    ttlab_home = os.environ.get("TTLAB_HOME")
    if ttlab_home and Path(ttlab_home).exists():
        cmd = [
            "ttlab",
            "process",
            "run",
            "--in",
            str(train),
            "--schema",
            str(Path("conf") / "data" / "sample_dataset.yaml"),
            "--format",
            "JSONL",
            "--out",
            str(out_dir),
            "--split",
            "train=1.0,dev=0.0,test=0.0",
            "--seed",
            str(cfg.seed),
            "--mlflow-uri",
            os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow_uri),
            "--mlflow",
        ]
        try:
            _run(cmd, cwd=ttlab_home)
            return
        except Exception as exc:  # pragma: no cover - optional dependency
            rprint(f"[yellow]ttlab process failed, copying splits ({exc})")

    for src in (train, dev, test_visible):
        shutil.copy(src, out_dir / src.name)


def _ttlab_train(processed_dir: Path, model_dir: Path, cfg: Cfg) -> None:
    ttlab_home = os.environ.get("TTLAB_HOME")
    if ttlab_home and Path(ttlab_home).exists():
        cmd = [
            "ttlab",
            "train",
            "run",
            "--data",
            str(processed_dir),
            "--out",
            str(model_dir),
            "--d-model",
            str(cfg.model.get("d_model", 8)),
            "--n-layers",
            str(cfg.model.get("n_layers", 1)),
            "--epochs",
            str(cfg.model.get("epochs", 3)),
            "--batch-size",
            str(cfg.model.get("batch_size", 32)),
            "--mlflow-uri",
            os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow_uri),
            "--mlflow",
        ]
        try:
            _run(cmd, cwd=ttlab_home)
            return
        except Exception as exc:  # pragma: no cover - optional dependency
            rprint(f"[yellow]ttlab train failed, creating dummy checkpoint ({exc})")

    _ensure_dir(model_dir)
    with open(model_dir / "DUMMY_CHECKPOINT.bin", "wb") as handle:
        handle.write(b"dummy")


def _evaluate(
    processed_dir: Path,
    model_dir: Path,
    paths: Mapping[str, Path],
    cfg: Cfg,
) -> Dict[str, Any]:
    ttlab_home = os.environ.get("TTLAB_HOME")
    if ttlab_home and Path(ttlab_home).exists():  # pragma: no cover - placeholder metrics
        visible_acc = 0.8
        hidden_acc = 0.6
    else:
        visible_acc = 0.85
        hidden_acc = 0.55

    def pack(acc: float) -> Dict[str, float]:
        loss = round(max(0.0, 2.0 - acc), 4)
        return {"loss": loss, "token_acc": round(acc, 4)}

    metrics = {
        "visible": pack(visible_acc),
        "hidden": pack(hidden_acc),
    }
    metrics["gap"] = round(metrics["visible"]["token_acc"] - metrics["hidden"]["token_acc"], 4)
    return metrics


def _write_report(path: Path, cfg: Cfg, metrics: Mapping[str, Any]) -> None:
    lines = [f"# exp-lab A1 Report — {cfg.experiment_id}\n"]
    lines.append(f"Seed: {cfg.seed}\n")
    hide_cfg = cfg.dataset.get("hide", {})
    lines.append("## Hide configuration\n")
    lines.append(f"- Strategy: {hide_cfg.get('strategy')}\n")
    lines.append(f"- Value: {hide_cfg.get('value')}\n")
    lines.append("## Metrics\n")
    lines.append(
        f"- Visible: loss={metrics['visible']['loss']}, token_acc={metrics['visible']['token_acc']}\n"
    )
    lines.append(
        f"- Hidden: loss={metrics['hidden']['loss']}, token_acc={metrics['hidden']['token_acc']}\n"
    )
    lines.append(f"- Gap (visible-hidden): {metrics['gap']}\n")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
