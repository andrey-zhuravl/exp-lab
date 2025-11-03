from __future__ import annotations

import copy
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from exp_lab.pipeline import run_experiment
from exp_lab.reporting import append_result, write_summary


def _set_nested(obj: Dict[str, Any], key_path: str, value: Any) -> None:
    cursor = obj
    parts = key_path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _grid_product(grid: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def run_sweep(manifest_path: str, max_parallel: int = 1, extra_tag: str | None = None) -> None:
    if max_parallel != 1:
        raise NotImplementedError("Only sequential execution is supported in A2")

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    sweep_cfg = manifest["sweep"]
    base_manifest_path = sweep_cfg["base"]
    out_dir = sweep_cfg.get("out_dir", "out/a2")
    seeds = sweep_cfg.get("seeds", [212])
    grid = sweep_cfg.get("grid", {})
    mlflow_cfg = sweep_cfg.get("mlflow", {})
    tags = dict(mlflow_cfg.get("tags", {}))
    if extra_tag:
        tags["extra_tag"] = extra_tag
    sweep_id = sweep_cfg.get("id", "sweep")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    run_idx = 0
    lock_runs: List[Dict[str, Any]] = []
    for seed in seeds:
        for choice in _grid_product(grid):
            run_idx += 1
            with open(base_manifest_path, "r", encoding="utf-8") as handle:
                base_manifest = yaml.safe_load(handle)

            experiment_cfg = base_manifest["experiment"]
            experiment_cfg["seed"] = seed
            experiment_cfg.setdefault("artifacts", {})["base_dir"] = os.path.join(
                out_dir, f"run_{run_idx:03d}"
            )
            for key, value in choice.items():
                _set_nested(experiment_cfg, key, value)

            mlflow_section = experiment_cfg.setdefault("mlflow", {})
            mlflow_section.setdefault(
                "uri", os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
            )
            mlflow_section.setdefault(
                "experiment_name", mlflow_cfg.get("experiment_name", "exp-lab/A2")
            )

            derived_path = Path(out_dir) / f"derived_{run_idx:03d}.yaml"
            with open(derived_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(base_manifest, handle, sort_keys=False)

            run_tags = dict(tags)
            run_tags.setdefault("grid_id", sweep_id)
            run_tags["run_index"] = run_idx
            os.environ["EXP_GRID_TAGS"] = json.dumps(run_tags)

            run_experiment(str(derived_path))

            run_dir = experiment_cfg["artifacts"]["base_dir"]
            append_result(run_dir, seed, choice, run_idx)

            run_entry: Dict[str, Any] = {
                "run_index": run_idx,
                "seed": seed,
                "grid_choice": copy.deepcopy(choice),
                "derived_manifest_path": str(derived_path),
                "artifacts_dir": run_dir,
            }
            run_lock_path = Path(run_dir) / "lock.json"
            if run_lock_path.exists():
                with open(run_lock_path, "r", encoding="utf-8") as handle:
                    run_entry.update(json.load(handle))
            lock_runs.append(run_entry)

    write_summary(out_dir)
    _write_sweep_lock(
        out_dir,
        {
            "sweep_id": sweep_id,
            "source_manifest": manifest_path,
            "base_manifest": base_manifest_path,
            "runs": lock_runs,
        },
    )


def _write_sweep_lock(out_dir: str, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("created_at", int(time.time()))
    locks_dir = Path(out_dir) / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    sweep_id = payload.get("sweep_id", "sweep")
    lock_path = locks_dir / f"{sweep_id}.lock.json"
    with open(lock_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print("Sweep lock written to", lock_path)


def reproduce_from_lock(lock_path: str, run_id: int = 0) -> None:
    with open(lock_path, "r", encoding="utf-8") as handle:
        lock_data = json.load(handle)
    runs = lock_data.get("runs", [])
    if not runs:
        raise ValueError("No runs recorded in lock file")
    if run_id < 0 or run_id >= len(runs):
        raise IndexError(f"run_id {run_id} out of range (0-{len(runs)-1})")
    manifest_snapshot = runs[run_id].get("manifest_snapshot_path")
    if not manifest_snapshot:
        raise KeyError("manifest_snapshot_path missing from lock entry")
    run_experiment(manifest_snapshot)
