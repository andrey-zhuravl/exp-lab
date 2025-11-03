from __future__ import annotations

import itertools
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable


_ACTIVE_RUN = None
_RUN_COUNTER = itertools.count()


def set_tracking_uri(uri: str) -> None:  # pragma: no cover - simple stub
    _ = uri


def set_experiment(name: str) -> None:  # pragma: no cover - simple stub
    _ = name


class _RunContext:
    def __init__(self, run_name: str | None = None):
        self.run_name = run_name or "run"
        self.info = SimpleNamespace(run_id=f"run-{next(_RUN_COUNTER)}-{int(time.time()*1000)}")

    def __enter__(self):
        global _ACTIVE_RUN
        _ACTIVE_RUN = self
        return self

    def __exit__(self, exc_type, exc, tb):
        global _ACTIVE_RUN
        _ACTIVE_RUN = None
        return False


def start_run(run_name: str | None = None):
    return _RunContext(run_name)


def active_run():
    return _ACTIVE_RUN


def log_params(params: Dict[str, Any]) -> None:  # pragma: no cover
    _ = params


def log_param(key: str, value: Any) -> None:  # pragma: no cover
    _ = (key, value)


def set_tag(key: str, value: Any) -> None:  # pragma: no cover
    _ = (key, value)


def log_artifact(path: str) -> None:  # pragma: no cover
    _ = path


def log_artifacts(path: str) -> None:  # pragma: no cover
    _ = path


def log_dict(payload: Dict[str, Any], artifact_file: str) -> None:  # pragma: no cover
    _ = (payload, artifact_file)


def log_metrics(metrics: Dict[str, float]) -> None:  # pragma: no cover
    _ = metrics


def log_metric(key: str, value: float) -> None:  # pragma: no cover
    _ = (key, value)
