from __future__ import annotations

import json
from pathlib import Path
from typing import List

__all__ = [
    "review",
    "accept",
    "apply_edits",
    "load_mistakes",
    "accept_item",
    "apply_pending_edits",
]

DEF_MISTAKES = "mistakes.jsonl"


def _mistakes_path(run_dir: str) -> Path:
    return Path(run_dir) / DEF_MISTAKES


def load_mistakes(run_dir: str) -> List[dict[str, object]]:
    path = _mistakes_path(run_dir)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        demo_record = {"id": "m1", "text": "cat finds the key in Paris", "label": "who", "pred": "what"}
        path.write_text(json.dumps(demo_record) + "\n", encoding="utf-8")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def review(run_dir: str) -> None:
    for record in load_mistakes(run_dir):
        print(json.dumps(record))


def _edits_path() -> Path:
    return Path("out") / "a3" / "hitl" / "edits.jsonl"


def _hard_set_path() -> Path:
    return Path("out") / "a3" / "hitl" / "hard_set.jsonl"


def accept_item(run_dir: str, item_id: str) -> dict[str, object]:
    if not item_id:
        raise ValueError("item_id is required")

    for record in load_mistakes(run_dir):
        if record.get("id") == item_id:
            edits_path = _edits_path()
            edits_path.parent.mkdir(parents=True, exist_ok=True)
            with edits_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"op": "upsert", "item": record}) + "\n")
            return record
    raise LookupError(f"item {item_id!r} not found")


def accept(run_dir: str, item_id: str | None) -> None:
    if item_id is None:
        raise SystemExit("--id required")
    try:
        accept_item(run_dir, item_id)
    except (ValueError, LookupError) as exc:  # pragma: no cover - CLI exit path
        raise SystemExit(str(exc)) from exc
    print(f"accepted {item_id}")


def apply_pending_edits() -> Path | None:
    edits_path = _edits_path()
    if not edits_path.exists():
        return None
    hard_path = _hard_set_path()
    hard_path.parent.mkdir(parents=True, exist_ok=True)
    hard_path.write_text(edits_path.read_text(encoding="utf-8"), encoding="utf-8")
    return hard_path


def apply_edits(run_dir: str) -> None:
    hard_path = apply_pending_edits()
    if hard_path is None:
        print("no edits")
    else:
        print(f"hard_set written: {hard_path}")


