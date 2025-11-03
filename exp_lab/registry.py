from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

IDX = Path("registry/tokenizers/index.json")
_DEFAULT = {"tokenizers": []}


def _load() -> Dict[str, Any]:
    if not IDX.exists():
        IDX.parent.mkdir(parents=True, exist_ok=True)
        with open(IDX, "w", encoding="utf-8") as handle:
            json.dump(_DEFAULT, handle)
    with open(IDX, "r", encoding="utf-8") as handle:
        data: Dict[str, Any] = json.load(handle)
    return data


def _save(idx: Dict[str, Any]) -> None:
    IDX.parent.mkdir(parents=True, exist_ok=True)
    with open(IDX, "w", encoding="utf-8") as handle:
        json.dump(idx, handle, indent=2, ensure_ascii=False)


def tokenizer_id(spec: Dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True).encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def get_or_register(spec: Dict[str, Any]) -> Dict[str, Any]:
    idx = _load()
    tid = tokenizer_id(spec)
    default_path = str(Path("registry/tokenizers") / tid / "tokenizer.json")
    for entry in idx["tokenizers"]:
        if entry["id"] == tid:
            if not entry.get("path"):
                entry["path"] = default_path
                _save(idx)
            return entry
    entry = {
        "id": tid,
        "spec": spec,
        "created_at": int(time.time()),
        "path": default_path,
    }
    idx["tokenizers"].append(entry)
    _save(idx)
    return entry


def cache_ls() -> None:
    idx = _load()
    print(json.dumps(idx, indent=2))


def cache_prune(older_than: str | None = None) -> None:
    _ = older_than
    print("no-op prune")
