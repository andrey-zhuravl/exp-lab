from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import FastAPI
import uvicorn

from exp_lab.hitl.review_cli import accept_item, apply_pending_edits
from exp_lab.rag.indexer import build_index, query, _load_cfg

TOOLS = [
    "generate",
    "train",
    "eval",
    "rag.index",
    "rag.query",
    "hitl.accept",
    "hitl.apply",
]

app = FastAPI(title="exp-lab MCP server")


@app.get("/tools")
def tools() -> Dict[str, Any]:
    return {"tools": TOOLS}


@app.post("/call")
def call(payload: Dict[str, Any]) -> Dict[str, Any]:
    tool = payload.get("tool")
    args = payload.get("args", {})

    if tool == "rag.index":
        manifest = args.get("manifest", "experiments/a3.rag.yaml")
        build_index(manifest)
        return {"ok": True}

    if tool == "rag.query":
        manifest = args.get("manifest", "experiments/a3.rag.yaml")
        cfg = _load_cfg(manifest)
        rag_cfg = cfg["experiment"]["rag"]
        result = query(rag_cfg["index_dir"], args.get("q", ""), int(args.get("k", rag_cfg.get("top_k", 3))))
        return result

    if tool == "hitl.accept":
        run_dir = args.get("run", "out/a3/run_001")
        item_id = args.get("id")
        try:
            record = accept_item(run_dir, item_id)
        except (ValueError, LookupError) as exc:
            return {"error": str(exc)}
        return {"ok": True, "accepted": record}

    if tool == "hitl.apply":
        hard_path = apply_pending_edits()
        if hard_path is None:
            return {"ok": False, "message": "no edits"}
        return {"ok": True, "hard_set": str(hard_path)}

    return {"error": "not_implemented", "tool": tool}


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    uvicorn.run(app, host=host, port=port)


def tools_list() -> None:
    print(json.dumps({"tools": TOOLS}, indent=2))


def call_tool_cli(tool: str | None, args_json: str | None) -> None:
    from fastapi.testclient import TestClient

    tool_name = tool or "rag.query"
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as exc:  # pragma: no cover - CLI guard
        raise SystemExit(f"Invalid JSON: {exc}") from exc

    client = TestClient(app)
    response = client.post("/call", json={"tool": tool_name, "args": args})
    print(response.json())
