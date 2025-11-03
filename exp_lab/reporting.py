from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List


_ROWS: list[Dict[str, Any]] = []


def append_result(run_dir: str, seed: int, choice: Dict[str, Any], run_idx: int) -> None:
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    row: Dict[str, Any] = {
        "run_idx": run_idx,
        "seed": seed,
        "gap": metrics.get("gap", 0.0),
        "visible": metrics.get("visible", {}).get("token_acc"),
        "hidden": metrics.get("hidden", {}).get("token_acc"),
    }
    for key, value in choice.items():
        row[key] = str(value)
    _ROWS.append(row)


def write_summary(out_dir: str) -> None:
    if not _ROWS:
        return
    columns = _collect_columns(_ROWS)
    rows = sorted(
        _ROWS,
        key=lambda row: (row.get("gap", 0.0), -float(row.get("visible", 0.0) or 0.0)),
    )

    csv_path = os.path.join(out_dir, "summary.csv")
    md_path = os.path.join(out_dir, "summary.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})

    md_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines))
    print("Summary written to", csv_path, md_path)


def _collect_columns(rows: List[Dict[str, Any]]) -> List[str]:
    preferred = ["run_idx", "seed", "gap", "visible", "hidden"]
    found: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in found:
                found.append(key)
    ordered = [col for col in preferred if col in found]
    for col in found:
        if col not in ordered:
            ordered.append(col)
    return ordered
