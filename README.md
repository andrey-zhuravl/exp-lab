# exp-lab A1 (one-button run)

Goal: synthetic grammar → process → tiny model → eval (visible vs hidden) → MLflow → report.

## Quickstart
```bash
pip install -e .
exp run -m experiments/a1.quickstart.yaml
cat out/a1/report.md
```

## A3 Quickstart (Agents, RAG & HITL)
```bash
pip install -e .
exp agent run -m experiments/a3.rag.yaml
exp rag eval -m experiments/a3.rag.yaml
exp mcp list
```
Artifacts, reports, and HITL edits are written under `out/a3/`.
