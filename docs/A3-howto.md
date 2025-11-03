# A3 HOWTO

The A3 milestone extends the base exp-lab runner with retrieval, agents, an MCP-style server, and a minimal human-in-the-loop (HITL) loop. Everything runs on CPU with bundled toy data.

## Prerequisites
- Install dependencies: `pip install -e .`
- Ensure MLflow is reachable at `http://127.0.0.1:5000` (default from the manifest).

## End-to-end demo
```bash
pip install -e .
exp agent run -m experiments/a3.rag.yaml
exp rag eval -m experiments/a3.rag.yaml
```
The orchestrator will generate the dataset, train/evaluate the tiny model, build the TF-IDF index, evaluate retrieval, and write the consolidated report to `out/a3/report.md`.

## RAG pipeline
- Source documents live in `data/rag_demo/*.md`; add new Markdown files here to grow the toy corpus.
- Maintain QA pairs in `data/rag_demo/qa_pairs.jsonl` with objects of the form `{ "q": ..., "doc_id": ... }`.
- Build the index: `exp rag index -m experiments/a3.rag.yaml` (artifacts land in `out/a3/index/`).
- Query interactively: `exp rag query -m experiments/a3.rag.yaml -q "Which article mentions roles?" -k 3`.
- Evaluate retrieval (Recall@k, MRR@k, lexical F1): `exp rag eval -m experiments/a3.rag.yaml` → `out/a3/rag_report.md` + metrics JSON.

## HITL workflow
1. Review mistakes extracted from the latest run: `exp hitl review --run out/a3/run_001`.
2. Accept an item into the edits queue: `exp hitl accept --run out/a3/run_001 --id m1`.
3. Apply edits to the hard set (mirrors the queue): `exp hitl apply --run out/a3/run_001`.
4. Trigger incremental fine-tuning: `exp agent finetune --from out/a3/run_001` (creates a temporary manifest with `model.finetune_from`).

Accepted edits are appended to `out/a3/hitl/edits.jsonl` and copied into `out/a3/hitl/hard_set.jsonl` upon apply.

## MCP server
- Start the server locally: `exp mcp serve` (FastAPI on `127.0.0.1:8765`).
- List tools: `exp mcp list` → returns `generate`, `train`, `eval`, `rag.index`, `rag.query`, `hitl.accept`, `hitl.apply`.
- Call a tool without leaving the CLI: `exp mcp call --tool rag.query --args '{"q":"Which article mentions roles?","manifest":"experiments/a3.rag.yaml"}'`.

## Jenkins automation
`Jenkinsfile.a3` runs the install, orchestrator, and retrieval evaluation stages, then archives `out/a3/report.md`, `out/a3/rag_report.md`, and the JSON metrics for inspection.
