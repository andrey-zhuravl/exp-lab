from __future__ import annotations

import argparse

from exp_lab.agents.runner import finetune_from, run_all
from exp_lab.pipeline import report_experiment, run_experiment
from exp_lab.registry import cache_ls, cache_prune
from exp_lab.sweeps import reproduce_from_lock, run_sweep
from exp_lab.rag.indexer import build_index, query_cli
from exp_lab.rag.eval import eval_rag
from exp_lab.hitl.review_cli import accept, apply_edits, review
from exp_lab.mcp.server import call_tool_cli, serve, tools_list


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="exp", description="exp-lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment manifest")
    run_parser.add_argument("-m", "--manifest", required=True)

    report_parser = subparsers.add_parser("report", help="Show report for manifest")
    report_parser.add_argument("-m", "--manifest", required=True)

    sweep_parser = subparsers.add_parser("sweep", help="Execute sweep manifest")
    sweep_parser.add_argument("-m", "--manifest", required=True)
    sweep_parser.add_argument("--max-parallel", type=int, default=1)
    sweep_parser.add_argument("--tag", default=None)

    reproduce_parser = subparsers.add_parser("reproduce", help="Re-run a locked sweep run")
    reproduce_parser.add_argument("--lock", required=True)
    reproduce_parser.add_argument("--run-id", type=int, default=0)

    cache_parser = subparsers.add_parser("cache", help="Manage tokenizer cache")
    cache_parser.add_argument("cmd", nargs="?", default="ls", choices=["ls", "prune"])
    cache_parser.add_argument("--older-than", default=None)

    agent_parser = subparsers.add_parser("agent", help="Run task-oriented agents")
    agent_parser.add_argument("action", choices=["run", "finetune"])
    agent_parser.add_argument("-m", "--manifest", default="experiments/a3.rag.yaml")
    agent_parser.add_argument("--from", dest="from_ckpt", default=None)

    rag_parser = subparsers.add_parser("rag", help="RAG utilities")
    rag_parser.add_argument("action", choices=["index", "query", "eval"])
    rag_parser.add_argument("-m", "--manifest", default="experiments/a3.rag.yaml")
    rag_parser.add_argument("-q", "--query", default=None)
    rag_parser.add_argument("-k", "--top-k", type=int, default=3)

    hitl_parser = subparsers.add_parser("hitl", help="Human-in-the-loop review")
    hitl_parser.add_argument("action", choices=["review", "accept", "apply"])
    hitl_parser.add_argument("--run", default="out/a3/run_001")
    hitl_parser.add_argument("--id", dest="item_id", default=None)

    mcp_parser = subparsers.add_parser("mcp", help="MCP-style tool server")
    mcp_parser.add_argument("action", choices=["serve", "list", "call"])
    mcp_parser.add_argument("--tool", default=None)
    mcp_parser.add_argument("--args", dest="tool_args", default=None)
    mcp_parser.add_argument("--host", default="127.0.0.1")
    mcp_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)

    if args.command == "run":
        run_experiment(args.manifest)
    elif args.command == "report":
        report_experiment(args.manifest)
    elif args.command == "sweep":
        run_sweep(args.manifest, max_parallel=args.max_parallel, extra_tag=args.tag)
    elif args.command == "reproduce":
        reproduce_from_lock(args.lock, args.run_id)
    elif args.command == "cache":
        if args.cmd == "ls":
            cache_ls()
        else:
            cache_prune(args.older_than)
    elif args.command == "agent":
        if args.action == "run":
            run_all(args.manifest)
        else:
            finetune_from(args.manifest, args.from_ckpt)
    elif args.command == "rag":
        if args.action == "index":
            build_index(args.manifest)
        elif args.action == "query":
            query_cli(args.manifest, args.query or "who can use NormLang?", args.top_k)
        else:
            eval_rag(args.manifest)
    elif args.command == "hitl":
        if args.action == "review":
            review(args.run)
        elif args.action == "accept":
            accept(args.run, args.item_id)
        else:
            apply_edits(args.run)
    elif args.command == "mcp":
        if args.action == "serve":
            serve(args.host, args.port)
        elif args.action == "list":
            tools_list()
        else:
            call_tool_cli(args.tool, args.tool_args)
    else:  # pragma: no cover - parser enforces commands
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
