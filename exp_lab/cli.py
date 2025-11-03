from __future__ import annotations

import argparse

from exp_lab.pipeline import report_experiment, run_experiment
from exp_lab.registry import cache_ls, cache_prune
from exp_lab.sweeps import reproduce_from_lock, run_sweep


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
    else:  # pragma: no cover - parser enforces commands
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
