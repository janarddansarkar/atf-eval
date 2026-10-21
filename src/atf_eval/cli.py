from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from atf_eval.aggregate import DEFAULT_WEIGHTS, resolve_weights, weights_from_cli
from atf_eval.dataset import group_by_conversation, load_dataset
from atf_eval.llm_provider import load_dotenv
from atf_eval.loader import load_adapter
from atf_eval.report import aggregate, render_results_table, write_reports
from atf_eval.runner import run_evaluation
from atf_eval.scaffold import write_scaffold
from atf_eval.settings import evaluation_settings


_DEFAULT_WEIGHTS_SPEC = ",".join(f"{k}={v:.4g}" for k, v in DEFAULT_WEIGHTS.items())


def add_weights_arguments(parser: argparse.ArgumentParser) -> None:
    """--weights / --weights-file (mutually exclusive), shared by `run`,
    `submit` and `dashboard` so every command accepts weights the same way."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--weights",
        default=None,
        metavar="SPEC",
        help='ATF component weights as "nts=0.4,sts=0.3,tis=0.2,os=0.1": all four, each a finite '
        f"non-negative number, summing to 1. Default: {_DEFAULT_WEIGHTS_SPEC}",
    )
    group.add_argument(
        "--weights-file",
        default=None,
        metavar="PATH",
        help='The same weights as a JSON object {"nts": ..., "sts": ..., "tis": ..., "os": ...}. '
        "Cannot be combined with --weights.",
    )


def validated_cli_weights(args: argparse.Namespace) -> dict[str, float] | None:
    """The weights given by --weights / --weights-file, validated by the same
    resolve_weights() the API uses; None when neither flag was given (the
    caller then uses DEFAULT_WEIGHTS). Raises ValueError on invalid input."""
    config = weights_from_cli(getattr(args, "weights", None), getattr(args, "weights_file", None))
    return None if config is None else resolve_weights(config)[0]


def _weights_error(e: ValueError) -> int:
    print(f"atf-eval: error: invalid ATF weights: {e}", file=sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atf-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Evaluate an agent against a golden dataset")
    run_parser.add_argument("--dataset", required=True, help="Golden dataset JSONL path")
    run_parser.add_argument("--adapter", required=True, help="module.path:ClassName")
    run_parser.add_argument("--output-dir", default="./results")
    add_weights_arguments(run_parser)
    run_parser.add_argument("--tool-arg-tolerance", type=float, default=0.0)
    run_parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N conversations")
    run_parser.add_argument("--concurrency", type=int, default=1)

    report_parser = subparsers.add_parser(
        "report", help="Print the per-conversation results table from a saved report.json"
    )
    report_parser.add_argument("path", help="Path to a report_<timestamp>.json written by `atf-eval run`")

    init_parser = subparsers.add_parser(
        "init", help="Generate a starter adapter + matching golden dataset (fastest way to get going)"
    )
    init_parser.add_argument(
        "directory", nargs="?", default=".", help="Where to write the starter files (default: current directory)"
    )
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing files")

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Score golden + observed-fixture JSON directories (deterministic + LLM Groups 1-5) "
        "and produce an interactive HTML dashboard",
    )
    from atf_eval.dashboard.generate import add_arguments as add_dashboard_arguments

    add_dashboard_arguments(dashboard_parser)

    serve_parser = subparsers.add_parser("serve", help="Run atf-eval as an HTTP API (FastAPI)")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--data-dir", default=None, help="where goldens/evaluations are stored "
                              "(default: $ATF_SERVER_DATA or ./atf_server_data)")
    serve_parser.add_argument("--workers", type=int, default=None,
                              help="evaluations scored at the same time; further submissions wait as "
                              "`queued` (default: $ATF_SERVER_WORKERS or 2)")

    submit_parser = subparsers.add_parser(
        "submit", help="Run your agent's adapter locally and have an atf-eval server score it")
    submit_parser.add_argument("--adapter", required=True, help="module.path:ClassName")
    submit_parser.add_argument("--dataset", help="golden dataset JSONL (customer inputs + expected trajectory)")
    submit_parser.add_argument("--golden-id", help="use a golden already stored on the server "
                               "(still needs --dataset for the customer inputs)")
    submit_parser.add_argument("--upload-golden", action="store_true",
                               help="store --dataset on the server and reference it by id")
    submit_parser.add_argument("--server", default=None, help="atf-eval server URL (default: $ATF_SERVER or http://127.0.0.1:8000)")
    submit_parser.add_argument("--output-dir", default="./results", help="where to save the dashboard and CSVs")
    submit_parser.add_argument("--no-llm-evals", action="store_true", help="skip the Group 1-5 LLM metrics")
    submit_parser.add_argument("--provider", default=None)
    submit_parser.add_argument("--judge-model", default=None)
    submit_parser.add_argument("--effort", default="low")
    add_weights_arguments(submit_parser)
    submit_parser.add_argument("--concurrency", type=int, default=1, help="conversations run in parallel")
    submit_parser.add_argument("--limit", type=int, default=None, help="only the first N conversations")
    submit_parser.add_argument("--timeout", type=float, default=3600.0)
    return parser


def _print_summary(summary: dict) -> None:
    print("=== ATF Evaluation Summary ===")
    print(f"Conversations:   {summary['total_conversations']}")
    print(f"Turns:           {summary['total_turns']}")
    atf = summary["atf"]
    atf_score_str = f"{atf['score']:.3f}" if atf["score"] is not None else "N/A"
    print(f"\nATF score:       {atf_score_str}")
    print(f"Metric coverage: {atf['coverage']:.1%}")
    print("\nMetric groups (mean):")
    for key, group in summary["groups"].items():
        score_str = f"{group['score']:.3f}" if group["score"] is not None else "N/A"
        print(f"  {key.upper():4s} {score_str:>8s}  ({group['status']})")


def cmd_run(args: argparse.Namespace) -> int:
    try:  # validate before loading the dataset or running the agent
        weights, subweights = resolve_weights(validated_cli_weights(args))
    except ValueError as e:
        return _weights_error(e)

    turns = load_dataset(args.dataset)
    conversations = group_by_conversation(turns)

    if args.limit is not None:
        conversations = dict(list(conversations.items())[: args.limit])

    adapter = load_adapter(args.adapter)

    results = run_evaluation(
        conversations,
        adapter,
        weights=weights,
        subweights=subweights,
        concurrency=args.concurrency,
        tolerance=args.tool_arg_tolerance,
    )

    settings = evaluation_settings(args.tool_arg_tolerance, weights, subweights)
    json_path, csv_path = write_reports(results, args.output_dir, settings)
    _print_summary(aggregate(results))
    print("\nPer-conversation results:\n")
    print(render_results_table([
        {
            "conversation_id": r.conversation_id,
            "atf": r.atf,
            "metric_coverage": r.metric_coverage,
            "nts": r.nts,
            "sts": r.sts,
            "tis": r.tis,
            "os": r.os,
        }
        for r in results
    ]))
    print(f"\nReports written to:\n  {json_path}\n  {csv_path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    load_dotenv()
    try:
        import uvicorn

        from atf_eval.api.app import DEFAULT_WORKERS, create_app
    except ImportError:
        print("atf-eval serve needs the server extra: pip install 'atf-eval[server]'", file=sys.stderr)
        return 1
    workers = args.workers
    if workers is None:
        raw = os.environ.get("ATF_SERVER_WORKERS", "").strip()
        try:
            workers = int(raw) if raw else DEFAULT_WORKERS
        except ValueError:
            print(f"atf-eval: error: ATF_SERVER_WORKERS must be an integer, got {raw!r}", file=sys.stderr)
            return 2
    if workers < 1:
        print(f"atf-eval: error: --workers must be at least 1, got {workers}", file=sys.stderr)
        return 2
    print(f"[atf-eval] scoring up to {workers} evaluation(s) at a time", file=sys.stderr)
    uvicorn.run(create_app(args.data_dir, workers=workers), host=args.host, port=args.port)
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    load_dotenv()
    from atf_eval.client import AtfApiError, AtfClient, collect

    if not args.dataset:
        print("--dataset is required (it holds the customer inputs the agent is driven with)", file=sys.stderr)
        return 2
    try:  # validate locally before running the agent; None -> the server's DEFAULT_WEIGHTS
        weights = validated_cli_weights(args)
    except ValueError as e:
        return _weights_error(e)
    golden = load_dataset(args.dataset)
    adapter = load_adapter(args.adapter)
    options = {
        "llm_evals": not args.no_llm_evals,
        "provider": args.provider,
        "judge_model": args.judge_model,
        "effort": args.effort,
        "weights": weights,
    }
    try:
        client = AtfClient(args.server)
        client.health()
    except Exception as e:  # noqa: BLE001
        print(f"cannot reach the atf-eval server at {args.server or '$ATF_SERVER / http://127.0.0.1:8000'}: {e}\n"
              "start one with: atf-eval serve", file=sys.stderr)
        return 1
    try:
        golden_id = args.golden_id
        if args.upload_golden:
            golden_id = client.upload_golden(Path(args.dataset).stem, golden)
            print(f"[info] golden stored on the server as {golden_id}", file=sys.stderr)
        print(f"[info] running the agent over {len(group_by_conversation(golden))} conversation(s) ...", file=sys.stderr)
        collected = collect(adapter, golden, concurrency=args.concurrency, limit=args.limit)
        for err in collected.errors:
            print(f"[WARN] agent failed on {err['conversation_id']}/{err['turn_id']}: {err['error']}", file=sys.stderr)
        eval_id = client.submit(collected, golden_id=golden_id, golden=None if golden_id else golden,
                                options=options, label=f"{args.adapter} / {Path(args.dataset).name}")
        print(f"[info] submitted evaluation {eval_id}; waiting for the server ...", file=sys.stderr)
        result = client.wait(eval_id, timeout=args.timeout, on_status=_print_eval_status)
        saved = client.download(result, args.output_dir)
    except AtfApiError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    summary = result["summary"]
    print("\n=== ATF Evaluation (server-scored) ===")
    print(f"Conversations: {summary['conversations']}    Judge: {summary['judge'] or 'none (Group 1-5 are N/A)'}")
    print(f"ATF mean:      {_fmt(summary['atf_mean'])}\n")
    print(render_results_table([
        {"conversation_id": r["conversation_id"], "atf": r["atf"], "metric_coverage": r["metric_coverage"],
         "nts": r["nts"], "sts": r["sts"], "tis": r["tis"], "os": r["os"]}
        for r in result["runs"]
    ]))
    for s in result["skipped"]:
        print(f"[skipped] {s['conversation_id']}: {s['reason']}")
    print(f"\nSaved to {args.output_dir}:")
    for name, path in saved.items():
        print(f"  {name:15s} {path}")
    return 0


def _print_eval_status(doc: dict) -> None:
    status = doc["status"]
    if status == "queued" and doc.get("queue_position"):
        print(f"[info] queued: position {doc['queue_position']} -- the server is already scoring "
              f"{doc.get('running', '?')} of {doc.get('workers', '?')} evaluation(s) at a time", file=sys.stderr)
    elif status in ("queued", "running"):
        print(f"[info] {status} ...", file=sys.stderr)


def _fmt(v) -> str:
    return "N/A" if v is None else f"{v:.3f}"


def cmd_report(args: argparse.Namespace) -> int:
    with open(args.path, encoding="utf-8") as f:
        report = json.load(f)
    print(render_results_table(report["conversations"]))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    output_dir = Path(args.directory)
    try:
        written = write_scaffold(output_dir, force=args.force)
    except FileExistsError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    print("Generated:")
    for path in written:
        print(f"  {path}")
    cd_line = "" if str(output_dir) == "." else f"  cd {output_dir}\n"
    print(
        f"\nTry it right now:\n{cd_line}"
        "  atf-eval run --dataset golden_dataset.jsonl --adapter adapter:MyAgentAdapter\n"
        "\nThen open adapter.py and golden_dataset.jsonl and follow the TODOs / README.md "
        "to wire up your real agent."
    )
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from atf_eval.dashboard.generate import run as run_dashboard

    run_dashboard(args)
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "run":
        sys.exit(cmd_run(args))
    if args.command == "report":
        sys.exit(cmd_report(args))
    if args.command == "init":
        sys.exit(cmd_init(args))
    if args.command == "serve":
        sys.exit(cmd_serve(args))
    if args.command == "submit":
        sys.exit(cmd_submit(args))
    if args.command == "dashboard":
        sys.exit(cmd_dashboard(args))
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
