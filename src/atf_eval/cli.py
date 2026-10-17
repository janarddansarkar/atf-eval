from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atf_eval.aggregate import resolve_weights
from atf_eval.dataset import group_by_conversation, load_dataset
from atf_eval.llm_provider import (
    DEFAULT_PROVIDER, PROVIDER_ENV, PROVIDERS, get_provider, load_dotenv, provider_name,
)
from atf_eval.loader import load_adapter
from atf_eval.report import aggregate, render_results_table, write_reports
from atf_eval.runner import run_evaluation
from atf_eval.scaffold import write_scaffold
from atf_eval.settings import evaluation_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atf-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Evaluate an agent against a golden dataset")
    run_parser.add_argument("--dataset", required=True, help="Golden dataset JSONL path")
    run_parser.add_argument("--adapter", required=True, help="module.path:ClassName")
    run_parser.add_argument("--output-dir", default="./results")
    run_parser.add_argument(
        "--weights-file",
        default=None,
        help='JSON override of the ATF component weights {"nts", "sts", "tis", "rs", "os"} (must sum to 1); '
        "defaults to METRICS.md §7",
    )
    run_parser.add_argument("--tool-arg-tolerance", type=float, default=0.0)
    run_parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N conversations")
    run_parser.add_argument("--concurrency", type=int, default=1)
    run_parser.add_argument(
        "--provider",
        choices=list(PROVIDERS),
        default=None,
        help=f"LLM provider for RS's routing judge (default: ${PROVIDER_ENV}, else {DEFAULT_PROVIDER})",
    )
    run_parser.add_argument(
        "--judge-model",
        default=None,
        help="Model for RS's routing judge (METRICS.md §5); default: the provider's default model",
    )
    run_parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=None,
        help="Routing judge model effort (model-dependent; ignored by providers without it)",
    )
    run_parser.add_argument(
        "--no-routing-judge",
        action="store_true",
        help="Skip the LLM routing judge entirely -- RS reports N/A for every conversation instead",
    )

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
    submit_parser.add_argument("--no-routing-judge", action="store_true", help="skip RS's LLM judge (RS is N/A)")
    submit_parser.add_argument("--no-llm-evals", action="store_true", help="skip the Group 1-5 LLM metrics")
    submit_parser.add_argument("--provider", default=None)
    submit_parser.add_argument("--judge-model", default=None)
    submit_parser.add_argument("--effort", default="low")
    submit_parser.add_argument("--weights-file", default=None)
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
    turns = load_dataset(args.dataset)
    conversations = group_by_conversation(turns)

    if args.limit is not None:
        conversations = dict(list(conversations.items())[: args.limit])

    adapter = load_adapter(args.adapter)

    weights_config = None
    if args.weights_file:
        with open(args.weights_file, encoding="utf-8") as f:
            weights_config = json.load(f)
    weights, subweights = resolve_weights(weights_config)

    judge_client = None
    if not args.no_routing_judge:
        judge_client = get_provider(args.provider)
        if judge_client is None:
            key_env = PROVIDERS[provider_name(args.provider)].api_key_env
            print(
                f"[info] {key_env} not set -- RS will report N/A "
                "(pass --no-routing-judge to silence this, or set the key to enable it)",
                file=sys.stderr,
            )
    judge_model = (args.judge_model or judge_client.default_model) if judge_client else None
    judge_effort = args.effort if judge_client and judge_client.supports_effort else None

    results = run_evaluation(
        conversations,
        adapter,
        weights=weights,
        subweights=subweights,
        concurrency=args.concurrency,
        tolerance=args.tool_arg_tolerance,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_effort=judge_effort,
    )

    settings = evaluation_settings(
        judge_model, judge_effort, args.tool_arg_tolerance, weights, subweights,
        judge_provider=judge_client.name if judge_client else None,
    )
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
            "rs": r.rs,
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

        from atf_eval.api.app import create_app
    except ImportError:
        print("atf-eval serve needs the server extra: pip install 'atf-eval[server]'", file=sys.stderr)
        return 1
    uvicorn.run(create_app(args.data_dir), host=args.host, port=args.port)
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    load_dotenv()
    from atf_eval.client import AtfApiError, AtfClient, collect

    if not args.dataset:
        print("--dataset is required (it holds the customer inputs the agent is driven with)", file=sys.stderr)
        return 2
    golden = load_dataset(args.dataset)
    adapter = load_adapter(args.adapter)
    weights = json.loads(Path(args.weights_file).read_text(encoding="utf-8")) if args.weights_file else None
    options = {
        "routing_judge": not args.no_routing_judge,
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
        result = client.wait(eval_id, timeout=args.timeout)
        saved = client.download(result, args.output_dir)
    except AtfApiError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    summary = result["summary"]
    print("\n=== ATF Evaluation (server-scored) ===")
    print(f"Conversations: {summary['conversations']}    Judge: {summary['judge'] or 'none (RS / Group 1-5 are N/A)'}")
    print(f"ATF mean:      {_fmt(summary['atf_mean'])}\n")
    print(render_results_table([
        {"conversation_id": r["conversation_id"], "atf": r["atf"], "metric_coverage": r["metric_coverage"],
         "nts": r["nts"], "sts": r["sts"], "tis": r["tis"], "rs": r["rs"], "os": r["os"]}
        for r in result["runs"]
    ]))
    for s in result["skipped"]:
        print(f"[skipped] {s['conversation_id']}: {s['reason']}")
    print(f"\nSaved to {args.output_dir}:")
    for name, path in saved.items():
        print(f"  {name:15s} {path}")
    return 0


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
        "  atf-eval run --dataset golden_dataset.jsonl --adapter adapter:MyAgentAdapter --no-routing-judge\n"
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
