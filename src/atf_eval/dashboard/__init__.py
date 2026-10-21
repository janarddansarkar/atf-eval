"""Deterministic + LLM Groups 1-5 dashboard generator -- `atf-eval dashboard`.

See `generate.py` for the implementation; `add_arguments()`/`run()` are the
shared entry points used by both the CLI subcommand and this module's own
standalone `python -m atf_eval.dashboard.generate` invocation.
"""
from atf_eval.dashboard.generate import add_arguments, main, run

__all__ = ["add_arguments", "main", "run"]
