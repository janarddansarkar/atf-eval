from __future__ import annotations

import importlib
import os
import sys

from atf_eval.adapter import TrajectoryAgent


def _ensure_cwd_importable() -> None:
    """`python -m atf_eval` puts the current directory on sys.path
    automatically; the installed `atf-eval` console-script entry point does
    not. Adapter specs are meant to resolve relative to wherever the user is
    running the command from (e.g. an `atf-eval init`-generated project), so
    make that true regardless of how the CLI was invoked."""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def load_adapter(spec: str) -> TrajectoryAgent:
    """Instantiate an adapter from "module.path:ClassName", no constructor args."""
    if ":" not in spec:
        raise ValueError(
            f"invalid adapter spec {spec!r}, expected 'module.path:ClassName'"
        )
    module_path, class_name = spec.split(":", 1)

    _ensure_cwd_importable()
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"could not import adapter module {module_path!r}: {e}") from e

    try:
        adapter_cls = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"module {module_path!r} has no class {class_name!r}"
        ) from e

    return adapter_cls()
