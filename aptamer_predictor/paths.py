"""Shared path resolution helpers."""

from __future__ import annotations

import os


def resolve_model_dir(model_dir: str | None = None) -> str:
    """Resolve the model directory for CLI and TUI callers."""
    if model_dir:
        return os.path.abspath(model_dir)

    env_dir = os.environ.get("APTAMER_MODEL_DIR")
    if env_dir:
        return os.path.abspath(env_dir)

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(pkg_dir)
    return os.path.join(project_root, "models")
