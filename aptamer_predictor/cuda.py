"""CUDA device detection and management."""

from __future__ import annotations

_cached_device: str | None = None


def detect_cuda() -> bool:
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> str:
    """Return 'cuda' if available, otherwise 'cpu'. Result is cached."""
    global _cached_device
    if _cached_device is None:
        _cached_device = "cuda" if detect_cuda() else "cpu"
    return _cached_device
