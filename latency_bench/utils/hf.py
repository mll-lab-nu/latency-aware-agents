"""Hugging Face authentication helpers."""

from __future__ import annotations

import os


def resolve_token(explicit_token: str) -> str | None:
    """Resolve the Hugging Face token for artifact downloads.

    Args:
        explicit_token: CLI token value.

    Returns:
        CLI token, environment token, or ``None``.
    """
    token = str(explicit_token).strip()
    if token:
        return token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
