"""Locations of packaged base configs and source-only development tools."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "base"
