from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class AuthLoginResult:
    env_file: str | None = None
    hf_configured: bool = False
    hf_logged_in: bool = False
    wandb_configured: bool = False
    wandb_logged_in: bool = False


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    node = cfg
    for part in path.split("."):
        if node is None:
            return default
        if isinstance(node, dict):
            if part not in node:
                return default
            node = node[part]
            continue
        try:
            node = getattr(node, part)
        except Exception:
            return default
    return default if node is None else node


def _strip_env_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ[key] = _strip_env_quotes(value)


def _resolve_path(value: Any, *, workspace_dir: Path | None = None, repo_root: Path = REPO_ROOT) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if path.is_absolute():
        return path
    repo_path = repo_root / path
    if repo_path.exists():
        return repo_path
    if workspace_dir is not None:
        return workspace_dir / path
    return repo_path


def _default_env_candidates(workspace_dir: Path | None, repo_root: Path) -> Iterable[Path]:
    if workspace_dir is not None:
        yield workspace_dir / ".env"
        yield workspace_dir / "auth.env"
    yield repo_root.parent / ".env"
    yield repo_root / ".env"
    yield repo_root / "examples/rl_games/auth.env"


def load_auth_env(
    cfg: Any = None,
    *,
    workspace_dir: str | Path | None = None,
    repo_root: str | Path = REPO_ROOT,
    env_file: str | Path | None = None,
    require_env_file: bool = True,
) -> str | None:
    """Load auth.env values into os.environ and normalize token aliases."""
    repo_root_path = Path(repo_root).expanduser().resolve()
    workspace_path = Path(workspace_dir).expanduser().resolve() if workspace_dir not in (None, "") else None
    configured_env_file = env_file if env_file not in (None, "") else _cfg_get(cfg, "auth.env_file")

    loaded_env_file: Path | None = None
    if configured_env_file not in (None, ""):
        candidate = _resolve_path(configured_env_file, workspace_dir=workspace_path, repo_root=repo_root_path)
        if not candidate.exists():
            if require_env_file:
                raise FileNotFoundError(f"Auth env file not found: {candidate}")
        else:
            _load_env_file(candidate)
            loaded_env_file = candidate
    else:
        for candidate in _default_env_candidates(workspace_path, repo_root_path):
            if candidate.exists():
                _load_env_file(candidate)
                loaded_env_file = candidate
                break

    hf_token_env = str(_cfg_get(cfg, "auth.hf_token_env", "HF_TOKEN") or "HF_TOKEN")
    hf_token = (
        _cfg_get(cfg, "auth.hf_token")
        or os.environ.get(hf_token_env)
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if hf_token:
        os.environ["HF_TOKEN"] = str(hf_token)
        os.environ["HUGGINGFACE_HUB_TOKEN"] = str(hf_token)

    wandb_key_env = str(_cfg_get(cfg, "auth.wandb_api_key_env", "WANDB_API_KEY") or "WANDB_API_KEY")
    wandb_key = _cfg_get(cfg, "auth.wandb_api_key") or os.environ.get(wandb_key_env) or os.environ.get("WANDB_API_KEY")
    if wandb_key:
        os.environ["WANDB_API_KEY"] = str(wandb_key)

    return str(loaded_env_file) if loaded_env_file is not None else None


def login_training_services(
    cfg: Any = None,
    *,
    workspace_dir: str | Path | None = None,
    repo_root: str | Path = REPO_ROOT,
    env_file: str | Path | None = None,
    require_env_file: bool = True,
) -> AuthLoginResult:
    result = AuthLoginResult(
        env_file=load_auth_env(
            cfg,
            workspace_dir=workspace_dir,
            repo_root=repo_root,
            env_file=env_file,
            require_env_file=require_env_file,
        )
    )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    result.hf_configured = bool(hf_token)
    if hf_token:
        try:
            from huggingface_hub import login as hf_login

            hf_login(token=hf_token, add_to_git_credential=False)
            result.hf_logged_in = True
            logger.info("Logged into Hugging Face Hub")
        except Exception as exc:
            logger.warning("Hugging Face login skipped/failed: %s", exc)

    wandb_key = os.environ.get("WANDB_API_KEY")
    result.wandb_configured = bool(wandb_key)
    if wandb_key and os.environ.get("WANDB_MODE", "").lower() != "disabled":
        try:
            import wandb

            wandb.login(key=wandb_key, relogin=True)
            result.wandb_logged_in = True
            logger.info("Logged into Weights & Biases")
        except Exception as exc:
            logger.warning("Weights & Biases login skipped/failed: %s", exc)

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-dir", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--optional-env-file", action="store_true")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--wandb-api-key-env", default="WANDB_API_KEY")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    cfg = {
        "auth": {
            "env_file": args.env_file,
            "hf_token_env": args.hf_token_env,
            "wandb_api_key_env": args.wandb_api_key_env,
        }
    }
    result = login_training_services(
        cfg,
        workspace_dir=args.workspace_dir,
        require_env_file=not args.optional_env_file,
    )

    def state(configured: bool, logged_in: bool) -> str:
        if logged_in:
            return "logged-in"
        if configured:
            return "configured"
        return "not-configured"

    logger.info(
        "Auth summary: env_file=%s hf=%s wandb=%s",
        result.env_file or "none",
        state(result.hf_configured, result.hf_logged_in),
        state(result.wandb_configured, result.wandb_logged_in),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
