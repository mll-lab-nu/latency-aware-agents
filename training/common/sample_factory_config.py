from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def force_sample_factory_raw_frame_step_cfg(cfg: Any) -> None:
    cfg.env_frameskip = 1
    cfg.eval_env_frameskip = 1


@dataclass(frozen=True)
class SampleFactoryConfigParser:
    configure_parser: Callable[[argparse.ArgumentParser], None]
    finalize_cfg: Callable[[Any], None]

    def parse_train(self, argv: list[str] | None = None) -> Any:
        return self._parse(argv, evaluation=False)

    def parse_eval(self, argv: list[str] | None = None) -> Any:
        return self._parse(argv, evaluation=True)

    def _parse(self, argv: list[str] | None, *, evaluation: bool) -> Any:
        from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

        parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
        self.configure_parser(parser)
        cfg = parse_full_cfg(parser, argv)
        self.finalize_cfg(cfg)
        return cfg
