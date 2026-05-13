from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


OBSERVATION_PROTOCOLS = {"raw_rgb_v1"}
PREPROCESS_BACKENDS = {"torch"}


def parse_int_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


@dataclass(frozen=True)
class FlappyPpoObservationConfig:
    image_size: int = 84
    grayscale: bool = False
    stack_frames: int = 4
    observation_protocol: str = "raw_rgb_v1"
    preprocess_backend: str = "torch"

    def __post_init__(self) -> None:
        image_size = int(self.image_size)
        stack_frames = int(self.stack_frames)
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if stack_frames <= 0:
            raise ValueError("stack_frames must be positive")
        if self.observation_protocol not in OBSERVATION_PROTOCOLS:
            raise ValueError(f"unsupported observation_protocol: {self.observation_protocol}")
        if self.preprocess_backend not in PREPROCESS_BACKENDS:
            raise ValueError(f"unsupported preprocess_backend: {self.preprocess_backend}")
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "grayscale", bool(self.grayscale))
        object.__setattr__(self, "stack_frames", stack_frames)
        object.__setattr__(self, "observation_protocol", str(self.observation_protocol))
        object.__setattr__(self, "preprocess_backend", str(self.preprocess_backend))

    @property
    def base_channels(self) -> int:
        return 1 if self.grayscale else 3

    @property
    def stacked_channels(self) -> int:
        return self.base_channels * self.stack_frames

    @property
    def in_channels(self) -> int:
        return self.stacked_channels

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "grayscale": self.grayscale,
            "stack_frames": self.stack_frames,
            "observation_protocol": self.observation_protocol,
            "preprocess_backend": self.preprocess_backend,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FlappyPpoObservationConfig":
        return observation_config_from_mapping(data)


def observation_config_from_args(args: Any) -> FlappyPpoObservationConfig:
    return FlappyPpoObservationConfig(
        image_size=int(getattr(args, "image_size")),
        grayscale=bool(getattr(args, "grayscale")),
        stack_frames=int(getattr(args, "stack_frames")),
        observation_protocol=str(getattr(args, "observation_protocol", "raw_rgb_v1")),
        preprocess_backend=str(getattr(args, "preprocess_backend", "torch")),
    )


def observation_config_from_mapping(data: Mapping[str, Any]) -> FlappyPpoObservationConfig:
    return FlappyPpoObservationConfig(
        image_size=int(data.get("image_size", 84)),
        grayscale=bool(data.get("grayscale", False)),
        stack_frames=int(data.get("stack_frames", 4)),
        observation_protocol=str(data.get("observation_protocol", "raw_rgb_v1")),
        preprocess_backend=str(data.get("preprocess_backend", "torch")),
    )
