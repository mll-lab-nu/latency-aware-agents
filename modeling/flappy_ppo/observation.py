from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


OBS_MODES = {"image_only", "latency_channel"}


def parse_int_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


@dataclass(frozen=True)
class FlappyPpoObservationConfig:
    image_size: int = 84
    grayscale: bool = False
    stack_frames: int = 4
    obs_mode: str = "latency_channel"
    frame_skip_values: tuple[int, ...] = field(default_factory=lambda: (0, 1, 2, 3, 4, 5))

    def __post_init__(self) -> None:
        image_size = int(self.image_size)
        stack_frames = int(self.stack_frames)
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if stack_frames <= 0:
            raise ValueError("stack_frames must be positive")
        if self.obs_mode not in OBS_MODES:
            raise ValueError(f"unsupported obs_mode: {self.obs_mode}")
        frame_skip_values = tuple(max(0, int(x)) for x in self.frame_skip_values)
        if not frame_skip_values:
            raise ValueError("frame_skip_values cannot be empty")
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "grayscale", bool(self.grayscale))
        object.__setattr__(self, "stack_frames", stack_frames)
        object.__setattr__(self, "frame_skip_values", frame_skip_values)

    @property
    def base_channels(self) -> int:
        return 1 if self.grayscale else 3

    @property
    def stacked_channels(self) -> int:
        return self.base_channels * self.stack_frames

    @property
    def in_channels(self) -> int:
        return self.stacked_channels + (1 if self.obs_mode == "latency_channel" else 0)

    @property
    def max_frame_skip(self) -> int:
        return max(1, max(self.frame_skip_values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "grayscale": self.grayscale,
            "stack_frames": self.stack_frames,
            "obs_mode": self.obs_mode,
            "frame_skip_values": list(self.frame_skip_values),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FlappyPpoObservationConfig":
        return cls(
            image_size=int(data.get("image_size", 84)),
            grayscale=bool(data.get("grayscale", False)),
            stack_frames=int(data.get("stack_frames", 4)),
            obs_mode=str(data.get("obs_mode", "latency_channel")),
            frame_skip_values=tuple(parse_int_list(data.get("frame_skip_values", (0, 1, 2, 3, 4, 5)))),
        )


def observation_config_from_args(args: Any) -> FlappyPpoObservationConfig:
    return FlappyPpoObservationConfig(
        image_size=int(getattr(args, "image_size")),
        grayscale=bool(getattr(args, "grayscale")),
        stack_frames=int(getattr(args, "stack_frames")),
        obs_mode=str(getattr(args, "obs_mode")),
        frame_skip_values=tuple(parse_int_list(getattr(args, "frame_skip_values"))),
    )


def observation_config_from_mapping(data: Mapping[str, Any]) -> FlappyPpoObservationConfig:
    return FlappyPpoObservationConfig(
        image_size=int(data.get("image_size", 84)),
        grayscale=bool(data.get("grayscale", False)),
        stack_frames=int(data.get("stack_frames", 4)),
        obs_mode=str(data.get("obs_mode", "latency_channel")),
        frame_skip_values=tuple(parse_int_list(data.get("frame_skip_values", (0, 1, 2, 3, 4, 5)))),
    )
