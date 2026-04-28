from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence


OBS_MODES = {"image_only", "latency_channel"}
OBSERVATION_PROTOCOLS = {"atari_frame_v1"}
PREPROCESS_BACKENDS = {"torch_stack_v1"}


def parse_int_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


@dataclass(frozen=True)
class DemonAttackPpoObservationConfig:
    image_size: int = 84
    base_channels: int = 1
    stack_frames: int = 4
    obs_mode: Literal["image_only", "latency_channel"] = "latency_channel"
    frame_skip_values: tuple[int, ...] = field(default_factory=lambda: (0, 1, 2, 3, 4, 5))
    observation_protocol: str = "atari_frame_v1"
    preprocess_backend: str = "torch_stack_v1"

    def __post_init__(self) -> None:
        image_size = int(self.image_size)
        base_channels = int(self.base_channels)
        stack_frames = int(self.stack_frames)
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if stack_frames <= 0:
            raise ValueError("stack_frames must be positive")
        if self.obs_mode not in OBS_MODES:
            raise ValueError(f"unsupported obs_mode: {self.obs_mode}")
        if self.observation_protocol not in OBSERVATION_PROTOCOLS:
            raise ValueError(
                f"unsupported observation_protocol: {self.observation_protocol}"
            )
        if self.preprocess_backend not in PREPROCESS_BACKENDS:
            raise ValueError(
                f"unsupported preprocess_backend: {self.preprocess_backend}"
            )
        frame_skip_values = tuple(max(0, int(x)) for x in self.frame_skip_values)
        if not frame_skip_values:
            raise ValueError("frame_skip_values cannot be empty")
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "base_channels", base_channels)
        object.__setattr__(self, "stack_frames", stack_frames)
        object.__setattr__(self, "frame_skip_values", frame_skip_values)
        object.__setattr__(self, "observation_protocol", str(self.observation_protocol))
        object.__setattr__(self, "preprocess_backend", str(self.preprocess_backend))

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
            "base_channels": self.base_channels,
            "stack_frames": self.stack_frames,
            "obs_mode": self.obs_mode,
            "frame_skip_values": list(self.frame_skip_values),
            "observation_protocol": self.observation_protocol,
            "preprocess_backend": self.preprocess_backend,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DemonAttackPpoObservationConfig":
        return cls(
            image_size=int(data.get("image_size", data.get("screen_size", 84))),
            base_channels=int(data.get("base_channels", 1)),
            stack_frames=int(data.get("stack_frames", 4)),
            obs_mode=str(data.get("obs_mode", "latency_channel")),  # type: ignore[arg-type]
            frame_skip_values=tuple(
                parse_int_list(data.get("frame_skip_values", (0, 1, 2, 3, 4, 5)))
            ),
            observation_protocol=str(data.get("observation_protocol", "atari_frame_v1")),
            preprocess_backend=str(data.get("preprocess_backend", "torch_stack_v1")),
        )


def observation_config_from_args(args: Any) -> DemonAttackPpoObservationConfig:
    return DemonAttackPpoObservationConfig(
        image_size=int(getattr(args, "image_size", getattr(args, "screen_size", 84))),
        base_channels=int(getattr(args, "base_channels", 1)),
        stack_frames=int(getattr(args, "stack_frames")),
        obs_mode=str(getattr(args, "obs_mode")),  # type: ignore[arg-type]
        frame_skip_values=tuple(parse_int_list(getattr(args, "frame_skip_values"))),
        observation_protocol=str(getattr(args, "observation_protocol", "atari_frame_v1")),
        preprocess_backend=str(getattr(args, "preprocess_backend", "torch_stack_v1")),
    )


def observation_config_from_mapping(
    data: Mapping[str, Any],
) -> DemonAttackPpoObservationConfig:
    return DemonAttackPpoObservationConfig.from_dict(data)
