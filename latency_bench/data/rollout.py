"""Collect canonical teacher demonstrations through shared episode and storage rules."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm
import yaml

from latency_bench.data import rollout_probe as probe
from latency_bench.data.action_layouts import ACTION_LAYOUT_G1_SONIC_ABSOLUTE_JOINT_TARGET_V1, ACTION_LAYOUT_GYMNASIUM_CONTINUOUS_V1
from latency_bench.data.filters import accepts_episode, accepts_episode_contract, filter_train_rows
from latency_bench.data.ghost_trail import GhostTrailConfig
from latency_bench.data.hf_parquet import (
    completed_episode_indices_from_shards, write_episode_row_shard,
)
from latency_bench.data.rollout_episode import episode_image_dir, finalize_attempt_rows, write_filtered_episode_images
from latency_bench.data.rollout_export import finalize_dataset
from latency_bench.data.rollout_image_io import RolloutImageWriter, build_episode_image_frames
from latency_bench.data.rollout_rows import teacher_dataset_row
from latency_bench.utils.io import write_json_atomic


class CollectionSession:
    """Own episode allocation, acceptance, paired samples and durable episode shards.

    Execution adapters request work per free slot and report decisions and completed
    episodes. They do not choose splits, filter rows or write collection artifacts.
    """

    def __init__(self, *, phase, output_dir, task, options, probe_state, execution_metadata):
        self.phase = phase
        self.capture = phase != "probe"
        self.output_dir = output_dir
        self.task = task
        self.options = options
        self.state = probe_state
        self.runtime = execution_metadata
        self.clock_stride = round(self.runtime["env_fps"] / self.runtime["obs_fps"])
        self.splits = probe.build_episode_plan(
            max_episodes=options["max_episodes"], val_fraction=options["val_fraction"],
            split_seed=options["split_seed"], base_seed=0,
        )
        completed = completed_episode_indices_from_shards(output_dir)
        self.remaining = iter(spec for spec in self.state["accepted_specs"]
                              if spec["episode_idx"] not in completed)
        self.episodes = {}
        self.specs = {}
        self.image_writer = RolloutImageWriter(
            max_workers=options["image_writer_workers"], row_buffer_size=options["row_buffer_size"],
            export_image_size=options["export_image_size"],
        ) if self.capture else None
        self.progress = tqdm(total=options["max_episodes"], unit="episode", initial=(
            len(completed) if phase == "replay" else len(self.state["accepted_specs"])
        ))

    @property
    def target_reached(self):
        return self.phase != "replay" and len(self.state["accepted_specs"]) >= self.options["max_episodes"]

    def next_episode(self):
        if self.phase == "replay":
            spec = next(self.remaining, None)
            if spec is None:
                return None
        else:
            if self.target_reached:
                return None
            attempt = self.state["next_attempt_idx"]
            budget = self.options["max_attempts"]
            if budget is not None and attempt >= budget:
                return None
            spec = probe._attempt_episode_spec(attempt_idx=attempt, base_seed=self.options["base_seed"])
            self.state["next_attempt_idx"] += 1
        self.specs[spec["episode_idx"]] = spec
        if self.capture:
            self.episodes[spec["episode_idx"]] = {"rows": [], "frames": {}}
        return spec

    def decision(self, episode_id, *, decision_step, image, state, issued_action, info, reward):
        if not self.capture:
            return
        episode = self.episodes[episode_id]
        spec = self.specs[episode_id]
        image_path = episode_image_dir(self.output_dir, spec["split"], episode_id) / f"decision_step_{decision_step:06d}.png"
        episode["rows"].append(teacher_dataset_row(
            env_name=self.task["env_name"], spec=spec, decision_step=decision_step,
            image=str(image_path), raw_action=issued_action, info=info, reward=reward,
            env_fps=self.runtime["env_fps"], obs_fps=self.runtime["obs_fps"],
            action_spec=self.task["action_spec"], base_prompt=self.task["base_prompt"],
            prompt_override=self.task["prompt_override"], state=state,
            image_transform=self.options["image_transform_config"].image_transform,
        ))
        # The adapter snapshots before advancing. Ownership transfers with the sample.
        episode["frames"][decision_step] = image

    def complete_episode(self, episode_id, summary):
        spec = self.specs.pop(episode_id)
        episode = self.episodes.pop(episode_id) if self.capture else None
        if self.target_reached:
            return
        if self.phase == "replay":
            if summary["episode_length"] != spec["episode_length"]:
                raise RuntimeError(
                    f"Probe/replay length mismatch for seed={spec['seed']}: "
                    f"probe={spec['episode_length']} replay={summary['episode_length']}"
                )
            accepted_spec = spec
        else:
            self.state["attempted_episodes"] += 1
            accepted = accepts_episode(
                self.options["filter_preset"], self.task["integration_name"], summary,
                max_steps_per_episode=self.options["max_steps_per_episode"],
                demon_episode_threshold=self.options["demon_episode_threshold"],
            )
            if self.options["episode_acceptance"] is not None:
                accepted = accepted and accepts_episode_contract(
                    summary, self.options["episode_acceptance"], env_fps=self.runtime["env_fps"],
                )
            accepted = accepted and probe._passes_episode_return_gate(
                summary, episode_return_field=self.task["action_spec"].episode_return_field,
                episode_return_gt=self.options["episode_return_gt"],
            )
            if accepted:
                index = len(self.state["accepted_specs"])
                accepted_spec = {
                    **{key: value for key, value in summary.items() if key != "final_info"},
                    "attempt_idx": spec["attempt_idx"], "episode_idx": index,
                    "seed": spec["seed"], "split": self.splits[index]["split"],
                }
                self.state["accepted_specs"].append(accepted_spec)
            else:
                self.state["rejected_episodes"] += 1
                accepted_spec = None
        if accepted_spec is not None:
            if self.capture:
                self._write_episode(accepted_spec, episode)
            self.progress.update(1)
        if self.phase != "replay":
            if (self.options["max_attempts"] is not None or accepted_spec is not None
                    or self.state["attempted_episodes"] % probe.PROBE_STATE_WRITE_EVERY == 0):
                probe.write_probe_state(self.output_dir, self.state)
            probe._update_probe_progress(self.progress, self.state, allocated_attempts=self.state["next_attempt_idx"])

    def _write_episode(self, spec, episode):
        rows = finalize_attempt_rows(
            episode["rows"], final_dir=episode_image_dir(self.output_dir, spec["split"], spec["episode_idx"]),
            episode_idx=spec["episode_idx"], split=spec["split"],
        )
        raw_trace = self.task["action_spec"].layout == ACTION_LAYOUT_G1_SONIC_ABSOLUTE_JOINT_TARGET_V1
        selected = (filter_train_rows(self.options["filter_preset"], self.task["integration_name"], rows,
                                      seed=spec["seed"])
                    if spec["split"] == "train" and not raw_trace else rows)
        frames = episode["frames"]
        if not self.task["images_pretransformed"]:
            frames = build_episode_image_frames(
                env_name=self.task["integration_name"], rows=rows,
                frames=[frames[row["decision_step"]] for row in rows],
                config=self.options["image_transform_config"],
            )
        selected = write_filtered_episode_images(
            self.image_writer, selected, frames, context_window=self.options["context_window"],
        )
        # A shard is a completion marker, so image failures must precede it.
        self.image_writer.flush()
        write_episode_row_shard(self.output_dir, selected, episode_spec=spec)

    def close(self):
        try:
            if self.image_writer is not None:
                self.image_writer.close()
        finally:
            self.progress.close()
            self.episodes.clear()
            self.specs.clear()


def collect_dataset(
    *, execution_factory: Callable, task: dict, output_dir: Path,
    max_episodes: int, max_steps_per_episode: int, val_fraction: float, split_seed: int,
    num_envs: int, base_seed: int | None, two_pass: bool,
    max_attempts: int | None = None, replay_num_envs: int = 16,
    export_image_size=None, image_writer_workers=0, row_buffer_size=256,
    filter_preset="none", demon_episode_threshold=None, episode_return_gt=None,
    episode_acceptance=None, context_window=1, probe_only=False,
    image_transform_config=None, preserve_replay_slots=False,
) -> dict:
    """Run shared collection phases; adapters only construct and execute environments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0 <= val_fraction <= 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if max_episodes < 1 or max_steps_per_episode < 1 or num_envs < 1 or replay_num_envs < 1:
        raise ValueError("episode, step and environment counts must be positive")
    options = dict(
        max_episodes=max_episodes, max_steps_per_episode=max_steps_per_episode,
        val_fraction=val_fraction, split_seed=split_seed, num_envs=num_envs,
        replay_num_envs=replay_num_envs, base_seed=base_seed, max_attempts=max_attempts,
        export_image_size=export_image_size, image_writer_workers=image_writer_workers,
        row_buffer_size=row_buffer_size, filter_preset=filter_preset,
        demon_episode_threshold=demon_episode_threshold, episode_return_gt=episode_return_gt,
        episode_acceptance=episode_acceptance, context_window=context_window,
        image_transform_config=GhostTrailConfig() if image_transform_config is None else image_transform_config,
    )
    completed = completed_episode_indices_from_shards(output_dir)
    state = probe.load_probe_state(output_dir, completed_episode_indices=completed)
    if (completed and task["action_spec"].layout == ACTION_LAYOUT_GYMNASIUM_CONTINUOUS_V1
            and state.get("label_semantics") != "teacher_issued_command"):
        raise ValueError("Existing continuous-action shards have legacy labels; restart replay in a new output directory")
    state["label_semantics"] = "teacher_issued_command"

    def run_phase(phase, slots):
        execution = execution_factory(capture=phase != "probe", num_envs=slots)
        session = None
        try:
            state["runtime_metadata"] = dict(execution.metadata)
            if base_seed is None:
                options["base_seed"] = execution.seed
            probe.write_probe_state(output_dir, state)
            session = CollectionSession(phase=phase, output_dir=output_dir, task=task, options=options,
                                        probe_state=state, execution_metadata=execution.metadata)
            execution.run(session)
        finally:
            try:
                if session is not None:
                    session.close()
            finally:
                execution.close()

    if len(state["accepted_specs"]) < max_episodes:
        run_phase("probe" if two_pass else "direct", num_envs)
    if probe_only:
        return state
    if len(state["accepted_specs"]) < max_episodes:
        write_json_atomic(output_dir / "incomplete_probe.json", {
            "status": "incomplete_attempt_budget_exhausted", "requested_episodes": max_episodes,
            "accepted_episodes": len(state["accepted_specs"]), "attempted_episodes": state["attempted_episodes"],
            "rejected_episodes": state["rejected_episodes"], "attempt_budget": max_attempts,
        })
        raise RuntimeError(f"Probe accepted {len(state['accepted_specs'])}/{max_episodes} episodes "
                           f"within max_attempts={max_attempts}; replay was not started.")
    if two_pass:
        completed = completed_episode_indices_from_shards(output_dir)
        remaining = [spec for spec in state["accepted_specs"] if spec["episode_idx"] not in completed]
        if remaining:
            if preserve_replay_slots and completed:
                raise RuntimeError("GPU history replay must start from all probe slots; clear partial "
                                   "episode_rows and images before replaying again")
            slots = replay_num_envs if preserve_replay_slots else min(replay_num_envs, len(remaining))
            run_phase("replay", slots)
    return state


def collect_rollouts(config: dict[str, Any]) -> dict[str, Any]:
    """Collect student observations and current teacher commands as canonical Parquet.

    Configured external environment/policy factories remain the user extension
    points. Framework adapters use the same collect_dataset/finalize_dataset pipeline.
    """
    # The shared SF coordinator does not need evaluation factory dependencies.
    from latency_bench.data.rollout_config import resolve_collection
    from latency_bench.data.rollout_executor import ExecutorCollectionExecution

    config, task, options, source_metadata = resolve_collection(config)
    output_dir = Path(config["logging"]["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not probe.probe_state_path(output_dir).exists():
        raise FileExistsError(f"Collection output already exists without resumable state: {output_dir}")
    def execution_factory(*, capture, num_envs):
        return ExecutorCollectionExecution(config, num_envs=num_envs)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    state = collect_dataset(execution_factory=execution_factory, task=task, output_dir=output_dir, **options)
    if options["probe_only"]:
        return {"output_dir": str(output_dir), "probe_state": state}
    metadata = finalize_dataset(output_dir=output_dir, task=task, options=options,
                                probe_state=state, source_metadata=source_metadata)
    return {"output_dir": str(output_dir), "metadata": metadata}
