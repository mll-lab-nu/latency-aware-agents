"""Execute Sample Factory decisions for the shared demonstration collector."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from latency_bench.core.decision_action_history import DecisionActionHistory
from latency_bench.data.ghost_trail import GhostTrailConfig, DemonGhostTrailStream
from latency_bench.data.rollout import collect_dataset
from latency_bench.data.rollout_image_io import frame_from_env_raw_rgb_info, frames_from_env_raw_rgb_info
from latency_bench.integrations.sample_factory import teacher_runtime
from latency_bench.integrations.sample_factory.teacher_drivers import BatchedTeacherRolloutDriver, SerialTeacherRolloutDriver


class TeacherCollectionExecution:
    """Own SF inference, recurrent slots and decision-stride environment execution."""

    def __init__(self, config, *, output_dir, num_envs, capture, deterministic, image_transform_config,
                 rollout_seed_offset=0):
        self.config = config
        self.capture = capture
        self.deterministic = deterministic
        self.transform = image_transform_config
        self.cfg, self.runtime, self.driver, frame_stack = teacher_runtime.build_rollout_driver_runtime(
            config, output_dir=output_dir, num_envs=num_envs, export_env_raw_rgb_frames=capture,
        )
        self.seed = int(self.cfg.seed) + rollout_seed_offset
        self.metadata = {
            "checkpoint_train_step": int(self.runtime["checkpoint"]["train_step"]),
            "frame_stack": int(frame_stack), "env_fps": float(self.cfg.env_fps),
            "obs_fps": float(self.cfg.obs_fps),
        }

    def close(self):
        self.driver.close()

    def run(self, session):
        driver = self.driver
        cfg = self.cfg
        integration = self.config["integration_name"]
        env_name = self.config["env_name"]
        batched = isinstance(driver, BatchedTeacherRolloutDriver)
        g1 = env_name.startswith("g1_sonic_")
        include_state = integration == "gymnasium" and bool(cfg.gym_state_labels_json)
        history = (integration in {"flappy", "demon_attack", "deadly_corridor"}
                   or env_name in teacher_runtime._GYMNASIUM_ATARI_GPU_ENVS) and getattr(cfg, "action_history_decisions", None) is not None
        zero_history = None
        if self.capture and "export_zero_action_history" in self.config:
            if cfg.latency_type != "zero" and not (cfg.latency_type == "fixed" and cfg.fixed_latency_ms == 0):
                raise ValueError("Export-only zero action history requires fixed latency 0 ms")
            zero_history = DecisionActionHistory(driver.action_space, num_envs=driver.slot_count,
                                                decisions=self.config["export_zero_action_history"])
        ghost = self.capture and integration == "demon_attack" and self.transform.image_transform == "demon_attack_ghost_trail"
        specs = [None] * driver.slot_count
        observations = [None] * driver.slot_count
        infos = [None] * driver.slot_count
        frames = [None] * driver.slot_count
        streams = [None] * driver.slot_count
        lengths = [0] * driver.slot_count
        returns = [0.0] * driver.slot_count
        raw_returns = [0.0] * driver.slot_count
        rnn = torch.zeros((driver.slot_count, self.runtime["rnn_size"]), dtype=torch.float32, device=self.runtime["device"])

        def allocate(slots):
            plan = {}
            for slot in slots:
                specs[slot] = session.next_episode()
                if specs[slot] is not None:
                    plan[slot] = specs[slot]["seed"]
            return plan

        def apply_resets(plan):
            if not plan:
                return
            results = (driver.reset_slots(plan, [spec is not None for spec in specs]) if batched
                       else driver.reset_slots(plan))
            if zero_history is not None:
                zero_history.reset(list(plan))
            for slot, (obs, info) in results.items():
                observations[slot], infos[slot] = obs, info
                lengths[slot] = 0
                returns[slot] = raw_returns[slot] = 0.0
                rnn[slot].zero_()
                if self.capture:
                    frames[slot] = frame_from_env_raw_rgb_info(info)
                    if ghost:
                        streams[slot] = DemonGhostTrailStream(self.transform.history_frames)

        apply_resets(allocate(range(driver.slot_count)))
        while any(spec is not None for spec in specs) and not session.target_reached:
            slots = [slot for slot, spec in enumerate(specs) if spec is not None]
            # Snapshot only student inputs, before env.step can reuse the buffers.
            images = {slot: np.array(frames[slot], copy=True) for slot in slots} if self.capture else {}
            states = {}
            if self.capture:
                exported_histories = zero_history.observation() if zero_history is not None else None
                for slot in slots:
                    state = (exported_histories[slot] if zero_history is not None
                             else observations[slot]["transport"] if history
                             else infos[slot]["gymnasium_state"] if include_state else None)
                    states[slot] = None if state is None else np.array(state, copy=True)
            actions, next_rnn, results = teacher_runtime.step_teacher_policy(
                self.runtime, driver, active_slots=slots, active_obs=observations, rnn_states=rnn,
                deterministic=self.deterministic, preserve_leading_batch=integration == "gymnasium",
            )
            rnn[slots] = next_rnn
            if zero_history is not None:
                zero_history.append(slots, np.ones(len(slots)), actions, actions)
            reset_plan = {}
            for index, slot in enumerate(slots):
                if session.target_reached:
                    specs[slot] = None
                    continue
                spec, result = specs[slot], results[slot]
                step = lengths[slot]
                lengths[slot] += 1
                returns[slot] += float(result.reward)
                if integration == "demon_attack":
                    raw_returns[slot] += float(result.info["raw_reward"])
                life_loss, done = teacher_runtime.rollout_step_flags(
                    integration, result, next_raw_frame=lengths[slot] * session.clock_stride,
                    max_steps_per_episode=session.options["max_steps_per_episode"],
                )
                if self.capture:
                    issued = actions[index]
                    if g1:
                        env = driver.envs[slot] if isinstance(driver, SerialTeacherRolloutDriver) else driver.control_env
                        native = env.unwrapped
                        issued = np.asarray(issued, dtype=np.float64).reshape(-1)
                        if native._control_mode == "raw_body_target":
                            issued = native.g1_absolute_request_from_raw_action(issued)
                    session.decision(
                        spec["episode_idx"], decision_step=step, image=images[slot], state=states[slot],
                        issued_action=issued, info=result.info | {"raw_done": result.terminated or result.truncated},
                        reward=result.reward,
                    )
                if done:
                    summary = {
                        "episode_length": lengths[slot], "episode_raw_frames": lengths[slot] * session.clock_stride,
                        session.task["action_spec"].episode_return_field: returns[slot],
                        "termination_reason": "terminated" if result.terminated and not life_loss
                        else "truncated" if result.truncated else "horizon",
                        "final_info": result.info,
                    }
                    if integration == "demon_attack":
                        summary["episode_raw_return"] = raw_returns[slot]
                    session.complete_episode(spec["episode_idx"], summary)
                    reset_plan.update(allocate([slot]))
                    observations[slot] = infos[slot] = frames[slot] = streams[slot] = None
                    rnn[slot].zero_()
                elif life_loss:
                    observations[slot], infos[slot] = driver.continue_after_life_loss(slot)
                    rnn[slot].zero_()
                    if zero_history is not None:
                        zero_history.reset([slot])
                    if self.capture:
                        frames[slot] = frame_from_env_raw_rgb_info(infos[slot])
                        if ghost:
                            streams[slot].clear()
                else:
                    observations[slot], infos[slot] = result.next_obs, result.info
                    if self.capture:
                        if ghost:
                            for frame in frames_from_env_raw_rgb_info(result.info):
                                streams[slot].append(frame)
                            frames[slot] = streams[slot].render()
                        else:
                            frames[slot] = frame_from_env_raw_rgb_info(result.info)
            if session.target_reached:
                break
            if reset_plan:
                apply_resets(reset_plan)
            elif batched:
                driver.sync_active([spec is not None for spec in specs])


def collection_task(config):
    """Translate checkpoint observation and prompt choices into the common task contract."""
    flat = config["flat_cfg"]
    history = ("export_zero_action_history" in config or (
        (config["integration_name"] in {"flappy", "demon_attack", "deadly_corridor"}
         or config["env_name"] in teacher_runtime._GYMNASIUM_ATARI_GPU_ENVS)
        and flat.get("action_history_decisions") is not None))
    return {
        "env_name": config["env_name"], "integration_name": config["integration_name"],
        "action_spec": config["action_spec"], "base_prompt": config["base_prompt"],
        "prompt_override": config["base_prompt"] if history or config["env_name"].startswith("g1_sonic_") else None,
        "images_pretransformed": False, "image_source": "env_raw_rgb",
    }


def collect_teacher_dataset(config, *, output_dir: Path, deterministic, rollout_seed_offset=0,
                            image_transform_config=None, two_pass=True, **options):
    """Translate SF runtime construction into the shared collection pipeline."""
    transform = GhostTrailConfig() if image_transform_config is None else image_transform_config
    options.setdefault("filter_preset", "clean_v1")
    task = collection_task(config)
    task["images_pretransformed"] = (config["integration_name"] == "demon_attack"
                                      and transform.image_transform == "demon_attack_ghost_trail")
    flat = config["flat_cfg"]
    preserve_slots = ((config["integration_name"] in {"flappy", "demon_attack"}
                       or config["env_name"] in teacher_runtime._GYMNASIUM_ATARI_GPU_ENVS)
                      and ("export_zero_action_history" in config or flat.get("action_history_decisions") == 8))

    def factory(*, capture, num_envs):
        return TeacherCollectionExecution(config, output_dir=output_dir, num_envs=num_envs,
                                          capture=capture, deterministic=deterministic,
                                          image_transform_config=transform, rollout_seed_offset=rollout_seed_offset)

    return collect_dataset(execution_factory=factory, task=task, output_dir=output_dir,
                           base_seed=None, two_pass=two_pass, preserve_replay_slots=preserve_slots,
                           image_transform_config=transform, **options)
