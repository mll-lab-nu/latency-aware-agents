"""Scheme B (variant b) per-frame KV-memory supervision logic.

These tests pin the bookkeeping of `Qwenvl_OFT._forward_memory` WITHOUT loading
the 4B Qwen3-VL checkpoint: the heavy collaborators (streaming forward, action head,
loss) are stubbed, and the method is invoked with a stub `self`. They assert the parts
that are easy to get wrong:

- clamp/padding frames are never fed to memory and never supervised (`valid` mask),
- every real frame is supervised once, at the window it naturally has (1 -> window),
- the per-step target is that frame's own action chunk (`actions_per_frame[k]`),
- same (instruction, valid) rollouts batch together; different ones do not,
- the per-step density weights drive the final weighted-mean loss,
- for an episode-start sample the fed-frame order equals eval's (cadence consistency).

Run with the starVLA env, e.g.:
  /home/lixinyuan/miniconda3/envs/starvla_rl_games_gr00t/bin/python -m pytest \
    starVLA/tests/test_kv_memory_scheme_b.py -q
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT


class _StubMemory:
    """Minimal stand-in for FrameKVMemory: only detach_ is exercised here."""

    def detach_(self):
        return None


class _ActionHead:
    def predict_action(self, queries):
        # queries is [B, *]; return a [B, 1, dim] tensor (content irrelevant: the stub
        # loss reads the target, not the prediction).
        batch = queries.shape[0]
        return torch.zeros(batch, 1, 2)


class _StubQwenOFT:
    """Carries exactly the attributes/methods `_forward_memory` touches."""

    action_horizon = 1
    action_env_dim = 2
    action_token_id = 0

    def __init__(self):
        self.action_model = _ActionHead()
        self.step_calls = []  # (instruction, frames, batch_size) per forward step

    # --- profiling (disabled) ---
    def _profile_timing_enabled(self):
        return False

    # --- prompt / memory plumbing ---
    def _instruction_with_state(self, lang, state):
        return lang

    def _new_kv_memory(self):
        return _StubMemory()

    def _kv_forward_step_batched(self, memories, frames, instruction):
        self.step_calls.append((instruction, list(frames), len(memories)))
        batch = len(memories)
        return torch.zeros(batch, 1, 4), torch.zeros(batch, 1, dtype=torch.long)

    def _gather_action_token_embeddings(self, last_hidden, new_ids, action_token_id=None):
        return last_hidden

    def _prepare_action_array(self, action):
        return np.asarray(action)

    # Stub loss: mean of the TARGET so the returned scalar encodes which label was used.
    def _compute_action_loss(self, pred_actions, actions_target, action_env_dims=None, rl_games_tasks=None):
        return actions_target.float().mean()


def _example(lang, frames, valid, actions_per_frame, density_weight):
    return {
        "lang": lang,
        "image": list(frames),
        "valid": np.asarray(valid, dtype=bool),
        "actions_per_frame": np.asarray(actions_per_frame, dtype=np.float32),
        "density_weight": np.asarray(density_weight, dtype=np.float32),
    }


def _run(examples):
    stub = _StubQwenOFT()
    out = Qwenvl_OFT._forward_memory(stub, examples)
    return stub, out


def test_padding_frames_skipped_and_every_real_frame_supervised():
    # R=5, first two frames are clamp padding (episode start, base_index=2).
    frames = ["f0", "f1", "f2", "f3", "f4"]
    valid = [False, False, True, True, True]
    actions = [[[9.0, 9.0]], [[9.0, 9.0]], [[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]]]
    weights = [9.0, 9.0, 1.0, 1.0, 1.0]
    stub, _ = _run([_example("play", frames, valid, actions, weights)])

    # Exactly the three real frames are fed, in chronological order. The padding
    # frames f0/f1 never reach the memory.
    fed = [call[1][0] for call in stub.step_calls]
    assert fed == ["f2", "f3", "f4"]
    assert len(stub.step_calls) == 3


def test_episode_start_cadence_matches_eval_frame_order():
    # Eval at an episode start feeds real frames one-by-one into an empty memory,
    # growing the window 1->2->3. Scheme-B training must feed the SAME frames in the
    # same order for an is_start sample.
    frames = ["a", "b", "c", "d"]
    valid = [False, True, True, True]
    actions = [[[0.0, 0.0]]] * 4
    weights = [1.0, 1.0, 1.0, 1.0]
    stub, _ = _run([_example("go", frames, valid, actions, weights)])

    fed_order = [call[1][0] for call in stub.step_calls]
    assert fed_order == ["b", "c", "d"]  # window grows 1,2,3 over exactly these


def test_target_is_per_frame_action_and_density_weighted():
    # Single fully-real rollout: loss = sum_k mean(action_k) * w_k / sum_k w_k.
    frames = ["f0", "f1", "f2"]
    valid = [True, True, True]
    actions = [[[2.0, 2.0]], [[4.0, 4.0]], [[6.0, 6.0]]]  # means: 2, 4, 6
    weights = [1.0, 2.0, 3.0]
    _, out = _run([_example("p", frames, valid, actions, weights)])

    expected = (2.0 * 1.0 + 4.0 * 2.0 + 6.0 * 3.0) / (1.0 + 2.0 + 3.0)
    assert out["action_loss"].item() == pytest.approx(expected)
    assert out["loss_weight"] == pytest.approx(1.0 + 2.0 + 3.0)


def test_same_layout_batches_and_different_valid_splits():
    # Two samples share instruction AND valid -> one batched forward per step (B=2).
    a = _example("same", ["x0", "x1"], [True, True], [[[1.0, 1.0]], [[1.0, 1.0]]], [1.0, 1.0])
    b = _example("same", ["y0", "y1"], [True, True], [[[1.0, 1.0]], [[1.0, 1.0]]], [1.0, 1.0])
    stub, _ = _run([a, b])
    assert all(call[2] == 2 for call in stub.step_calls)
    assert len(stub.step_calls) == 2  # two steps, each batched over both samples

    # Different valid patterns must NOT batch together: they evolve different windows.
    c = _example("same", ["z0", "z1"], [False, True], [[[1.0, 1.0]], [[1.0, 1.0]]], [1.0, 1.0])
    stub2, _ = _run([a, c])
    assert {call[2] for call in stub2.step_calls} == {1}  # each runs per-sample


def test_returns_timing_and_batch_stats_keys():
    # The trainer reads output_dict["timing"]/["batch_stats"] when profiling.
    a = _example("p", ["f0"], [True], [[[0.0, 0.0]]], [1.0])
    _, out = _run([a])
    assert "timing" in out and "batch_stats" in out


def test_mixture_dataset_preserves_kv_memory_fields_from_single_dataset():
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset

    class SingleDataset:
        dataset_name = "fake"
        data_cfg = None
        lerobot_info_meta = {"total_videos": 0}
        modality_keys = {"video": ["video.image"]}

        def get_step_data(self, trajectory_id, base_index):
            return {"trajectory_id": trajectory_id, "base_index": base_index}

        def transforms(self, raw_data):
            return raw_data

        def _pack_sample(self, data, trajectory_id=None, base_index=None):
            if trajectory_id is None or base_index is None:
                return {"action": np.array([[0.0]], dtype=np.float32), "image": ["frame"], "lang": "play"}
            return {
                "action": np.array([[0.0]], dtype=np.float32),
                "image": ["frame"],
                "lang": "play",
                "valid": np.array([True], dtype=bool),
                "actions_per_frame": np.array([[[1.0]]], dtype=np.float32),
                "density_weight": np.array([1.0], dtype=np.float32),
            }

        def _attach_rl_games_metadata(self, sample, base_index):
            sample["latency"] = 2

    mixture = LeRobotMixtureDataset.__new__(LeRobotMixtureDataset)
    mixture._getitem_count = 0
    mixture.datasets = [SingleDataset()]
    mixture.sample_step = lambda index: (mixture.datasets[0], 7, 3)

    sample = LeRobotMixtureDataset.__getitem__(mixture, 0)

    assert sample["valid"].tolist() == [True]
    assert sample["actions_per_frame"].shape == (1, 1, 1)
    assert sample["density_weight"].tolist() == [1.0]
    assert sample["latency"] == 2
