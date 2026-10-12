---
pipeline_tag: reinforcement-learning
tags:
  - reinforcement-learning
  - robotics
  - mikasa
---

# MIKASA InterceptGrabFast H1 PPO teacher

The small privileged-state PPO teacher used to collect the published
`InterceptGrabFast-VLA-v0` H1 rollout dataset. The task runs at 20 Hz with a 7D
`pd_ee_delta_pose` action. The checkpoint is evaluated with deterministic mean
actions.

The full setting and commands are in the
[MIKASA cookbook](https://github.com/ZihanWang314/latency-sensitive-bench/blob/master/docs/mikasa/cookbook.md).
The matching rollouts are in the
[dataset repository](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/4d8be25dc6fccfe244c2fa6be4b771ee448f2c2a/zero-latency/mikasa-intercept-grab-fast/teacher-rollouts-v1).

## Files

```text
zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/checkpoint.pt
```

`final_success_ckpt.pt` is the PyTorch checkpoint consumed by the official
MIKASA collector and evaluator. The evaluation summary records the fixed-seed
100-episode canonical evaluation. The checkpoint is 1,170,029 bytes.

## Training setting

- Observation: 49D privileged state.
- Action: 7D H1 end-effector delta pose.
- Reward: `normalized_dense`.
- Seed: 123.
- Learning rate: `1e-4`.
- PPO update epochs: 8.
- Parallel environments: 1,024.
- Batch size: 61,440.
- MIKASA integration revision: `cf4f96f319022f89c9d7cfbd639d19bc10ed44fb`
  on vendor baseline `16634db18bef08128ed79346469c86fc12169aed`.
- Training result: official early stop at 12,288,000 environment steps after
  passing the `16/16` trainer gate.

## Evaluation

The canonical strict evaluation used seeds `4242424242..4242424341`. The teacher
passed `100/100` episodes with mean return `36.819`.

After cloning the benchmark and installing its MIKASA environment:

```bash
hf download MLL-Lab/LAGEN-models \
  zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/checkpoint.pt \
  --revision e3fea81529fd98b5d5782db38803fe4062215dda \
  --local-dir outputs/mikasa/published_teacher

third_party/MIKASA-Robo/.venv/bin/python scripts/mikasa/evaluate.py \
  --policy ppo \
  --checkpoint outputs/mikasa/published_teacher/zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/checkpoint.pt \
  --episodes 100 \
  --output-dir outputs/mikasa/published_teacher_eval
```

This checkpoint is intended as a rollout teacher and zero-delay baseline for
the matching MIKASA environment. It is not the trained StarVLA student. No
license metadata is asserted here.
