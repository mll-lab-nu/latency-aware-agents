# Hopper RGB+state adaptation

Latest experiment: [September 2026 profile-latency results](index.html#profile-pipeline-20260911). Earlier L0/L2 results below refer to historical experiments with their own evaluation conditions.

Open [`index.html`](index.html) for the canonical Hopper-v4 adaptation record.
The current path uses an independent task-owned wrapper, the native three-dimensional
continuous torque space, an ordered 11-dimensional state contract, L0 Sample Factory
training, L0-L4 latency-bench evaluation, 100 accepted L0 teacher episodes,
native RGB+11D-state LeRobot conversion, and a 5k-step QwenOFT run. The final
text-bin VLA checkpoint failed closed-loop evaluation (L0 mean return 173.136;
L2 mean return 9.513). A controlled 5k-step continuous-state projector rerun
raises those means to 1,419.935 and 315.890, respectively, while remaining well
below the same-seed L0 teacher. The report preserves the original failure
diagnosis and separates representation gains from the remaining distribution gap.

Fresh profiling-only run: [20260911T023128Z-p-only4](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/1ac0e2b05094319042c7213a23ce7b42a2320cfe/hopper/profile_latency/provenance/profiling/status.md), env/obs FPS 10 / 10. Profiling and publication are complete; the subsequent full pipeline is recorded below.

- [September 11 realtime baseline](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/1ac0e2b05094319042c7213a23ce7b42a2320cfe/hopper/profile_latency/provenance/baseline/status.md): completed 20 episodes at 10 FPS; return 67.88 ± 150.09.

- [Hopper pipeline continuation](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/1ac0e2b05094319042c7213a23ce7b42a2320cfe/hopper/profile_latency/provenance/pipeline/status.md): reuses P/B; H100 completed the 10M-frame teacher recipe with the new profile at 10 FPS; simulated teacher evaluation completed at 3,657.61 ± 13.88, with all 10 episodes exceeding 3000. R/C produced 98,817 verified rows with all 100 episodes above the original gate; 123 data files are uploaded and verified; VLA completed 5000 steps with exit 0 and W&B finished (held-out L1 0.01595611); all 22 final bundle files are published and verified; matched F evaluation completed all 20 seeds: return 424.58 ± 348.01 versus B 67.88 ± 150.09 (6.26×; 20/20 paired gains), mean length 139.45, no episode reaches 1000 steps. Hopper retains strict data return >3000 and the continuous-projector VLA.

## Standard Pipeline archive

[Complete experiment archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/1ac0e2b05094319042c7213a23ce7b42a2320cfe/hopper) · [Verification receipt](assets/standard_pipeline.json). This local directory retains report display media; original records are linked at a fixed revision. Only raw demonstrations are distributed, with conversion performed on the training server.
