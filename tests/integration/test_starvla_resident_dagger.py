"""Long-lived contract for controller inference through resident training ranks."""

import json
import threading
import time

from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.integrations.starvla_resident_dagger import (
    ResidentInferencePool,
    serve_resident_inference,
)
from scripts.starvla.dagger_mixture import write_dagger_mixture


class _RankPolicy:
    def __init__(self, rank):
        self.rank = rank
        self.resets = []

    def predict_batch(self, observations):
        return [
            PolicyOutput(action=Action([self.rank, observation.metadata["slot_id"]]))
            for observation in observations
        ]

    def reset_state(self, slot_id):
        self.resets.append(slot_id)


def test_resident_pool_uses_every_rank_and_survives_client_disconnect(tmp_path):
    control_dir = tmp_path / ("long-control-directory-" * 4)
    socket_dir = tmp_path / "s"
    returned = [None, None]
    policies = [_RankPolicy(rank) for rank in range(2)]

    def serve(rank):
        returned[rank] = serve_resident_inference(
            policy=policies[rank],
            control_dir=control_dir,
            socket_dir=socket_dir,
            step=500,
            rank=rank,
            world_size=2,
            stop_requested=lambda: False,
        )

    threads = [threading.Thread(target=serve, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    while not (control_dir / "step_500.ready").is_file():
        time.sleep(0.01)

    observations = [
        Observation(data=None, env_step=0, sim_time_ms=0.0, metadata={"slot_id": slot})
        for slot in range(4)
    ]
    first = ResidentInferencePool(
        socket_dir=socket_dir, step=500, worker_count=2, batch_size=1
    )
    assert [output.action.value for output in first.predict_batch(observations)] == [
        [0, 0], [1, 1], [0, 2], [1, 3]
    ]
    first.reset_state(3)
    assert [policy.resets for policy in policies] == [[], [3]]
    first.close()
    assert all(thread.is_alive() for thread in threads)

    second = ResidentInferencePool(
        socket_dir=socket_dir, step=500, worker_count=2, batch_size=4
    )
    assert [output.action.value for output in second.predict_batch(observations[:2])] == [
        [0, 0], [1, 1]
    ]
    second.close()

    command = {"metrics": {"round": 2}}
    (control_dir / "step_500.continue.json").write_text(json.dumps(command))
    for thread in threads:
        thread.join(timeout=2)
    assert returned == [command, command]


def test_replay_mixture_keeps_bootstrap_latest_and_history_weights(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    deltas = [tmp_path / f"round-{index:02d}" for index in range(1, 4)]
    destination = tmp_path / "mixture.json"
    write_dagger_mixture(bootstrap, deltas[:1], "mikasa_h8", destination)
    first = json.loads(destination.read_text())["train"]
    assert [row[1] for row in first] == [1.0, 1.0]

    write_dagger_mixture(bootstrap, deltas, "mikasa_h8", destination)
    later = json.loads(destination.read_text())
    assert [row[1] for row in later["train"]] == [1.0, 0.25, 0.25, 1.0]
    assert later["validation"] == [[str(bootstrap / "lerobot__val"), 1.0, "mikasa_h8"]]
