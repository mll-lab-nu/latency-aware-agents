"""Long-lived contract for controller inference through resident training ranks."""

import json
import threading
import time

from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.integrations.starvla_resident_dagger import (
    ResidentInferencePool,
    serve_resident_inference,
)


class _RankPolicy:
    def __init__(self, rank):
        self.rank = rank

    def predict_batch(self, observations):
        return [
            PolicyOutput(action=Action([self.rank, observation.metadata["slot_id"]]))
            for observation in observations
        ]


def test_resident_pool_uses_every_rank_and_survives_client_disconnect(tmp_path):
    control_dir = tmp_path / ("long-control-directory-" * 4)
    socket_dir = tmp_path / "s"
    returned = [None, None]

    def serve(rank):
        returned[rank] = serve_resident_inference(
            policy=_RankPolicy(rank),
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
