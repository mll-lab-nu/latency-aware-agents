# RoboChallenge Table30v2 — Policy adapter

`model2robochallenge_interface.py` provides `RoboChallengePolicy` for the upstream
[RoboChallengeInference protocol](https://github.com/RoboChallenge/RoboChallengeInference/tree/cvpr).

## I/O contract (with the RC server)

| Direction | Endpoint | Robot=`ur5` |
|---|---|---|
| GET state | `/state.pkl` with `image_type=cam_global,cam_arm` and `action_type=leftjoint` | returns `{images: {cam: PNG}, action: joint(6)+gripper(1) = 7-d, state, pending_actions, timestamp}` |
| POST action | `/action` with `action_type=leftpos` | body `{actions: [[ee_pose(7) + gripper(1)], ...], duration: 0.05}` |

We deliberately use **two different action_types per cycle**: `leftjoint` to
fetch the joint state our model was trained on, `leftpos` to post the 8-d
ee-pose actions our model produces. The mock server treats each request
independently so this is fine.

> If you ever retrain with `state` based on ee-pose, set both
> `state_action_type` and `post_action_type` to `leftpos` in `ROBOT_SPECS`.

Robot specs (`image_types`, `state_dim`, `action_dim`) live in
`model2robochallenge_interface.py::ROBOT_SPECS` and must stay in sync with the
training data layout described in
`examples/RoboChallenge_table30v2/train_files/data_registry/data_config.py`.

## Production integration (not implemented)

Mirror upstream `demo.py` + `robot/job_worker.py::job_loop`. Replace
`DummyPolicy` with `RoboChallengePolicy(checkpoint, robot_tag=...)`. The production
loop is not wired here. The upstream integration has this form:

```python
from examples.RoboChallenge_table30v2.eval_files.model2robochallenge_interface import RoboChallengePolicy
policy = RoboChallengePolicy(args.checkpoint, robot_tag="ur5")
gpu_client = GPUClient(policy)
job_loop(client, gpu_client, args.submission_id, [224, 224],
         policy.image_type, policy.state_action_type, duration)
```

Caveat for the real `job_loop`: it uses a single `action_type` for both get/post.
You'll either need to (a) use the same action_type both ways (re-train with
matching state) or (b) sub-class `GPUClient.infer` to fetch joint state itself.
