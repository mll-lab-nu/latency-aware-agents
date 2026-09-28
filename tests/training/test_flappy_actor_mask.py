from types import SimpleNamespace

import gymnasium as gym
import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.timing import Timing


class _ActorCritic:
    action_space = gym.spaces.Discrete(2)

    def __init__(self):
        self.policy_weight = torch.nn.Parameter(torch.tensor(0.4))
        self.value_weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward_head(self, obs):
        return obs["obs"]

    def forward_core(self, head, rnn_states):
        return head, rnn_states

    def forward_tail(self, core, *, values_only, sample_actions):
        self._distribution = torch.distributions.Categorical(
            logits=torch.cat((core * self.policy_weight, -core * self.policy_weight), dim=1)
        )
        return {"values": core * self.value_weight}

    def action_distribution(self):
        return self._distribution


def _losses(admitted, advantages, *, enabled=True):
    learner = Learner.__new__(Learner)
    learner.cfg = SimpleNamespace(
        mask_unadmitted_actor=enabled,
        recurrence=1,
        ppo_clip_ratio=0.2,
        ppo_clip_value=0.2,
        use_rnn=False,
        with_vtrace=False,
        exploration_loss_coeff=0.1,
        value_loss_coeff=1.0,
    )
    learner.timing = Timing()
    learner.actor_critic = _ActorCritic()
    learner.exploration_loss_func = learner._entropy_exploration_loss
    def kl_loss(action_space, logits, distribution, valid, invalid):
        selected = distribution.logits[:, 0][valid]
        return selected, selected.mean()

    learner.kl_loss_func = kl_loss
    obs = torch.tensor([[0.0], [1.0], [2.0]])
    mb = AttrDict(
        normalized_obs={"obs": obs},
        rnn_states=torch.zeros(3, 1),
        actions=torch.zeros(3, dtype=torch.long),
        log_prob_actions=torch.full((3,), -0.7),
        action_logits=torch.zeros(3, 2),
        advantages=torch.tensor(advantages, dtype=torch.float32),
        returns=torch.tensor([1.0, 2.0, 3.0]),
        values=torch.zeros(3),
        valids=torch.ones(3, dtype=torch.bool),
        command_admitted=torch.tensor(admitted),
    )
    _, policy, entropy, _, kl, value, summaries = learner._calculate_losses(mb, 0)
    return policy, entropy, kl, value, summaries, learner


def test_dropped_advantage_does_not_change_actor_or_critic_loss():
    first = _losses([True, True, False], [1.0, 2.0, 10.0])
    changed = _losses([True, True, False], [1.0, 2.0, 10000.0])
    for index in (0, 1, 2, 3):
        torch.testing.assert_close(first[index], changed[index])
    torch.testing.assert_close(first[4]["adv_mean"], torch.tensor(1.5))

    unmasked = _losses([True, True, False], [1.0, 2.0, 10000.0], enabled=False)
    assert not torch.allclose(first[0], unmasked[0])
    assert not torch.allclose(first[2], unmasked[2])


def test_all_dropped_frames_still_train_critic():
    policy, entropy, kl, value, _, learner = _losses([False, False, False], [1.0, 2.0, 3.0])
    assert policy.item() == 0.0
    assert entropy == 0.0
    assert kl == 0.0
    assert value.item() > 0.0
    (policy + value).backward()
    assert learner.actor_critic.value_weight.grad is not None
    assert learner.actor_critic.policy_weight.grad.item() == 0.0


def test_one_admitted_frame_has_finite_actor_loss():
    policy, entropy, kl, value, summaries, _ = _losses([False, True, False], [1.0, 2.0, 3.0])
    assert torch.isfinite(policy)
    assert torch.isfinite(entropy)
    assert torch.isfinite(kl)
    assert torch.isfinite(value)
    assert summaries["adv_mean"].item() == 0.0
    assert summaries["adv_std"].item() == 1.0
