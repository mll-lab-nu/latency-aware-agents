"""Long-lived native PPO regression: admission masks actor terms, not values.

Run with the MIKASA interpreter and the repository/MIKASA roots on PYTHONPATH:
python -m unittest mikasa_robo_suite.vla.dataset_collectors.test_ppo_latency
"""

from types import SimpleNamespace
import unittest

import torch

from mikasa_robo_suite.vla.dataset_collectors.get_dataset_collectors_ckpt import ppo_losses


class PpoAdmissionTest(unittest.TestCase):
    def test_zero_latency_partial_admission_and_busy_batch(self):
        args = SimpleNamespace(clip_vloss=False, clip_coef=0.2, norm_adv=False)
        for admission in ([True, True, True], [True, False, True], [False, False, False]):
            with self.subTest(admission=admission):
                admitted = torch.tensor(admission)
                logprobs = torch.zeros(3, requires_grad=True)
                entropy = torch.ones(3, requires_grad=True)
                values = torch.zeros(3, requires_grad=True)
                pg, vf, ent, old_kl, kl, clipfrac = ppo_losses(
                    args, logprobs, entropy, values, torch.zeros(3),
                    torch.tensor([1., 2., 3.]), torch.ones(3), torch.zeros(3), admitted,
                )
                (pg + vf - 0.01 * ent).backward()
                torch.testing.assert_close(values.grad, torch.full((3,), -1 / 3))
                if admitted.any():
                    expected = -torch.tensor([1., 2., 3.]) * admitted / admitted.sum()
                    torch.testing.assert_close(logprobs.grad, expected)
                    torch.testing.assert_close(entropy.grad, -0.01 * admitted / admitted.sum())
                else:
                    self.assertIsNone(logprobs.grad)
                    self.assertIsNone(entropy.grad)
                for metric in (old_kl, kl, clipfrac):
                    self.assertEqual(metric.item(), 0)

    def test_busy_samples_do_not_change_normalization_entropy_or_kl(self):
        args = SimpleNamespace(clip_vloss=False, clip_coef=0.2, norm_adv=True)
        inputs = (
            torch.tensor([0.1, 100., -0.1]), torch.tensor([1., 100., 2.]),
            torch.zeros(3), torch.zeros(3), torch.tensor([1., 1e9, 3.]),
            torch.ones(3), torch.zeros(3),
        )
        admitted = torch.tensor([True, False, True])
        actual = ppo_losses(args, *inputs, admitted)
        expected = ppo_losses(args, *(x[admitted] for x in inputs), torch.ones(2, dtype=torch.bool))
        for actual_term, expected_term in zip(actual, expected):
            torch.testing.assert_close(actual_term, expected_term)


if __name__ == "__main__":
    unittest.main()
