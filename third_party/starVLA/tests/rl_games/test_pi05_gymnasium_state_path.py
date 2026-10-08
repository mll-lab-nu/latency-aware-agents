"""CPU contracts: real QwenPI routing and small DiT; synthetic VLM features/data."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch
from torch import nn

pi05 = importlib.import_module("starVLA.model.framework.VLM4A.QwenPI_v3")


@pytest.mark.parametrize(
    ("state_dim", "action_dim"),
    [(0, 6), (4, 5), (11, 3), (17, 6), (17, 6), (27, 8), (376, 17)],
    ids=["air_raid", "inverted_pendulum", "hopper", "half_cheetah", "walker2d", "ant", "humanoid"],
)
def test_native_state_forward_backward_prediction_reload(monkeypatch, tmp_path, state_dim, action_dim):
    class SyntheticVLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=48, num_hidden_layers=2))
            self.features = nn.Parameter(torch.randn(1, 3, 48))

        def build_qwenvl_inputs(self, *, images, instructions, profiler):
            assert instructions == ["move"]
            assert len(images) == 1
            return {}

        def forward(self, **kwargs):
            return SimpleNamespace(hidden_states=(self.features, self.features))

    monkeypatch.setattr(pi05, "get_vlm_model", lambda config: SyntheticVLM())
    config = OmegaConf.create({
        "framework": {"action_model": {
            "state_dim": state_dim, "state_encoding": "continuous_projector",
            "action_dim": action_dim, "action_env_dim": action_dim, "action_horizon": 1,
            "num_target_vision_tokens": 2, "max_seq_len": 4,
            "diffusion_model_cfg": {"action_dit_hidden_dim": 32, "attention_head_dim": 16,
                                    "dropout": 0.0, "final_dropout": False},
        }},
        "trainer": {}, "datasets": {"vla_data": {"include_state": state_dim != 0}},
    })
    torch.manual_seed(7)
    model = pi05.Qwen_PI_v3(config).eval()
    example = {"image": [Image.new("RGB", (224, 224))], "lang": "move",
               "action": np.linspace(-0.5, 0.5, action_dim, dtype=np.float32)[None]}
    seen = []
    if state_dim:
        example["state"] = np.linspace(-0.8, 0.8, state_dim, dtype=np.float32)[None]
        encoder = model.action_model.state_encoder
        assert encoder.layer1.in_features == state_dim
        assert encoder.layer1.out_features == encoder.layer2.in_features == 1024
        assert encoder.layer2.out_features == 32  # Reduced DiT width only in this CPU fixture.
        encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    else:
        assert model.action_model.state_encoder is None
        assert not any("state_encoder" in name for name, _ in model.named_parameters())

    torch.manual_seed(19)
    loss = model([example])["action_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.action_model.action_decoder.layer2.weight.grad.abs().sum() > 0
    if state_dim:
        torch.testing.assert_close(seen[0], torch.from_numpy(example["state"])[None].repeat(2, 1, 1))
        for parameter in encoder.parameters():
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0

    torch.manual_seed(23)
    prediction = model.predict_action([example])["normalized_actions"]
    assert prediction.shape == (1, 1, action_dim)
    assert np.isfinite(prediction).all()
    torch.manual_seed(23)
    np.testing.assert_array_equal(prediction, model.predict_action([example])["normalized_actions"])
    if state_dim:
        torch.testing.assert_close(seen[-1], torch.from_numpy(example["state"])[None])
        changed = {**example, "state": example["state"] + 0.5}
        torch.manual_seed(23)
        perturbed = model.predict_action([changed])["normalized_actions"]
        assert not np.allclose(prediction, perturbed, atol=1e-6, rtol=1e-6)
        torch.manual_seed(23)
        np.testing.assert_array_equal(perturbed, model.predict_action([changed])["normalized_actions"])

    checkpoint = tmp_path / "small_pi05.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = pi05.Qwen_PI_v3(config).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.manual_seed(23)
    np.testing.assert_array_equal(prediction, restored.predict_action([example])["normalized_actions"])


def test_noninterleaved_head_keeps_cross_attention_on_every_layer():
    config = OmegaConf.create({"framework": {"action_model": pi05.QwenPI_v3DefaultConfig().action_model}})
    action = config.framework.action_model
    action.action_dim = action.action_env_dim = 3
    action.action_horizon = 1
    action.state_dim = 0
    action.diffusion_model_cfg = {
        "num_layers": 2, "input_embedding_dim": 32, "cross_attention_dim": 32,
        "attention_head_dim": 16, "num_attention_heads": 2,
        "interleave_self_attention": False, "dropout": 0.0, "final_dropout": False,
    }
    head = pi05.LayerwiseFlowmatchingActionHead(config).eval()
    features = [torch.randn(1, 3, 32), torch.randn(1, 3, 32)]
    seen = []
    for block in head.model.transformer_blocks:
        block.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs["encoder_hidden_states"]), with_kwargs=True
        )
    loss = head(features, torch.zeros(1, 1, 3))
    loss.backward()
    prediction = head.predict_action(features)
    assert torch.isfinite(loss) and torch.isfinite(prediction).all()
    assert len(seen) == 2 * (1 + action.num_inference_timesteps)
    for index, hidden in enumerate(seen):
        assert hidden is features[index % 2]
