import pytest


def _action_config():
    return {
        "diffusion_model_cfg": {
            "num_layers": 1,
            "input_embedding_dim": 32,
            "cross_attention_dim": 32,
            "num_attention_heads": 1,
            "attention_head_dim": 32,
            "output_dim": 32,
        },
        "action_hidden_dim": 32,
        "hidden_size": 32,
        "action_dim": 3,
        "action_env_dim": 3,
        "action_horizon": 4,
        "state_dim": 2,
        "num_inference_timesteps": 2,
        "num_target_vision_tokens": 1,
        "add_pos_embed": False,
        "max_seq_len": 16,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
    }


def test_training_keeps_prefix_clean_and_masks_its_loss():
    torch = pytest.importorskip("torch")
    from starVLA.model.modules.action_model.flow_matching_head.prefix_conditioning import (
        flow_training_inputs,
        masked_action_loss,
    )

    actions = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    noise = -torch.ones_like(actions)
    time = torch.tensor([0.2, 0.7])
    prefix_mask = torch.tensor([
        [True, True, False, False],
        [True, False, False, False],
    ])
    trajectory, token_time = flow_training_inputs(actions, noise, time, prefix_mask)

    torch.testing.assert_close(trajectory[prefix_mask], actions[prefix_mask])
    torch.testing.assert_close(
        token_time[prefix_mask], torch.ones_like(token_time[prefix_mask])
    )
    torch.testing.assert_close(
        trajectory[~prefix_mask],
        (token_time[..., None] * actions + (1 - token_time[..., None]) * noise)[
            ~prefix_mask
        ],
    )

    prediction = torch.randn(2, 4, 3, requires_grad=True)
    target = torch.randn_like(prediction)
    loss_mask = ~prefix_mask
    masked_action_loss(prediction, target, loss_mask).backward()
    assert torch.count_nonzero(prediction.grad[prefix_mask]).item() == 0
    assert torch.count_nonzero(prediction.grad[loss_mask]).item() > 0


def test_training_without_prefix_matches_standard_flow_inputs():
    torch = pytest.importorskip("torch")
    from starVLA.model.modules.action_model.flow_matching_head.prefix_conditioning import (
        flow_training_inputs,
    )

    actions = torch.randn(2, 4, 3)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.2, 0.7])
    trajectory, token_time = flow_training_inputs(actions, noise, time, None)
    expected_time = time[:, None].expand(2, 4)

    torch.testing.assert_close(token_time, expected_time)
    torch.testing.assert_close(
        trajectory,
        expected_time[..., None] * actions + (1 - expected_time[..., None]) * noise,
    )


def test_qwenpi_and_qwengroot_preserve_exact_prefixes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from starVLA.model.modules.action_model import GR00T_ActionHeader as groot_module
    from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
        LayerwiseFlowmatchingActionHead,
    )

    common = _action_config()
    qwenpi = LayerwiseFlowmatchingActionHead(
        OmegaConf.create({"framework": {"action_model": common}})
    ).eval()
    groot_module.DiTConfig["DiT-Test"] = {
        "input_embedding_dim": 32,
        "attention_head_dim": 32,
        "num_attention_heads": 1,
    }
    groot_config = dict(common)
    groot_config["action_model_type"] = "DiT-Test"
    qwengroot = FlowmatchingActionHead(
        OmegaConf.create({"framework": {"action_model": groot_config}})
    ).eval()

    committed = torch.randn(2, 4, 3)
    prefix_mask = torch.tensor([
        [True, True, False, False],
        [True, False, False, False],
    ])
    state = torch.randn(2, 1, 2)
    qwenpi_actions = qwenpi.predict_action(
        [torch.randn(2, 3, 32)],
        state,
        action_prefix=committed,
        action_prefix_mask=prefix_mask,
    )
    qwengroot_actions = qwengroot.predict_action(
        torch.randn(2, 3, 32),
        state,
        action_prefix=committed,
        action_prefix_mask=prefix_mask,
    )

    assert torch.equal(qwenpi_actions[prefix_mask], committed[prefix_mask])
    assert torch.equal(qwengroot_actions[prefix_mask], committed[prefix_mask])


def test_qwenpi_loss_mask_ignores_missing_action_targets():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
        LayerwiseFlowmatchingActionHead,
    )

    head = LayerwiseFlowmatchingActionHead(
        OmegaConf.create({"framework": {"action_model": _action_config()}})
    ).eval()
    actions = torch.randn(2, 4, 3)
    state = torch.randn(2, 1, 2)
    hidden = [torch.randn(2, 3, 32)]
    loss_mask = torch.tensor([
        [True, False, False, False],
        [False, True, True, True],
    ])

    torch.manual_seed(77)
    loss = head(hidden, actions, state, action_loss_mask=loss_mask)
    changed = actions.clone()
    changed[~loss_mask] = 10000
    torch.manual_seed(77)
    torch.testing.assert_close(
        head(hidden, changed, state, action_loss_mask=loss_mask), loss
    )
