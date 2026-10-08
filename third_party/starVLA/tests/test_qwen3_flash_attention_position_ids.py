from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("enabled", [True, False])
def test_gradient_checkpointing_follows_config_during_backward(monkeypatch, enabled):
    """Qwen backend regression: configured checkpointing recomputes during backward."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    omega = pytest.importorskip("omegaconf")
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface

    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
        text_config={
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 2, "head_dim": 8,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 1, "hidden_size": 16, "intermediate_size": 32,
            "num_heads": 2, "out_hidden_size": 16,
        },
    )).eval()
    processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    qwenvl = {"base_vlm": "Qwen/Qwen3-VL-test", "enable_gradient_checkpointing": enabled}
    interface = _QWen3_VL_Interface(omega.OmegaConf.create({"framework": {"qwenvl": qwenvl}}))
    assert interface.model.is_gradient_checkpointing is enabled

    interface.train()
    forwards = []
    model.model.language_model.layers[0].register_forward_pre_hook(lambda *args: forwards.append(1))
    model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits.sum().backward()
    assert len(forwards) == (2 if enabled else 1)


def test_flash_attention_patch_removes_mrope_position_ids_only_for_fa2():
    pytest.importorskip("transformers")

    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

    from starVLA.model.modules.vlm.QWen3 import _patch_qwen3vl_flash_attention_position_ids

    calls = []
    cls = qwen3_vl.Qwen3VLTextAttention
    original_forward = cls.forward
    try:
        original_patch_marker = cls._starvla_flash_attention_position_ids_patched
        had_patch_marker = True
    except AttributeError:
        had_patch_marker = False

    def fake_forward(self, *args, **kwargs):
        calls.append(kwargs)
        return "ok"

    cls.forward = fake_forward
    if had_patch_marker:
        del cls._starvla_flash_attention_position_ids_patched

    try:
        _patch_qwen3vl_flash_attention_position_ids()

        fa2_attention = SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="flash_attention_2")
        )
        cls.forward(fa2_attention, position_ids="mrope-position-ids", other="kept")
        assert "position_ids" not in calls[-1]
        assert calls[-1]["other"] == "kept"

        sdpa_attention = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
        cls.forward(sdpa_attention, position_ids="mrope-position-ids", other="kept")
        assert calls[-1]["position_ids"] == "mrope-position-ids"
        assert calls[-1]["other"] == "kept"
    finally:
        cls.forward = original_forward
        if had_patch_marker:
            cls._starvla_flash_attention_position_ids_patched = original_patch_marker
        elif hasattr(cls, "_starvla_flash_attention_position_ids_patched"):
            del cls._starvla_flash_attention_position_ids_patched


def test_flex_attention_patch_uses_fixed_triton_stages():
    pytest.importorskip("transformers")

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from starVLA.model.modules.vlm.QWen3 import _patch_qwen3vl_flex_attention_support

    calls = []

    def fake_flex(*args, **kwargs):
        calls.append(kwargs)
        return "ok"

    original_flex = ALL_ATTENTION_FUNCTIONS["flex_attention"]
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = fake_flex
    try:
        _patch_qwen3vl_flex_attention_support()
        ALL_ATTENTION_FUNCTIONS["flex_attention"]()
        assert calls[-1]["kernel_options"] == {
            "BACKEND": "TRITON",
            "fwd_num_stages": 2,
            "bwd_num_stages": 1,
        }
    finally:
        ALL_ATTENTION_FUNCTIONS["flex_attention"] = original_flex


def test_position_cache_tracks_layout_images_weights_and_training(monkeypatch):
    """Long-lived Qwen backend contract: cached geometry stays exact across inference/training."""
    import copy
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface

    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
        image_token_id=4, video_token_id=5, vision_start_token_id=6,
        text_config={
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 8, "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
            "out_hidden_size": 16, "patch_size": 2, "temporal_patch_size": 2,
            "num_position_embeddings": 16, "deepstack_visual_indexes": [],
        },
    )).eval()
    baseline = copy.deepcopy(model)
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "from_pretrained", lambda *a, **kw: model)
    monkeypatch.setattr(AutoProcessor, "from_pretrained",
                        lambda *a, **kw: SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right")))
    interface = _QWen3_VL_Interface(OmegaConf.create({
        "framework": {"qwenvl": {"base_vlm": "test", "enable_gradient_checkpointing": False}},
    })).eval()
    inputs = {
        "input_ids": torch.tensor([[1, 6, 4, 4, 4, 4, 7]]),
        "attention_mask": torch.ones(1, 7, dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "pixel_values": torch.randn(16, 24),
    }
    with torch.inference_mode():
        first = model(**inputs, use_cache=False).logits
        for batch, grid, token, masked in [
            (1, [1, 4, 4], 1, False), (1, [1, 2, 8], 2, False), (2, [1, 4, 4], 2, True),
        ]:
            varied = {key: value.repeat(batch, 1) for key, value in inputs.items()}
            varied["input_ids"][:, 0] = token
            varied["attention_mask"][:, 0] = 0 if masked else 1
            varied["image_grid_thw"][:] = torch.tensor(grid)
            varied["pixel_values"] = torch.randn_like(varied["pixel_values"])
            actual = model(**varied, use_cache=False).logits
            torch.testing.assert_close(actual, baseline(**varied, use_cache=False).logits, rtol=0, atol=0)
        inputs["pixel_values"] = torch.randn_like(inputs["pixel_values"])
        changed = model(**inputs, use_cache=False).logits
        assert not torch.equal(first, changed)
        model.model.visual.pos_embed.weight.add_(0.2)
        baseline.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(**inputs, use_cache=False).logits,
                                   baseline(**inputs, use_cache=False).logits, rtol=0, atol=0)
        interface.load_state_dict(interface.state_dict())
    inputs = {key: value.clone() for key, value in inputs.items()}
    interface.train()
    model(**inputs, use_cache=False).logits.sum().backward()
    assert model.model.visual.pos_embed.weight.grad is not None


if __name__ == "__main__":
    test_flash_attention_patch_removes_mrope_position_ids_only_for_fa2()
    test_flex_attention_patch_uses_fixed_triton_stages()
