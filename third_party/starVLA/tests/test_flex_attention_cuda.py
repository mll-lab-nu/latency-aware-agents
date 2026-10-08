import pytest


torch = pytest.importorskip("torch")
TARGET_STACK = torch.__version__ == "2.11.0+cu128" and torch.version.cuda == "12.8"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TARGET_STACK,
    reason="requires the fixed PyTorch 2.11.0+cu128 CUDA 12.8 stack",
)
def test_fixed_torch_211_flex_attention_forward_backward_is_finite():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    from starVLA.model.modules.vlm.QWen3 import _FLEX_KERNEL_OPTIONS

    query = torch.randn(
        1,
        2,
        512,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn(
        1,
        2,
        512,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    value = torch.randn(
        1,
        2,
        512,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    def causal_mask(_batch, _head, query_index, key_value_index):
        return query_index >= key_value_index

    block_mask = create_block_mask(
        causal_mask,
        B=1,
        H=2,
        Q_LEN=512,
        KV_LEN=512,
        device="cuda",
    )
    output = flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        kernel_options=_FLEX_KERNEL_OPTIONS,
    )
    output.float().square().mean().backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(key.grad).all()
    assert torch.isfinite(value.grad).all()
