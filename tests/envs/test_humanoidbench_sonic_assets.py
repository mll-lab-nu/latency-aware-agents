from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from latency_bench.envs import humanoidbench_sonic_assets as assets


def test_prepare_g1_sonic_assets_returns_explicit_pinned_paths(tmp_path, monkeypatch):
    snapshot = tmp_path / "assets"
    (snapshot / "sonic_g1").mkdir(parents=True)
    decoder = tmp_path / "decoder" / "model_decoder.onnx"
    decoder.parent.mkdir()
    graph = helper.make_graph(
        [],
        "decoder",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 994])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 29])],
    )
    onnx.save(helper.make_model(graph), decoder)
    monkeypatch.setattr(assets, "snapshot_download", lambda **_kwargs: snapshot)
    monkeypatch.setattr(assets, "hf_hub_download", lambda **_kwargs: decoder)

    asset_root, dynamic_decoder = assets.prepare_g1_sonic_assets(tmp_path / "cache")

    assert asset_root == snapshot / "sonic_g1"
    assert dynamic_decoder == decoder.with_name("model_decoder_dynamic.onnx")
    model = onnx.load(dynamic_decoder)
    assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert model.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "batch"
