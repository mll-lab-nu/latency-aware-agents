from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace


STARVLA_ROOT = Path(__file__).resolve().parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))


class FakeBatchInputs(dict):
    def __init__(self):
        super().__init__({"input_ids": FakeTensor()})
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


class FakeTensor:
    def clone(self):
        return self


class FakeProcessor:
    def __init__(self):
        self.messages = None
        self.kwargs = None
        self.call_count = 0
        self.batch_inputs = FakeBatchInputs()

    def apply_chat_template(self, messages, **kwargs):
        self.call_count += 1
        self.messages = messages
        self.kwargs = kwargs
        return self.batch_inputs

    def __call__(self, **kwargs):  # pragma: no cover - inference must not hit this
        raise AssertionError("inference must build inputs via apply_chat_template, not __call__")


class FakeProfiler:
    def __init__(self):
        self.timings = {}

    def time(self, stage):
        @contextmanager
        def timer():
            yield
            self.timings[stage] = 1.0

        return timer()


def _load_qwen3_module():
    transformers_module = ModuleType("transformers")
    transformers_module.AutoProcessor = object
    transformers_module.Qwen3VLForConditionalGeneration = object
    modeling_outputs_module = ModuleType("transformers.modeling_outputs")
    modeling_outputs_module.CausalLMOutputWithPast = object
    trainer_utils_module = ModuleType("starVLA.training.trainer_utils")
    trainer_utils_module.initialize_overwatch = lambda name: SimpleNamespace()
    sys.modules["transformers"] = transformers_module
    sys.modules["transformers.modeling_outputs"] = modeling_outputs_module
    sys.modules["starVLA.training.trainer_utils"] = trainer_utils_module

    module_path = STARVLA_ROOT / "starVLA/model/modules/vlm/QWen3.py"
    spec = importlib.util.spec_from_file_location("qwen3_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen3_build_inputs_records_substages_via_unified_apply_chat_template():
    module = _load_qwen3_module()
    interface = module._QWen3_VL_Interface.__new__(module._QWen3_VL_Interface)
    processor = FakeProcessor()
    profiler = FakeProfiler()
    interface.processor = processor
    interface.model = SimpleNamespace(device="cuda:0")
    interface.config = SimpleNamespace(
        datasets=SimpleNamespace(vla_data={}),
        trainer=SimpleNamespace(profile_timing=SimpleNamespace(enabled=False)),
    )

    batch_inputs = interface.build_qwenvl_inputs(
        images=[["image-0"]],
        instructions=["shoot the enemy"],
        profiler=profiler,
    )

    # Inference builds inputs through the single unified tokenizing call, byte-for-byte
    # the same as without a profiler — timing only wraps it, never reshapes it.
    assert batch_inputs is processor.batch_inputs
    assert processor.call_count == 1
    assert processor.kwargs == {
        "tokenize": True,
        "padding": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert processor.messages == [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "image-0"},
                    {"type": "text", "text": "shoot the enemy"},
                ],
            }
        ]
    ]
    assert processor.batch_inputs.to_calls == ["cuda:0"]
    assert profiler.timings == {
        "starvla_qwen_message_build_ms": 1.0,
        "starvla_qwen_processor_ms": 1.0,
        "starvla_qwen_h2d_ms": 1.0,
    }
