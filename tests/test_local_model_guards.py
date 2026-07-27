"""Pre-flight guards for the trainable local model (spec §M8.3a–c).

Three failure modes this closes, all of which used to surface as something
other than a configuration error:

* a model too large for the machine died inside ``from_pretrained``;
* ``LOCAL_MODEL_QUANTIZE`` without CUDA died inside bitsandbytes;
* ``LORA_TARGET_MODULES`` that matched no module trained zero parameters to
  completion and reported success.
"""
import sys
import types

import pytest

import aegis.config as cfg
from aegis.layers.weight_modifier import WeightModifier


@pytest.fixture
def wm():
    return WeightModifier()


class _FakeModel:
    """Minimal stand-in exposing the one API the guard walks."""

    def __init__(self, names):
        self._names = names

    def named_modules(self):
        for name in self._names:
            yield name, object()


# ── parameter-count inference ────────────────────────────────────────

def test_parameter_count_read_from_a_size_suffix():
    assert WeightModifier._declared_parameter_count("Qwen/Qwen2.5-3B-Instruct") == 3_000_000_000


def test_parameter_count_handles_fractional_sizes():
    assert WeightModifier._declared_parameter_count("Qwen/Qwen3-1.7B") == 1_700_000_000


def test_parameter_count_handles_million_scale():
    assert WeightModifier._declared_parameter_count("tiny/model-350M") == 350_000_000


def test_parameter_count_is_none_when_unknowable():
    assert WeightModifier._declared_parameter_count("some/unlabelled-model") is None


def test_parameter_count_prefers_a_local_config(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 100}',
        encoding="utf-8")
    # 12·64²·2 + 2·100·64 = 98304 + 12800
    assert WeightModifier._declared_parameter_count(str(tmp_path)) == 111_104


def test_parameter_count_falls_back_when_the_config_is_corrupt(tmp_path):
    model_dir = tmp_path / "Qwen2.5-3B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{ broken", encoding="utf-8")
    assert WeightModifier._declared_parameter_count(str(model_dir)) == 3_000_000_000


# ── memory check ─────────────────────────────────────────────────────

def test_memory_check_refuses_a_model_that_cannot_fit(monkeypatch):
    monkeypatch.setattr(WeightModifier, "available_memory_bytes",
                        staticmethod(lambda: 2 * 2 ** 30))
    verdict = WeightModifier.memory_check("Qwen/Qwen2.5-7B", for_training=True)
    assert verdict["ok"] is False


def test_refusal_names_the_model_the_need_and_the_remedy(monkeypatch):
    monkeypatch.setattr(WeightModifier, "available_memory_bytes",
                        staticmethod(lambda: 2 * 2 ** 30))
    reason = WeightModifier.memory_check("Qwen/Qwen2.5-7B", for_training=True)["reason"]
    assert "Qwen2.5-7B" in reason           # which model
    assert "GiB" in reason                  # how much is needed vs available
    assert "3B" in reason                   # what to do instead


def test_memory_check_accepts_a_model_that_fits(monkeypatch):
    monkeypatch.setattr(WeightModifier, "available_memory_bytes",
                        staticmethod(lambda: 64 * 2 ** 30))
    assert WeightModifier.memory_check("Llama-3.2-1B", for_training=True)["ok"] is True


def test_training_needs_more_headroom_than_inference(monkeypatch):
    monkeypatch.setattr(WeightModifier, "available_memory_bytes",
                        staticmethod(lambda: 64 * 2 ** 30))
    infer = WeightModifier.memory_check("Qwen2.5-3B", for_training=False)
    train = WeightModifier.memory_check("Qwen2.5-3B", for_training=True)
    assert train["required_bytes"] > infer["required_bytes"]


def test_unknown_model_size_is_permitted_not_refused(monkeypatch):
    # "I could not measure" must not become "I refuse" — that would block every
    # model whose id carries no size suffix.
    monkeypatch.setattr(WeightModifier, "available_memory_bytes",
                        staticmethod(lambda: 1))
    verdict = WeightModifier.memory_check("some/unlabelled-model")
    assert verdict["ok"] is True
    assert verdict["reason"] == "unknown_size"


def test_unknown_available_memory_is_permitted(monkeypatch):
    monkeypatch.setattr(WeightModifier, "available_memory_bytes",
                        staticmethod(lambda: None))
    verdict = WeightModifier.memory_check("Qwen2.5-3B")
    assert verdict["ok"] is True
    assert verdict["reason"] == "available_memory_unknown"


def test_bytes_per_param_matches_the_dtype():
    assert WeightModifier._bytes_per_param("bfloat16") == 2
    assert WeightModifier._bytes_per_param("float16") == 2
    assert WeightModifier._bytes_per_param("float32") == 4


def test_unknown_dtype_is_costed_conservatively():
    # Guessing low would let an oversized model through the guard.
    assert WeightModifier._bytes_per_param("mystery") == 4


def test_available_memory_returns_a_positive_number_or_none():
    value = WeightModifier.available_memory_bytes()
    assert value is None or value > 0


def test_load_model_refuses_instead_of_dying_on_a_huge_model(wm, monkeypatch):
    monkeypatch.setattr(WeightModifier, "available_memory_bytes",
                        staticmethod(lambda: 2 * 2 ** 30))
    monkeypatch.setattr("aegis.layers.weight_modifier.LOCAL_MODEL_PATH",
                        "Qwen/Qwen2.5-72B")
    result = wm.load_model()
    assert result["success"] is False
    assert "72B" in result["error"]


# ── quantization without CUDA ────────────────────────────────────────

def test_no_quantization_requested_is_no_error():
    assert cfg.quantization_error("") is None


def test_unrecognised_quantization_is_rejected():
    message = cfg.quantization_error("3bit")
    assert message and "3bit" in message


def test_quantization_without_cuda_is_refused(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    message = cfg.quantization_error("4bit")
    assert message is not None
    assert "CUDA" in message
    assert "bfloat16" in message            # tells the operator what to do


def test_quantization_with_cuda_is_allowed(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert cfg.quantization_error("8bit") is None


def test_quantization_without_torch_is_refused(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    message = cfg.quantization_error("4bit")
    assert message and "torch" in message


def test_load_model_refuses_when_quantization_is_impossible(wm, monkeypatch):
    monkeypatch.setattr("aegis.layers.weight_modifier.quantization_error",
                        lambda: "no CUDA here")
    assert wm.load_model() == {"success": False, "error": "no CUDA here"}


# ── LoRA target modules ──────────────────────────────────────────────

def test_target_modules_come_from_the_environment_not_a_constant():
    # The whole point of §M8.3c: swapping model family must not need a code edit.
    assert isinstance(cfg.LORA_TARGET_MODULES, list)
    assert cfg.LORA_TARGET_MODULES  # never empty, even with a blank env var


def test_matching_targets_are_those_present_in_the_model():
    model = _FakeModel(["layers.0.self_attn.q_proj", "layers.0.self_attn.v_proj"])
    assert WeightModifier.matching_lora_targets(
        model, ["q_proj", "k_proj", "v_proj"]) == ["q_proj", "v_proj"]


def test_no_match_is_reported_as_empty():
    model = _FakeModel(["layers.0.self_attn.qkv_proj"])   # Phi-3 style
    assert WeightModifier.matching_lora_targets(model, ["q_proj", "v_proj"]) == []


def test_unintrospectable_model_keeps_the_configured_targets():
    # "Could not look" must not be read as "looked and found nothing".
    assert WeightModifier.matching_lora_targets(object(), ["q_proj"]) == ["q_proj"]


def test_module_leaf_names_are_deduplicated_and_sorted():
    model = _FakeModel(["a.q_proj", "b.q_proj", "a.o_proj"])
    assert WeightModifier.module_leaf_names(model) == ["o_proj", "q_proj"]


def test_module_leaf_names_is_none_when_unwalkable():
    assert WeightModifier.module_leaf_names(object()) is None


def test_prepare_lora_refuses_a_mismatched_target_list(wm, monkeypatch):
    mod = types.ModuleType("peft")
    mod.LoraConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    mod.get_peft_model = lambda model, cfg_: None
    mod.TaskType = types.SimpleNamespace(CAUSAL_LM="causal")
    monkeypatch.setitem(sys.modules, "peft", mod)
    monkeypatch.setattr("aegis.layers.weight_modifier.LORA_TARGET_MODULES",
                        ["q_proj", "v_proj"])

    wm.model = _FakeModel(["layers.0.self_attn.qkv_proj"])
    with pytest.raises(ValueError) as excinfo:
        wm._prepare_lora()
    message = str(excinfo.value)
    assert "qkv_proj" in message            # what the model actually has
    assert "zero parameters" in message     # why refusing beats proceeding


def test_prepare_lora_refuses_when_zero_parameters_end_up_trainable(wm, monkeypatch):
    class _Param:
        def __init__(self, n, grad):
            self._n, self.requires_grad = n, grad

        def numel(self):
            return self._n

    peft_model = types.SimpleNamespace(
        parameters=lambda: iter([_Param(100, False)]))
    mod = types.ModuleType("peft")
    mod.LoraConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    mod.get_peft_model = lambda model, cfg_: peft_model
    mod.TaskType = types.SimpleNamespace(CAUSAL_LM="causal")
    monkeypatch.setitem(sys.modules, "peft", mod)

    wm.model = _FakeModel(["layers.0.self_attn.q_proj"])
    with pytest.raises(ValueError):
        wm._prepare_lora()


def test_prepare_lora_reports_the_modules_it_attached_to(wm, monkeypatch):
    class _Param:
        def __init__(self, n, grad):
            self._n, self.requires_grad = n, grad

        def numel(self):
            return self._n

    peft_model = types.SimpleNamespace(
        parameters=lambda: iter([_Param(100, True), _Param(900, False)]))
    mod = types.ModuleType("peft")
    mod.LoraConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    mod.get_peft_model = lambda model, cfg_: peft_model
    mod.TaskType = types.SimpleNamespace(CAUSAL_LM="causal")
    monkeypatch.setitem(sys.modules, "peft", mod)
    monkeypatch.setattr("aegis.layers.weight_modifier.LORA_TARGET_MODULES",
                        ["q_proj", "v_proj"])

    wm.model = _FakeModel(["layers.0.self_attn.q_proj"])
    info = wm._prepare_lora()
    assert info["trainable_params"] == 100
    assert info["target_modules"] == ["q_proj"]


# ── configuration defaults for the reference profile (§M8.3a) ────────

def test_default_dtype_is_bfloat16_not_float32():
    # float32 doubles the memory a model needs on a CPU-only box and buys
    # nothing — this default was the single biggest waste in the profile.
    assert cfg.LOCAL_MODEL_DTYPE == "bfloat16"


def test_default_context_length_matches_a_modern_model():
    assert cfg.LOCAL_MODEL_MAX_LENGTH >= 4096
