"""Unit tests for aegis/layers/weight_modifier.py.

No real model is ever loaded and no real training runs. torch/transformers/peft/
datasets are stubbed via sys.modules injection, and every heavy path is either
mocked or driven with fakes. All file writes are redirected into tmp_path so the
real data/ checkpoint dir is never touched.
"""
import sys
import json
import time
import types
import asyncio

import pytest

import aegis.layers.weight_modifier as wm_mod
from aegis.layers.weight_modifier import WeightModifier


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def ckpt_dir(tmp_path, monkeypatch):
    """Redirect the checkpoint dir (and hence training_stats.json) into tmp."""
    d = tmp_path / "weight_checkpoints"
    d.mkdir()
    monkeypatch.setattr(wm_mod, "WEIGHT_CHECKPOINTS_DIR", d)
    return d


@pytest.fixture
def wm(ckpt_dir):
    """A fresh modifier whose stats file lives under tmp."""
    return WeightModifier()


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class FakeParam:
    def __init__(self, n, requires_grad):
        self._n = n
        self.requires_grad = requires_grad

    def numel(self):
        return self._n


class FakeDataset:
    def __init__(self, n, cols=None):
        self._n = n
        self.column_names = cols if cols is not None else ["instruction", "output"]

    def __len__(self):
        return self._n

    def map(self, fn, remove_columns=None):
        return self


class FakeTrainResult:
    def __init__(self, loss):
        self.training_loss = loss


def _fake_transformers(train_loss=0.5, eval_loss=0.2, has_eval_strategy=True):
    """Build a stand-in `transformers` module for _train_sync."""
    mod = types.ModuleType("transformers")

    class TrainingArguments:
        # signature drives the eval_strategy vs evaluation_strategy branch
        if has_eval_strategy:
            def __init__(self, output_dir=None, eval_strategy=None, **kwargs):
                self.output_dir = output_dir
        else:
            def __init__(self, output_dir=None, evaluation_strategy=None, **kwargs):
                self.output_dir = output_dir

    class TrainerCallback:
        pass

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self):
            return FakeTrainResult(train_loss)

        def evaluate(self):
            return {"eval_loss": eval_loss}

    mod.TrainingArguments = TrainingArguments
    mod.TrainerCallback = TrainerCallback
    mod.Trainer = Trainer
    return mod


# --------------------------------------------------------------------------- #
# _load_stats / _save_stats                                                   #
# --------------------------------------------------------------------------- #
def test_load_stats_missing_file(wm):
    assert wm.total_trainings == 0
    assert wm.current_checkpoint is None


def test_load_stats_from_file(ckpt_dir):
    stats = ckpt_dir / "training_stats.json"
    stats.write_text(json.dumps({
        "total_trainings": 3,
        "total_rollbacks": 1,
        "last_train_time": 123.0,
        "current_checkpoint": "/x/lora_1",
        "baseline_val_loss": 0.4,
        "train_history": [{"i": i} for i in range(30)],
    }), encoding="utf-8")
    m = WeightModifier()
    assert m.total_trainings == 3
    assert m.total_rollbacks == 1
    assert m.current_checkpoint == "/x/lora_1"
    assert m._baseline_val_loss == 0.4
    assert len(m.train_history) == 20  # truncated to last 20


def test_load_stats_corrupt(ckpt_dir):
    (ckpt_dir / "training_stats.json").write_text("not json", encoding="utf-8")
    m = WeightModifier()  # must not raise
    assert m.total_trainings == 0


def test_save_stats_roundtrip(wm):
    wm.total_trainings = 5
    wm.current_checkpoint = "/y/lora_9"
    wm._save_stats()
    data = json.loads(wm._stats_path.read_text(encoding="utf-8"))
    assert data["total_trainings"] == 5
    assert data["current_checkpoint"] == "/y/lora_9"


def test_save_stats_swallows_errors(wm):
    class Boom:
        def write_text(self, *a, **k):
            raise OSError("nope")

    wm._stats_path = Boom()
    wm._save_stats()  # must not raise


# --------------------------------------------------------------------------- #
# can_train                                                                   #
# --------------------------------------------------------------------------- #
def test_can_train_in_progress(wm):
    wm.training_in_progress = True
    ok, reason = wm.can_train()
    assert ok is False and "in progress" in reason


def test_can_train_cooldown(wm, monkeypatch):
    monkeypatch.setattr(wm_mod, "TRAIN_MIN_INTERVAL_SECONDS", 3600)
    wm.last_train_time = time.time()
    ok, reason = wm.can_train()
    assert ok is False and "Cooldown" in reason


def test_can_train_ready_first_time(wm):
    wm.last_train_time = 0
    assert wm.can_train() == (True, "Ready")


def test_can_train_ready_after_cooldown(wm, monkeypatch):
    monkeypatch.setattr(wm_mod, "TRAIN_MIN_INTERVAL_SECONDS", 1)
    wm.last_train_time = time.time() - 10
    ok, _ = wm.can_train()
    assert ok is True


# --------------------------------------------------------------------------- #
# load_model                                                                  #
# --------------------------------------------------------------------------- #
def test_load_model_already_loaded(wm):
    wm.model_loaded = True
    res = wm.load_model()
    assert res["success"] is True and "already" in res["message"]


def test_load_model_import_error(wm, monkeypatch):
    # An empty transformers module makes `from transformers import ...` raise.
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    res = wm.load_model()
    assert res["success"] is False
    assert "Missing dependency" in res["error"]


def _install_fake_transformers_for_load(monkeypatch, tokenizer, model, raise_on_model=False):
    mod = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return tokenizer

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(**kwargs):
            if raise_on_model:
                raise RuntimeError("model boom")
            AutoModelForCausalLM.kwargs = kwargs
            return model

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mod.AutoTokenizer = AutoTokenizer
    mod.AutoModelForCausalLM = AutoModelForCausalLM
    mod.BitsAndBytesConfig = BitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return mod


def test_load_model_success_no_quant(wm, monkeypatch):
    tok = types.SimpleNamespace(pad_token=None, eos_token="<eos>")
    model = object()
    monkeypatch.setattr(wm_mod, "LOCAL_MODEL_QUANTIZE", "")
    _install_fake_transformers_for_load(monkeypatch, tok, model)
    res = wm.load_model()
    assert res["success"] is True
    assert wm.model_loaded is True
    assert wm.model is model
    assert tok.pad_token == "<eos>"  # pad_token was None -> set to eos


def test_load_model_4bit_quant(wm, monkeypatch):
    tok = types.SimpleNamespace(pad_token="<p>", eos_token="<e>")
    monkeypatch.setattr(wm_mod, "LOCAL_MODEL_QUANTIZE", "4bit")
    mod = _install_fake_transformers_for_load(monkeypatch, tok, object())
    res = wm.load_model()
    assert res["success"] is True
    assert "quantization_config" in mod.AutoModelForCausalLM.kwargs


def test_load_model_8bit_quant(wm, monkeypatch):
    tok = types.SimpleNamespace(pad_token="<p>", eos_token="<e>")
    monkeypatch.setattr(wm_mod, "LOCAL_MODEL_QUANTIZE", "8bit")
    mod = _install_fake_transformers_for_load(monkeypatch, tok, object())
    assert wm.load_model()["success"] is True


def test_load_model_applies_existing_checkpoint(wm, monkeypatch, tmp_path):
    cp = tmp_path / "lora_existing"
    cp.mkdir()
    wm.current_checkpoint = str(cp)
    tok = types.SimpleNamespace(pad_token="<p>", eos_token="<e>")
    monkeypatch.setattr(wm_mod, "LOCAL_MODEL_QUANTIZE", "")
    _install_fake_transformers_for_load(monkeypatch, tok, object())

    called = {}
    monkeypatch.setattr(wm, "_apply_lora_checkpoint",
                        lambda p: called.setdefault("path", p))
    wm.load_model()
    assert str(called["path"]) == str(cp)


def test_load_model_exception(wm, monkeypatch):
    tok = types.SimpleNamespace(pad_token="<p>", eos_token="<e>")
    monkeypatch.setattr(wm_mod, "LOCAL_MODEL_QUANTIZE", "")
    _install_fake_transformers_for_load(monkeypatch, tok, object(), raise_on_model=True)
    res = wm.load_model()
    assert res["success"] is False
    assert "model boom" in res["error"]


# --------------------------------------------------------------------------- #
# _apply_lora_checkpoint                                                       #
# --------------------------------------------------------------------------- #
def test_apply_lora_checkpoint_success(wm, monkeypatch):
    merged = object()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, path):
            return types.SimpleNamespace(merge_and_unload=lambda: merged)

    mod = types.ModuleType("peft")
    mod.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", mod)
    wm.model = object()
    wm._apply_lora_checkpoint(wm_mod.WEIGHT_CHECKPOINTS_DIR / "lora_x")
    assert wm.model is merged


def test_apply_lora_checkpoint_failure(wm, monkeypatch):
    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, path):
            raise RuntimeError("peft boom")

    mod = types.ModuleType("peft")
    mod.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", mod)
    before = wm.model
    wm._apply_lora_checkpoint(wm_mod.WEIGHT_CHECKPOINTS_DIR / "lora_x")  # swallowed
    assert wm.model is before


# --------------------------------------------------------------------------- #
# _prepare_lora                                                                #
# --------------------------------------------------------------------------- #
def test_prepare_lora(wm, monkeypatch):
    params = [FakeParam(100, True), FakeParam(900, False)]
    peft_model = types.SimpleNamespace(parameters=lambda: iter(params))

    mod = types.ModuleType("peft")
    mod.LoraConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    mod.get_peft_model = lambda model, cfg: peft_model
    mod.TaskType = types.SimpleNamespace(CAUSAL_LM="causal")
    monkeypatch.setitem(sys.modules, "peft", mod)

    wm.model = object()
    info = wm._prepare_lora()
    assert info["trainable_params"] == 100
    assert info["total_params"] == 1000
    assert wm.peft_model is peft_model


# --------------------------------------------------------------------------- #
# _load_dataset                                                                #
# --------------------------------------------------------------------------- #
def _install_fake_datasets(monkeypatch, train_ds, val_ds=None):
    mod = types.ModuleType("datasets")

    def load_dataset(fmt, data_files=None):
        result = {"train": train_ds}
        if "validation" in (data_files or {}):
            result["validation"] = val_ds
        return result

    mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", mod)
    return mod


def test_load_dataset_missing_train(wm, tmp_path):
    with pytest.raises(FileNotFoundError):
        wm._load_dataset(tmp_path)


def test_load_dataset_train_only(wm, tmp_path, monkeypatch):
    (tmp_path / "train.jsonl").write_text("{}", encoding="utf-8")
    _install_fake_datasets(monkeypatch, train_ds="T")
    train, val = wm._load_dataset(tmp_path)
    assert train == "T" and val is None


def test_load_dataset_with_val(wm, tmp_path, monkeypatch):
    (tmp_path / "train.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "val.jsonl").write_text("{}", encoding="utf-8")
    _install_fake_datasets(monkeypatch, train_ds="T", val_ds="V")
    train, val = wm._load_dataset(tmp_path)
    assert train == "T" and val == "V"


# --------------------------------------------------------------------------- #
# _tokenize_sample                                                             #
# --------------------------------------------------------------------------- #
def _fake_tokenizer():
    def call(prompt, truncation=None, max_length=None, padding=None):
        call.last_prompt = prompt
        return {"input_ids": [1, 2, 3]}
    return call


def test_tokenize_sample_with_input(wm):
    tok = _fake_tokenizer()
    wm.tokenizer = tok
    out = wm._tokenize_sample({"instruction": "do", "input": "x", "output": "y"})
    assert out["labels"] == [1, 2, 3]
    assert "### Input:" in tok.last_prompt


def test_tokenize_sample_without_input(wm):
    tok = _fake_tokenizer()
    wm.tokenizer = tok
    wm._tokenize_sample({"instruction": "do", "output": "y"})
    assert "### Input:" not in tok.last_prompt


# --------------------------------------------------------------------------- #
# train() — async orchestration                                               #
# --------------------------------------------------------------------------- #
def test_train_requires_ethics(wm):
    res = asyncio.run(wm.train(wm_mod.WEIGHT_CHECKPOINTS_DIR, ethics_approved=False))
    assert res["success"] is False
    assert "Ethics approval" in res["error"]


def test_train_blocked_when_in_progress(wm):
    wm.training_in_progress = True
    res = asyncio.run(wm.train(wm_mod.WEIGHT_CHECKPOINTS_DIR, ethics_approved=True))
    assert res["success"] is False
    assert "in progress" in res["error"]


def test_train_success_calls_train_sync(wm, monkeypatch):
    wm.model_loaded = True
    monkeypatch.setattr(wm, "_train_sync", lambda d: {"success": True, "checkpoint": "cp"})
    res = asyncio.run(wm.train(wm_mod.WEIGHT_CHECKPOINTS_DIR, ethics_approved=True))
    assert res["success"] is True
    assert wm.training_in_progress is False  # reset in finally


def test_train_loads_model_when_needed(wm, monkeypatch):
    wm.model_loaded = False
    loaded = {}
    monkeypatch.setattr(wm, "load_model",
                        lambda: loaded.setdefault("called", True) or {"success": True})
    monkeypatch.setattr(wm, "_train_sync", lambda d: {"success": True})
    asyncio.run(wm.train(wm_mod.WEIGHT_CHECKPOINTS_DIR, ethics_approved=True))
    assert loaded["called"] is True


def test_train_load_model_fails(wm, monkeypatch):
    wm.model_loaded = False
    monkeypatch.setattr(wm, "load_model", lambda: {"success": False, "error": "no deps"})
    res = asyncio.run(wm.train(wm_mod.WEIGHT_CHECKPOINTS_DIR, ethics_approved=True))
    assert res["success"] is False and res["error"] == "no deps"
    assert wm.training_in_progress is False


def test_train_executor_exception(wm, monkeypatch):
    wm.model_loaded = True

    def boom(d):
        raise RuntimeError("train crash")

    monkeypatch.setattr(wm, "_train_sync", boom)
    res = asyncio.run(wm.train(wm_mod.WEIGHT_CHECKPOINTS_DIR, ethics_approved=True))
    assert res["success"] is False
    assert "train crash" in res["error"]
    assert wm.training_in_progress is False


# --------------------------------------------------------------------------- #
# _train_sync — the heavy branch logic (fully mocked, no real training)       #
# --------------------------------------------------------------------------- #
def _setup_train_sync(wm, monkeypatch, train_len=100, val_ds=None,
                      train_loss=0.5, eval_loss=0.2, has_eval_strategy=True):
    """Wire up a modifier so _train_sync can run without real deps."""
    train_ds = FakeDataset(train_len)
    monkeypatch.setattr(wm, "_load_dataset", lambda d: (train_ds, val_ds))
    monkeypatch.setattr(wm, "_prepare_lora", lambda: wm.__dict__.__setitem__(
        "peft_model", _fake_peft_model()) or {"trainable_params": 10, "total_params": 100})
    monkeypatch.setitem(sys.modules, "transformers",
                        _fake_transformers(train_loss, eval_loss, has_eval_strategy))
    wm.tokenizer = types.SimpleNamespace(save_pretrained=lambda p: None)


def _fake_peft_model():
    saved = {}
    merged = object()
    return types.SimpleNamespace(
        save_pretrained=lambda p: saved.setdefault("saved", p),
        merge_and_unload=lambda: merged,
        _merged=merged,
    )


def test_train_sync_dataset_too_small(wm, monkeypatch):
    monkeypatch.setattr(wm_mod, "TRAIN_MIN_DATASET_SIZE", 50)
    _setup_train_sync(wm, monkeypatch, train_len=10)
    res = wm._train_sync(wm_mod.WEIGHT_CHECKPOINTS_DIR)
    assert res["success"] is False
    assert "too small" in res["error"]


def test_train_sync_success_no_val(wm, monkeypatch):
    monkeypatch.setattr(wm_mod, "TRAIN_MIN_DATASET_SIZE", 1)
    _setup_train_sync(wm, monkeypatch, train_len=100, val_ds=None,
                      has_eval_strategy=False)  # exercise evaluation_strategy branch
    res = wm._train_sync(wm_mod.WEIGHT_CHECKPOINTS_DIR)
    assert res["success"] is True
    assert res["val_loss"] is None
    assert wm.total_trainings == 1
    assert wm.current_checkpoint is not None
    assert wm.peft_model is None  # merged and cleared
    assert wm._train_progress["status"] == "completed"


def test_train_sync_success_with_val_sets_baseline(wm, monkeypatch):
    monkeypatch.setattr(wm_mod, "TRAIN_MIN_DATASET_SIZE", 1)
    _setup_train_sync(wm, monkeypatch, train_len=100, val_ds=FakeDataset(10),
                      eval_loss=0.15)
    wm._baseline_val_loss = None
    res = wm._train_sync(wm_mod.WEIGHT_CHECKPOINTS_DIR)
    assert res["success"] is True
    assert wm._baseline_val_loss == 0.15  # baseline established


def test_train_sync_degradation_rolls_back(wm, monkeypatch):
    monkeypatch.setattr(wm_mod, "TRAIN_MIN_DATASET_SIZE", 1)
    monkeypatch.setattr(wm_mod, "TRAIN_VAL_LOSS_THRESHOLD", 0.5)
    _setup_train_sync(wm, monkeypatch, train_len=100, val_ds=FakeDataset(10),
                      eval_loss=5.0)  # far above baseline+threshold
    wm._baseline_val_loss = 0.2
    reloaded = {}
    monkeypatch.setattr(wm, "load_model",
                        lambda: reloaded.setdefault("called", True) or {"success": True})
    res = wm._train_sync(wm_mod.WEIGHT_CHECKPOINTS_DIR)
    assert res["success"] is False
    assert "degradation" in res["error"].lower()
    assert wm.total_rollbacks == 1
    assert wm.last_train_time > 0        # cooldown applied to degraded runs
    assert reloaded["called"] is True    # base model reloaded
    assert wm._train_progress["status"] == "rolled_back"
    assert wm.train_history[-1]["status"] == "rolled_back"


# --------------------------------------------------------------------------- #
# _cleanup_checkpoints                                                         #
# --------------------------------------------------------------------------- #
def test_cleanup_checkpoints_keeps_recent(wm, monkeypatch, ckpt_dir):
    monkeypatch.setattr(wm_mod, "TRAIN_MAX_CHECKPOINTS", 2)
    for i in range(5):
        (ckpt_dir / f"lora_{i}").mkdir()
    wm._cleanup_checkpoints()
    remaining = sorted(p.name for p in ckpt_dir.glob("lora_*"))
    assert remaining == ["lora_3", "lora_4"]  # 2 most recent kept


def test_cleanup_checkpoints_swallows_errors(wm, monkeypatch, ckpt_dir):
    monkeypatch.setattr(wm_mod, "TRAIN_MAX_CHECKPOINTS", 0)
    (ckpt_dir / "lora_1").mkdir()
    monkeypatch.setattr(wm_mod.shutil, "rmtree",
                        lambda p: (_ for _ in ()).throw(OSError("locked")))
    wm._cleanup_checkpoints()  # must not raise


# --------------------------------------------------------------------------- #
# rollback_to_checkpoint                                                       #
# --------------------------------------------------------------------------- #
def test_rollback_checkpoint_in_progress(wm):
    wm.training_in_progress = True
    res = wm.rollback_to_checkpoint("/x")
    assert res["success"] is False and "in progress" in res["error"]


def test_rollback_checkpoint_not_found(wm, tmp_path):
    res = wm.rollback_to_checkpoint(str(tmp_path / "missing"))
    assert res["success"] is False and "not found" in res["error"]


def test_rollback_checkpoint_success(wm, monkeypatch, tmp_path):
    cp = tmp_path / "lora_1"
    cp.mkdir()
    monkeypatch.setattr(wm, "load_model", lambda: {"success": True})
    res = wm.rollback_to_checkpoint(str(cp))
    assert res["success"] is True
    assert wm.current_checkpoint == str(cp)
    assert wm.total_rollbacks == 1


def test_rollback_checkpoint_load_fails_no_increment(wm, monkeypatch, tmp_path):
    cp = tmp_path / "lora_1"
    cp.mkdir()
    monkeypatch.setattr(wm, "load_model", lambda: {"success": False, "error": "x"})
    res = wm.rollback_to_checkpoint(str(cp))
    assert res["success"] is False
    assert wm.total_rollbacks == 0


def test_rollback_checkpoint_exception(wm, monkeypatch, tmp_path):
    cp = tmp_path / "lora_1"
    cp.mkdir()
    monkeypatch.setattr(wm, "load_model",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    res = wm.rollback_to_checkpoint(str(cp))
    assert res["success"] is False and "boom" in res["error"]


# --------------------------------------------------------------------------- #
# rollback_to_base                                                             #
# --------------------------------------------------------------------------- #
def test_rollback_base_in_progress(wm):
    wm.training_in_progress = True
    res = wm.rollback_to_base()
    assert res["success"] is False and "in progress" in res["error"]


def test_rollback_base_success(wm):
    wm.model = object()
    wm.current_checkpoint = "/x"
    res = wm.rollback_to_base()
    assert res["success"] is True
    assert wm.model is None
    assert wm.current_checkpoint is None
    assert wm.total_rollbacks == 1


def test_rollback_base_exception(wm, monkeypatch):
    monkeypatch.setattr(wm, "_save_stats",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    res = wm.rollback_to_base()
    assert res["success"] is False and "boom" in res["error"]


# --------------------------------------------------------------------------- #
# generate                                                                     #
# --------------------------------------------------------------------------- #
def test_generate_not_loaded_raises(wm):
    with pytest.raises(RuntimeError):
        wm.generate("hi")


def test_generate_success(wm):
    wm.model_loaded = True

    class FakeInputs(dict):
        def to(self, device):
            return self

    inputs = FakeInputs(input_ids=types.SimpleNamespace(shape=[1, 3]))

    def tok_call(prompt, return_tensors=None):
        return inputs

    wm.tokenizer = types.SimpleNamespace(
        __call__=tok_call,
        pad_token_id=0,
        decode=lambda ids, skip_special_tokens=True: "  generated  ",
    )
    # tokenizer must be callable
    wm.tokenizer = _CallableTokenizer(inputs)
    wm.model = types.SimpleNamespace(
        device="cpu",
        generate=lambda **kwargs: [[10, 11, 12, 13, 14]],
    )
    out = wm.generate("hello", max_new_tokens=8)
    assert out == "generated"


class _CallableTokenizer:
    def __init__(self, inputs):
        self._inputs = inputs
        self.pad_token_id = 0

    def __call__(self, prompt, return_tensors=None):
        return self._inputs

    def decode(self, ids, skip_special_tokens=True):
        return "  generated  "


# --------------------------------------------------------------------------- #
# list_checkpoints / status                                                    #
# --------------------------------------------------------------------------- #
def test_list_checkpoints_empty(wm):
    assert wm.list_checkpoints() == []


def test_list_checkpoints_with_config(wm, ckpt_dir):
    d = ckpt_dir / "lora_1"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}),
                                           encoding="utf-8")
    wm.current_checkpoint = str(d)
    result = wm.list_checkpoints()
    assert len(result) == 1
    assert result[0]["is_current"] is True
    assert result[0]["lora_r"] == 16
    assert result[0]["lora_alpha"] == 32


def test_list_checkpoints_bad_config(wm, ckpt_dir):
    d = ckpt_dir / "lora_2"
    d.mkdir()
    (d / "adapter_config.json").write_text("not json", encoding="utf-8")
    result = wm.list_checkpoints()
    assert result[0]["name"] == "lora_2"
    assert "lora_r" not in result[0]  # bad config swallowed


def test_list_checkpoints_no_config(wm, ckpt_dir):
    (ckpt_dir / "lora_3").mkdir()
    result = wm.list_checkpoints()
    assert result[0]["name"] == "lora_3"


def test_status_shape(wm, ckpt_dir):
    st = wm.status()
    assert st["model_loaded"] is False
    assert st["total_trainings"] == 0
    assert "lora_config" in st
    assert st["lora_config"]["r"] == wm_mod.LORA_R
    assert "can_train" in st and "can_train_reason" in st
    assert isinstance(st["checkpoints"], list)
