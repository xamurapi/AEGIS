"""Weight Modifier — LoRA fine-tuning of local LLM with checkpointing and rollback."""
import json
import shutil
import asyncio
import logging
from pathlib import Path

from aegis._atomic import atomic_write_text
from aegis.config import (
    LOCAL_MODEL_PATH, LOCAL_MODEL_DEVICE, LOCAL_MODEL_DTYPE, LOCAL_MODEL_QUANTIZE,
    LOCAL_MODEL_MAX_LENGTH, LOCAL_MODEL_TRAIN_OVERHEAD, LOCAL_MODEL_INFER_OVERHEAD,
    quantization_error,
    WEIGHT_CHECKPOINTS_DIR, TRAIN_MAX_CHECKPOINTS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    TRAIN_BATCH_SIZE, TRAIN_GRADIENT_ACCUMULATION, TRAIN_EPOCHS,
    TRAIN_LEARNING_RATE, TRAIN_MIN_INTERVAL_SECONDS,
    TRAIN_MIN_DATASET_SIZE, TRAIN_VAL_LOSS_THRESHOLD,
)
from aegis.clock import CLOCK

logger = logging.getLogger("aegis.weight_modifier")


class WeightModifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.model_loaded = False
        self.training_in_progress = False
        self.last_train_time = 0.0
        self.train_history: list[dict] = []
        self.current_checkpoint: str | None = None
        self.total_trainings = 0
        self.total_rollbacks = 0
        self._baseline_val_loss: float | None = None
        self._train_task: asyncio.Task | None = None
        self._train_lock = asyncio.Lock()
        self._train_progress: dict = {}
        self._stats_path = WEIGHT_CHECKPOINTS_DIR / "training_stats.json"
        self._load_stats()

    def _load_stats(self):
        if self._stats_path.exists():
            try:
                data = json.loads(self._stats_path.read_text(encoding="utf-8"))
                self.total_trainings = data.get("total_trainings", 0)
                self.total_rollbacks = data.get("total_rollbacks", 0)
                self.last_train_time = data.get("last_train_time", 0.0)
                self.current_checkpoint = data.get("current_checkpoint")
                self._baseline_val_loss = data.get("baseline_val_loss")
                self.train_history = data.get("train_history", [])[-20:]
            except Exception:
                logger.warning("Failed to load training stats from %s", self._stats_path, exc_info=True)

    def _save_stats(self):
        data = {
            "total_trainings": self.total_trainings,
            "total_rollbacks": self.total_rollbacks,
            "last_train_time": self.last_train_time,
            "current_checkpoint": self.current_checkpoint,
            "baseline_val_loss": self._baseline_val_loss,
            "train_history": self.train_history[-20:],
        }
        try:
            atomic_write_text(self._stats_path, json.dumps(data))
        except Exception:
            pass

    def can_train(self) -> tuple[bool, str]:
        """Check if training is allowed right now."""
        if self.training_in_progress:
            return False, "Training already in progress"
        elapsed = CLOCK.now() - self.last_train_time
        if elapsed < TRAIN_MIN_INTERVAL_SECONDS and self.last_train_time > 0:
            remaining = int(TRAIN_MIN_INTERVAL_SECONDS - elapsed)
            return False, f"Cooldown: {remaining}s remaining (min interval: {TRAIN_MIN_INTERVAL_SECONDS}s)"
        return True, "Ready"

    # ── pre-flight guards (spec §M8.3a) ──────────────────────────────
    # Refusing at configuration time with a sentence the operator can act on is
    # worth more than a stack trace from inside from_pretrained. Both guards
    # answer the same question — "will this model actually fit here?" — before
    # anything expensive is attempted.

    @staticmethod
    def available_memory_bytes() -> int | None:
        """Physical memory currently available, or None if it cannot be read."""
        try:
            import psutil  # noqa: PLC0415 — optional dependency
            return int(psutil.virtual_memory().available)
        except Exception:
            pass
        try:  # Linux without psutil
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
        try:  # Windows without psutil
            import ctypes  # noqa: PLC0415

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            pass
        return None

    @staticmethod
    def _bytes_per_param(dtype: str) -> int:
        return {"float16": 2, "bfloat16": 2, "float32": 4}.get(dtype, 4)

    @staticmethod
    def _declared_parameter_count(model_path: str) -> int | None:
        """Parameter count inferred from the model id, e.g. '...-3B-Instruct'.

        A local config.json is read when present; otherwise the size suffix in
        the name is the only signal available before downloading gigabytes, and
        it is the signal the operator typed.
        """
        cfg_path = Path(model_path) / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                hidden = int(cfg.get("hidden_size", 0))
                layers = int(cfg.get("num_hidden_layers", 0))
                vocab = int(cfg.get("vocab_size", 0))
                if hidden and layers:
                    # 12·h²  per transformer block (attention + MLP), plus the
                    # embedding and output matrices.
                    return 12 * hidden * hidden * layers + 2 * vocab * hidden
            except Exception:
                logger.debug("Could not read %s for a parameter estimate", cfg_path)
        import re as _re
        match = _re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z0-9])", str(model_path))
        if match:
            return int(float(match.group(1)) * 1_000_000_000)
        match = _re.search(r"(\d+(?:\.\d+)?)\s*[mM](?![a-zA-Z0-9])", str(model_path))
        if match:
            return int(float(match.group(1)) * 1_000_000)
        return None

    @classmethod
    def memory_check(cls, model_path: str = LOCAL_MODEL_PATH, *,
                     for_training: bool = False) -> dict:
        """Would this model fit? Returns a verdict rather than raising.

        ``ok=False`` carries everything an operator needs to fix it: which
        model, how much it needs, how much there is, and what to do.
        """
        params = cls._declared_parameter_count(model_path)
        available = cls.available_memory_bytes()
        overhead = LOCAL_MODEL_TRAIN_OVERHEAD if for_training else LOCAL_MODEL_INFER_OVERHEAD
        if params is None:
            return {"ok": True, "reason": "unknown_size", "params": None,
                    "required_bytes": None, "available_bytes": available}
        required = int(params * cls._bytes_per_param(LOCAL_MODEL_DTYPE) * overhead)
        verdict = {"ok": True, "reason": "fits", "params": params,
                   "required_bytes": required, "available_bytes": available,
                   "dtype": LOCAL_MODEL_DTYPE, "for_training": for_training}
        if available is None:
            verdict["reason"] = "available_memory_unknown"
            return verdict
        if required > available:
            verdict["ok"] = False
            verdict["reason"] = (
                f"{model_path} has ~{params / 1e9:.2f}B parameters; "
                f"{'training' if for_training else 'loading'} it in "
                f"{LOCAL_MODEL_DTYPE} needs ~{required / 2**30:.1f} GiB but only "
                f"~{available / 2**30:.1f} GiB is available. Choose a smaller "
                f"model (the reference profile tops out at 3B for training), "
                f"or set LOCAL_MODEL_DTYPE=bfloat16, or free memory."
            )
        return verdict

    def load_model(self) -> dict:
        """Load the base model and tokenizer."""
        if self.model_loaded:
            return {"success": True, "message": "Model already loaded"}

        quant_error = quantization_error()
        if quant_error:
            return {"success": False, "error": quant_error}
        fit = self.memory_check(LOCAL_MODEL_PATH, for_training=False)
        if not fit["ok"]:
            return {"success": False, "error": fit["reason"], "memory_check": fit}

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            logger.info(f"Loading model: {LOCAL_MODEL_PATH}")

            # Quantization config
            quantization_config = None
            if LOCAL_MODEL_QUANTIZE == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, LOCAL_MODEL_DTYPE),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif LOCAL_MODEL_QUANTIZE == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # Determine dtype
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(LOCAL_MODEL_DTYPE, torch.float16)

            self.tokenizer = AutoTokenizer.from_pretrained(
                LOCAL_MODEL_PATH, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            load_kwargs = {
                "pretrained_model_name_or_path": LOCAL_MODEL_PATH,
                "torch_dtype": torch_dtype,
                "device_map": LOCAL_MODEL_DEVICE,
                "trust_remote_code": True,
            }
            if quantization_config:
                load_kwargs["quantization_config"] = quantization_config

            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

            # Apply existing LoRA checkpoint if available
            if self.current_checkpoint:
                cp_path = Path(self.current_checkpoint)
                if cp_path.exists():
                    self._apply_lora_checkpoint(cp_path)

            self.model_loaded = True
            logger.info("Model loaded successfully")
            return {"success": True, "message": f"Model loaded: {LOCAL_MODEL_PATH}"}

        except ImportError as e:
            return {"success": False, "error": f"Missing dependency: {e}. Install: pip install torch transformers peft"}
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return {"success": False, "error": str(e)}

    def _apply_lora_checkpoint(self, checkpoint_path: Path):
        """Apply a LoRA adapter checkpoint to the base model."""
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(checkpoint_path))
            self.model = self.model.merge_and_unload()
            logger.info(f"Applied LoRA checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to apply checkpoint {checkpoint_path}: {e}")

    @staticmethod
    def module_leaf_names(model) -> list[str] | None:
        """Distinct leaf module names of a model, sorted, or None if unknowable.

        These are the names LoRA matches against; showing them is what turns
        "training changed nothing" into an actionable error message. A model
        object that cannot be walked returns None rather than an empty list —
        "I could not look" and "I looked and found nothing" must not collapse
        into the same answer, because only the second one is a mismatch.
        """
        walk = getattr(model, "named_modules", None)
        if not callable(walk):
            return None
        try:
            names = {full_name.rsplit(".", 1)[-1]
                     for full_name, _ in walk() if full_name}
        except Exception:
            logger.debug("Could not enumerate model modules", exc_info=True)
            return None
        return sorted(names)

    @classmethod
    def matching_lora_targets(cls, model, targets=LORA_TARGET_MODULES) -> list[str]:
        """Which of the configured target modules actually exist in ``model``.

        When the model cannot be introspected the configured list is returned
        unchanged: refusing to train on the grounds that we could not check
        would be a worse failure than the one being guarded against.
        """
        present = cls.module_leaf_names(model)
        if present is None:
            return list(targets)
        return [t for t in targets if t in set(present)]

    def _prepare_lora(self):
        """Attach LoRA adapters to the model for training.

        The target-module list is architecture-specific (q/k/v/o_proj for Llama,
        Qwen, Mistral and Gemma; qkv_proj for Phi-3). With a mismatched list
        ``peft`` used to attach nothing and training would then run to
        completion having updated zero parameters — expensive, plausible-looking
        and worthless. Verified here instead, with the available names in the
        error (§M8.3c).
        """
        from peft import LoraConfig, get_peft_model, TaskType

        matched = self.matching_lora_targets(self.model)
        if not matched:
            available = self.module_leaf_names(self.model) or []
            projection_like = [n for n in available if "proj" in n or "attn" in n]
            raise ValueError(
                f"LORA_TARGET_MODULES={list(LORA_TARGET_MODULES)} matches no module "
                f"in {LOCAL_MODEL_PATH}. Training would update zero parameters. "
                f"Projection-like modules this model does have: "
                f"{projection_like or available[:40]}. "
                f"Set LORA_TARGET_MODULES to a comma-separated subset of those."
            )

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=matched,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.peft_model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.peft_model.parameters())
        logger.info(f"LoRA attached: {trainable:,} trainable / {total:,} total params "
                     f"({trainable / max(1, total) * 100:.2f}%)")
        if trainable == 0:
            raise ValueError(
                f"LoRA attached to {matched} but produced zero trainable "
                f"parameters — refusing to run a training cycle that cannot "
                f"change anything."
            )
        return {"trainable_params": trainable, "total_params": total,
                "target_modules": matched}

    def _load_dataset(self, dataset_dir: Path) -> tuple:
        """Load training and validation datasets from JSONL files."""
        from datasets import load_dataset

        train_path = dataset_dir / "train.jsonl"
        val_path = dataset_dir / "val.jsonl"

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        data_files = {"train": str(train_path)}
        if val_path.exists():
            data_files["validation"] = str(val_path)

        dataset = load_dataset("json", data_files=data_files)
        return dataset.get("train"), dataset.get("validation")

    def _tokenize_sample(self, sample: dict) -> dict:
        """Format and tokenize a single instruction-response sample."""
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=LOCAL_MODEL_MAX_LENGTH,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    async def train(self, dataset_dir: Path, ethics_approved: bool = False) -> dict:
        """Run LoRA fine-tuning on the dataset. Returns training result."""
        if not ethics_approved:
            return {"success": False, "error": "Ethics approval required before training"}

        # Training needs the weights AND their gradients AND optimizer state, so
        # it is checked against a larger budget than plain inference (§M8.3a).
        fit = self.memory_check(LOCAL_MODEL_PATH, for_training=True)
        if not fit["ok"]:
            return {"success": False, "error": fit["reason"], "memory_check": fit}

        # Atomic check-and-set: without the lock two concurrent train() calls
        # could both pass can_train() before either sets the flag.
        async with self._train_lock:
            can, reason = self.can_train()
            if not can:
                return {"success": False, "error": reason}
            self.training_in_progress = True

        self._train_progress = {"status": "preparing", "epoch": 0, "loss": 0.0}

        try:
            if not self.model_loaded:
                # load_model() loads/quantizes a multi-GB model (and may download
                # it) — running it directly in the coroutine would freeze the
                # whole tick loop for minutes (audit H2). Offload to an executor.
                load_result = await asyncio.get_running_loop().run_in_executor(
                    None, self.load_model
                )
                if not load_result["success"]:
                    return load_result

            result = await asyncio.get_running_loop().run_in_executor(
                None, self._train_sync, dataset_dir
            )
            return result
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.training_in_progress = False

    def _train_sync(self, dataset_dir: Path) -> dict:
        """Synchronous training logic (runs in executor)."""
        import inspect
        import torch
        from transformers import TrainingArguments, Trainer, TrainerCallback

        # Load dataset
        train_dataset, val_dataset = self._load_dataset(dataset_dir)
        if len(train_dataset) < TRAIN_MIN_DATASET_SIZE:
            return {
                "success": False,
                "error": f"Dataset too small: {len(train_dataset)} < {TRAIN_MIN_DATASET_SIZE} minimum",
            }

        # Tokenize
        train_dataset = train_dataset.map(self._tokenize_sample, remove_columns=train_dataset.column_names)
        if val_dataset:
            val_dataset = val_dataset.map(self._tokenize_sample, remove_columns=val_dataset.column_names)

        # Prepare LoRA
        lora_info = self._prepare_lora()

        # Checkpoint path
        timestamp = int(CLOCK.now())
        checkpoint_dir = WEIGHT_CHECKPOINTS_DIR / f"lora_{timestamp}"

        # Progress callback
        progress_ref = self._train_progress

        class ProgressCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, **kwargs):
                progress_ref["epoch"] = state.epoch
                progress_ref["status"] = "training"

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    progress_ref["loss"] = logs.get("loss", 0.0)
                    progress_ref["step"] = state.global_step
                    progress_ref["total_steps"] = state.max_steps

        # Training args. The eval-strategy kwarg was renamed from
        # `evaluation_strategy` to `eval_strategy` in transformers 4.46, so we
        # pick whichever the installed version accepts.
        eval_value = "epoch" if val_dataset else "no"
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        eval_kwarg = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=TRAIN_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=TRAIN_GRADIENT_ACCUMULATION,
            learning_rate=TRAIN_LEARNING_RATE,
            warmup_ratio=0.1,
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=1,
            fp16=LOCAL_MODEL_DTYPE == "float16" and torch.cuda.is_available(),
            bf16=LOCAL_MODEL_DTYPE == "bfloat16" and torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=False,
            **{eval_kwarg: eval_value},
        )

        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[ProgressCallback()],
        )

        self._train_progress["status"] = "training"
        train_result = trainer.train()

        # Evaluate
        val_loss = None
        if val_dataset:
            eval_result = trainer.evaluate()
            val_loss = eval_result.get("eval_loss", None)

        # Check for degradation
        degraded = False
        if val_loss is not None and self._baseline_val_loss is not None:
            if val_loss > self._baseline_val_loss + TRAIN_VAL_LOSS_THRESHOLD:
                degraded = True
                logger.warning(f"Degradation detected: val_loss={val_loss:.4f} > "
                               f"baseline={self._baseline_val_loss:.4f} + threshold={TRAIN_VAL_LOSS_THRESHOLD}")

        if degraded:
            # Rollback — don't save this checkpoint
            self.total_rollbacks += 1
            # Apply the cooldown to degraded runs too. If last_train_time were
            # only set on success, a degraded run would leave the cooldown clear
            # and could retrain immediately — an unbounded degradation loop.
            self.last_train_time = CLOCK.now()
            self._train_progress["status"] = "rolled_back"
            # Reload base model to undo LoRA. Drop the degraded model first so
            # its (V)RAM is released before the fresh copy loads.
            self.model_loaded = False
            self.peft_model = None
            self.model = None
            self.load_model()

            record = {
                "timestamp": CLOCK.now(),
                "dataset_dir": str(dataset_dir),
                "status": "rolled_back",
                "train_loss": train_result.training_loss,
                "val_loss": val_loss,
                "reason": "degradation_exceeded_threshold",
            }
            self.train_history.append(record)
            self._save_stats()

            return {
                "success": False,
                "error": "Training caused degradation — rolled back",
                "train_loss": round(train_result.training_loss, 4),
                "val_loss": round(val_loss, 4) if val_loss is not None else None,
                "baseline_val_loss": round(self._baseline_val_loss, 4) if self._baseline_val_loss is not None else None,
            }

        # Save LoRA adapter
        self.peft_model.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))

        # Merge LoRA into base model for inference
        self.model = self.peft_model.merge_and_unload()
        self.peft_model = None

        # Update baseline
        if val_loss is not None:
            if self._baseline_val_loss is None or val_loss < self._baseline_val_loss:
                self._baseline_val_loss = val_loss

        # Update state
        self.current_checkpoint = str(checkpoint_dir)
        self.last_train_time = CLOCK.now()
        self.total_trainings += 1
        self._train_progress["status"] = "completed"

        record = {
            "timestamp": CLOCK.now(),
            "dataset_dir": str(dataset_dir),
            "checkpoint": str(checkpoint_dir),
            "status": "applied",
            "train_loss": train_result.training_loss,
            "val_loss": val_loss,
            "train_samples": len(train_dataset),
            "epochs": TRAIN_EPOCHS,
            "lora_r": LORA_R,
        }
        self.train_history.append(record)
        self._save_stats()

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        return {
            "success": True,
            "checkpoint": str(checkpoint_dir),
            "train_loss": round(train_result.training_loss, 4),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "train_samples": len(train_dataset),
            "trainable_params": lora_info["trainable_params"],
        }

    def _cleanup_checkpoints(self):
        """Keep only the N most recent checkpoints."""
        dirs = sorted(WEIGHT_CHECKPOINTS_DIR.glob("lora_*"), reverse=True)
        for d in dirs[TRAIN_MAX_CHECKPOINTS:]:
            try:
                shutil.rmtree(d)
                logger.info(f"Removed old checkpoint: {d}")
            except Exception:
                pass

    def rollback_to_checkpoint(self, checkpoint_path: str) -> dict:
        """Rollback to a specific checkpoint."""
        if self.training_in_progress:
            # Mutating model/peft_model while _train_sync runs in the executor
            # would corrupt the in-flight run (save_pretrained on a freed model).
            return {"success": False, "error": "Training in progress — cannot roll back now"}
        cp = Path(checkpoint_path)
        if not cp.exists():
            return {"success": False, "error": f"Checkpoint not found: {checkpoint_path}"}

        try:
            self.model_loaded = False
            self.peft_model = None
            self.current_checkpoint = checkpoint_path
            self._save_stats()
            result = self.load_model()
            if result["success"]:
                self.total_rollbacks += 1
                self._save_stats()
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def rollback_to_base(self) -> dict:
        """Rollback to the original base model (no LoRA)."""
        if self.training_in_progress:
            return {"success": False, "error": "Training in progress — cannot roll back now"}
        try:
            self.model_loaded = False
            self.model = None
            self.peft_model = None
            self.current_checkpoint = None
            self.total_rollbacks += 1
            self._save_stats()
            return {"success": True, "message": "Rolled back to base model. Will reload on next use."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate text using the loaded model (for local inference)."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            # Greedy (do_sample=False) is deterministic — the project's
            # "zero randomness" guarantee. Sampling with temperature/top_p and no
            # seed made identical prompts yield different completions, so those
            # knobs are dropped entirely rather than left inert.
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def list_checkpoints(self) -> list[dict]:
        """List all available checkpoints."""
        dirs = sorted(WEIGHT_CHECKPOINTS_DIR.glob("lora_*"), reverse=True)
        result = []
        for d in dirs:
            config_file = d / "adapter_config.json"
            info = {"path": str(d), "name": d.name, "is_current": str(d) == self.current_checkpoint}
            if config_file.exists():
                try:
                    cfg = json.loads(config_file.read_text())
                    info["lora_r"] = cfg.get("r", "?")
                    info["lora_alpha"] = cfg.get("lora_alpha", "?")
                except Exception:
                    pass
            result.append(info)
        return result

    def status(self) -> dict:
        return {
            "model_loaded": self.model_loaded,
            "model_path": LOCAL_MODEL_PATH,
            "device": LOCAL_MODEL_DEVICE,
            "dtype": LOCAL_MODEL_DTYPE,
            "quantize": LOCAL_MODEL_QUANTIZE or "none",
            "training_in_progress": self.training_in_progress,
            "train_progress": self._train_progress,
            "total_trainings": self.total_trainings,
            "total_rollbacks": self.total_rollbacks,
            "last_train_time": self.last_train_time,
            "current_checkpoint": self.current_checkpoint,
            "baseline_val_loss": self._baseline_val_loss,
            "checkpoints": self.list_checkpoints(),
            "recent_history": self.train_history[-5:],
            "can_train": self.can_train()[0],
            "can_train_reason": self.can_train()[1],
            "lora_config": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_modules": LORA_TARGET_MODULES,
            },
        }
