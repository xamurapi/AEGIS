"""Layer 3: Self-Modification — sandbox testing, rollback & weight modification (SM-001..SM-008).

Sandbox testing is real — validates parameter bounds, checks for degradation
using actual metric history, and never uses random outcomes.
"""
import time
import copy
import logging

logger = logging.getLogger("aegis.self_modification")

# Cap on each in-memory audit list so a long-running process cannot grow them
# without bound (mirrors _metric_history's 50-entry cap).
_MAX_RECORDS = 200


class SelfModification:
    def __init__(self):
        self.modifications: list[dict] = []
        self.sandbox_results: list[dict] = []
        self.rollbacks: list[dict] = []
        self.current_version = "1.0.0"
        self.parameters: dict[str, float] = {
            "learning_rate": 0.001,
            "attention_heads": 12,
            "dropout": 0.1,
            "temperature": 0.7,
            "curiosity_weight": 0.5,
            "memory_decay": 0.02,
        }
        self._stable_checkpoint = copy.deepcopy(self.parameters)

        # Parameter bounds — real constraints instead of random accept/reject
        self._param_bounds: dict[str, tuple[float, float]] = {
            "learning_rate": (0.00001, 0.01),
            "attention_heads": (4, 32),
            "dropout": (0.0, 0.5),
            "temperature": (0.1, 1.5),
            "curiosity_weight": (0.1, 1.0),
            "memory_decay": (0.001, 0.1),
        }
        # History of metric after each applied modification
        self._metric_history: list[float] = [0.5]  # baseline metric

        # Code modification reference (set externally by substrate)
        self.code_modifier = None

        # Weight modification state
        self.weight_modifier = None  # set externally by substrate
        self.dataset_builder = None  # set externally by substrate
        self.weight_modifications: list[dict] = []
        self.weight_mod_total = 0
        self.weight_mod_success = 0
        self.weight_mod_rollbacks = 0

    @staticmethod
    def _cap(lst: list, item: dict, limit: int = _MAX_RECORDS) -> dict:
        """Append `item` to `lst`, then trim it in place to the last `limit`
        entries so the audit lists never grow without bound."""
        lst.append(item)
        excess = len(lst) - limit
        if excess > 0:
            del lst[:excess]
        return item

    def propose_modification(self, mod_type: str, target: str, new_value: float) -> dict:
        proposal = {
            "id": f"mod_{len(self.modifications):04d}",
            "timestamp": time.time(),
            "type": mod_type,  # parametric, architectural, meta
            "target": target,
            "old_value": self.parameters.get(target, None),
            "new_value": new_value,
            "status": "proposed",
        }
        return proposal

    def sandbox_test(self, proposal: dict, current_metric: float = 0.5) -> dict:
        """Real sandbox test — validates bounds, checks metric trend, applies constraints.

        current_metric: a real performance metric (0..1) from the system, e.g.
        derived from success_rate, energy, error_rate.
        """
        target = proposal["target"]
        new_value = proposal["new_value"]
        old_value = proposal["old_value"]

        tests_run = 0
        tests_passed = 0
        reasons = []

        # Test 1: Bounds check
        tests_run += 1
        if target in self._param_bounds:
            lo, hi = self._param_bounds[target]
            if lo <= new_value <= hi:
                tests_passed += 1
            else:
                reasons.append(f"Out of bounds: {new_value} not in [{lo}, {hi}]")
        else:
            tests_passed += 1  # unknown param — no bounds to violate

        # Test 2: Change magnitude — reject changes larger than 20%
        tests_run += 1
        if old_value and old_value != 0:
            change_pct = abs(new_value - old_value) / abs(old_value)
            if change_pct <= 0.2:
                tests_passed += 1
            else:
                reasons.append(f"Change too large: {change_pct*100:.1f}% > 20%")
        else:
            tests_passed += 1

        # Test 3: Metric trend — check if recent modifications have been improving
        tests_run += 1
        if len(self._metric_history) >= 3:
            recent = self._metric_history[-3:]
            trend = recent[-1] - recent[0]
            if trend >= -0.05:  # not declining significantly
                tests_passed += 1
            else:
                reasons.append(f"Declining metric trend: {trend:.4f}")
        else:
            tests_passed += 1  # not enough history

        # Test 4: Current metric is acceptable
        tests_run += 1
        if current_metric >= 0.3:
            tests_passed += 1
        else:
            reasons.append(f"Current metric too low: {current_metric:.3f}")

        # Test 5: Parameter-specific sanity checks
        tests_run += 1
        sane = True
        if target == "learning_rate" and new_value > 0.005:
            sane = False
            reasons.append("Learning rate dangerously high")
        if target == "dropout" and new_value > 0.4:
            sane = False
            reasons.append("Dropout too high — risk of underfitting")
        if target == "memory_decay" and new_value > 0.08:
            sane = False
            reasons.append("Memory decay too fast — knowledge loss risk")
        if sane:
            tests_passed += 1

        passed = tests_passed == tests_run
        # Estimate improvement from metric history
        avg_metric = sum(self._metric_history[-5:]) / max(1, len(self._metric_history[-5:]))
        metric_change = current_metric - avg_metric

        result = {
            "proposal_id": proposal["id"],
            "timestamp": time.time(),
            "passed": passed,
            "metric_change": round(metric_change, 4),
            "degradation": round(max(0, -metric_change), 4),
            "tests_run": tests_run,
            "tests_passed": tests_passed,
            "reasons": reasons,
        }
        self._cap(self.sandbox_results, result)

        # Record metric for trend tracking
        self._metric_history.append(current_metric)
        if len(self._metric_history) > 50:
            self._metric_history = self._metric_history[-50:]

        return result

    def apply_modification(self, proposal: dict, sandbox_result: dict) -> dict:
        if not sandbox_result["passed"]:
            proposal["status"] = "rejected"
            self._cap(self.modifications, proposal)
            return {"applied": False, "reason": "sandbox_failed"}

        if sandbox_result["degradation"] > 0.05:
            # The proposal has NOT been applied to self.parameters yet (that
            # happens below), so there is nothing to revert — just reject it.
            # Restoring _stable_checkpoint here would wrongly undo the PREVIOUS
            # accepted modification. Keep the audit trail without mutating state.
            proposal["status"] = "rejected_degradation"
            self._cap(self.rollbacks, {
                "timestamp": time.time(),
                "proposal_id": proposal["id"],
                "reason": "degradation_exceeded_threshold",
            })
            self._cap(self.modifications, proposal)
            return {"applied": False, "reason": "degradation_too_high"}

        if proposal["target"] in self.parameters:
            self._stable_checkpoint = copy.deepcopy(self.parameters)
            self.parameters[proposal["target"]] = proposal["new_value"]
            proposal["status"] = "applied"
            self.current_version = self._bump_patch(self.current_version)
        else:
            proposal["status"] = "target_not_found"

        self._cap(self.modifications, proposal)
        return {"applied": proposal["status"] == "applied", "version": self.current_version}

    @staticmethod
    def _version_parts(version: str) -> list[int]:
        """Parse a version into [major, minor, patch] ints, tolerating
        non-3-part / non-numeric forms ("1.0", "1.0.0-beta")."""
        parts = (version or "0.0.0").split(".")
        while len(parts) < 3:
            parts.append("0")
        out = []
        for p in parts[:3]:
            digits = "".join(c for c in p if c.isdigit())
            out.append(int(digits) if digits else 0)
        return out

    @classmethod
    def _bump_patch(cls, version: str) -> str:
        major, minor, patch = cls._version_parts(version)
        return f"{major}.{minor}.{patch + 1}"

    @classmethod
    def _bump_minor(cls, version: str) -> str:
        major, minor, _ = cls._version_parts(version)
        return f"{major}.{minor + 1}.0"

    def _rollback(self, proposal: dict):
        self.parameters = copy.deepcopy(self._stable_checkpoint)
        self._cap(self.rollbacks, {
            "timestamp": time.time(),
            "proposal_id": proposal["id"],
            "reason": "degradation_exceeded_threshold",
        })
        proposal["status"] = "rolled_back"
        self._cap(self.modifications, proposal)

    async def propose_weight_modification(self, memory, agent_system=None, ethics_core=None) -> dict:
        """Propose and execute a weight modification via LoRA fine-tuning.

        Full pipeline: build dataset -> ethics check -> train -> validate -> apply/rollback.
        """
        if not self.weight_modifier or not self.dataset_builder:
            return {"success": False, "error": "Weight modifier or dataset builder not configured"}

        can_train, reason = self.weight_modifier.can_train()
        if not can_train:
            return {"success": False, "error": reason}

        record = {
            "id": f"wmod_{self.weight_mod_total:04d}",
            "timestamp": time.time(),
            "status": "building_dataset",
        }
        self.weight_mod_total += 1

        # Step 1: Build dataset from memory
        logger.info("Weight modification: building dataset from memory...")
        dataset_result = self.dataset_builder.build_from_memory(memory, agent_system)
        if not dataset_result.get("success"):
            record["status"] = "dataset_failed"
            record["error"] = dataset_result.get("error", "Dataset build failed")
            self._cap(self.weight_modifications, record)
            return record

        record["dataset_size"] = dataset_result["total_size"]
        record["dataset_dir"] = dataset_result["dataset_dir"]

        # Step 2: Ethics check. Only an explicit "approved" verdict clears
        # training — a "review_required" (or any non-approved) status must NOT
        # be treated as approval and auto-proceed. Fail CLOSED if no ethics core
        # is wired: self-modifying training must never run ungated.
        if not ethics_core:
            record["status"] = "ethics_unavailable"
            record["error"] = "Ethics core not configured — training blocked"
            self._cap(self.weight_modifications, record)
            return record
        ethics_approved = True
        if ethics_core:
            eth_result = ethics_core.evaluate_action({
                "type": "weight_modification",
                "modifies_self": True,
                "irreversible": False,  # we have rollback
                "confidence": 0.7,
                "description": f"LoRA fine-tuning on {dataset_result['total_size']} samples",
            })
            if eth_result["status"] != "approved":
                record["status"] = "ethics_blocked" if eth_result["status"] == "blocked" else "ethics_review_required"
                record["ethics_score"] = eth_result["score"]
                self._cap(self.weight_modifications, record)
                return record
            ethics_approved = True

        # Step 3: Train
        record["status"] = "training"
        logger.info(f"Weight modification: starting LoRA training on {dataset_result['total_size']} samples...")
        from pathlib import Path
        # Guard the (expensive, external) training call: an exception here must
        # never leave the record stuck at status="training" and unrecorded, nor
        # desync the counters. Fail CLOSED to a recorded "failed" outcome.
        try:
            train_result = await self.weight_modifier.train(
                Path(dataset_result["dataset_dir"]),
                ethics_approved=ethics_approved,
            )
            if not isinstance(train_result, dict):
                train_result = {"success": False, "error": "Training returned no result"}
        except Exception as exc:
            logger.warning("Weight modification training raised", exc_info=True)
            train_result = {"success": False, "error": f"training_exception: {exc}"}

        if train_result.get("success"):
            record["status"] = "applied"
            record["checkpoint"] = train_result.get("checkpoint")
            record["train_loss"] = train_result.get("train_loss")
            record["val_loss"] = train_result.get("val_loss")
            self.weight_mod_success += 1

            # Bump minor version, tolerating non-3-part / non-numeric versions
            # restored from a checkpoint — a raw split()+int() here would crash
            # AFTER an expensive successful training and lose the record.
            self.current_version = self._bump_minor(self.current_version)
            record["version"] = self.current_version

            logger.info(f"Weight modification applied: loss={train_result.get('train_loss')}, "
                         f"version={self.current_version}")
        else:
            record["status"] = "failed"
            record["error"] = train_result.get("error", "Training failed")
            if "rolled_back" in train_result.get("error", ""):
                record["status"] = "rolled_back"
                self.weight_mod_rollbacks += 1

        self._cap(self.weight_modifications, record)
        return record

    def rollback_weights(self, checkpoint_path: str = None) -> dict:
        """Rollback weight modifications."""
        if not self.weight_modifier:
            return {"success": False, "error": "Weight modifier not configured"}

        if checkpoint_path:
            result = self.weight_modifier.rollback_to_checkpoint(checkpoint_path)
        else:
            result = self.weight_modifier.rollback_to_base()

        if result.get("success"):
            self.weight_mod_rollbacks += 1
            self._cap(self.weight_modifications, {
                "id": f"wmod_rollback_{int(time.time())}",
                "timestamp": time.time(),
                "status": "manual_rollback",
                "target": checkpoint_path or "base_model",
            })
        return result

    def status(self) -> dict:
        applied = sum(1 for m in self.modifications if m["status"] == "applied")
        # A proposal whose target does not exist, or that was rejected for
        # degradation, is a rejection too — otherwise success_rate is inflated.
        rejected = sum(1 for m in self.modifications
                       if m["status"] in ("rejected", "rolled_back",
                                          "target_not_found", "rejected_degradation"))

        weight_status = {}
        if self.weight_modifier:
            weight_status = self.weight_modifier.status()

        dataset_status = {}
        if self.dataset_builder:
            dataset_status = self.dataset_builder.status()

        return {
            "version": self.current_version,
            "parameters": self.parameters,
            "total_modifications": len(self.modifications),
            "applied": applied,
            "rejected": rejected,
            "rollbacks": len(self.rollbacks),
            "success_rate": round(applied / max(1, applied + rejected) * 100, 1),
            "recent_modifications": self.modifications[-5:],
            "recent_sandbox": self.sandbox_results[-5:],
            # Weight modification stats
            "weight_mod_total": self.weight_mod_total,
            "weight_mod_success": self.weight_mod_success,
            "weight_mod_rollbacks": self.weight_mod_rollbacks,
            "recent_weight_mods": self.weight_modifications[-5:],
            "weight_modifier": weight_status,
            "dataset_builder": dataset_status,
        }
