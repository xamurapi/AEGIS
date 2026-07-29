"""The behaviour policy: the fifth link, made into an object (spec M3).

    действие → результат → оценка → новое знание → **изменение поведения**

Before this, experience reached behaviour through exactly one channel: a
confidence penalty for causes with a history of failure. That is a real link,
but it is a whisper — it cannot say "not this, here", it cannot be inspected,
and nothing measures whether it changed a single decision.

This contour makes the last arrow explicit. Two mechanisms, deliberately
different in temperament:

* :class:`~aegis.layers.policy.store.PolicyStore` — a weight per
  ``(state, action)``, updated on every closed experience. Fast, quiet, never
  certain. It shifts rankings.
* :class:`~aegis.layers.policy.rules.RuleLifecycle` — explicit rules with
  evidence, a controlled trial, a measured effect and an expiry. Slow, loud,
  and able to say no. It removes options.

Everything here is subordinate to the gates that follow it. A rule can drop a
plan from consideration, but it cannot approve one, cannot touch a
safety-critical action, and cannot outrank ethics — which run afterwards and
are not negotiable (Appendix J).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import aegis.config as cfg
from aegis.layers.policy.counterfactual import ShadowEvaluator
from aegis.layers.policy.rules import (
    ACTIVE, CANDIDATE, PREFER, PREFER_BONUS, REFUTED, RETIRED, SUPPRESS, TRIAL,
    Rule, RuleLifecycle, RuleMiner, rule_id,
)
from aegis.layers.policy.store import PolicyStore

logger = logging.getLogger("aegis.policy")

__all__ = ["BehaviourPolicy", "PolicyStore", "Rule", "RuleMiner",
           "RuleLifecycle", "ShadowEvaluator", "ACTIVE", "CANDIDATE", "PREFER",
           "REFUTED", "RETIRED", "SUPPRESS", "TRIAL", "rule_id"]

#: How many experience rows the miner keeps to mine from. Enough for the
#: default `POLICY_MIN_SUPPORT` of 20 to be reachable in several distinct
#: (state, action) cells at once, bounded so the log cannot grow forever (§3.2).
MAX_EXPERIENCE_ROWS = 20000


class BehaviourPolicy:
    """Facade over preferences, rules, the trial harness and the measurements."""

    def __init__(self, store_dir: Path | None = None, telemetry=None,
                 min_support: int | None = None, weight: float | None = None,
                 trial_ticks: int | None = None):
        root = Path(store_dir) if store_dir is not None else cfg.POLICY_DIR
        self.telemetry = telemetry
        self.store = PolicyStore(store_path=root / "preferences.json",
                                 weight=weight)
        self.miner = RuleMiner(min_support=min_support)
        self.lifecycle = RuleLifecycle(store_path=root / "rules.json")
        self.shadow = ShadowEvaluator(store_path=root / "shadow.jsonl",
                                      trial_ticks=trial_ticks)
        self._experience_path = root / "experiences.jsonl"

        #: The miner's input. Held in memory and mirrored to disk, because
        #: mining walks it repeatedly and re-reading a JSONL log on every
        #: generation would put the file system inside the cognitive cycle.
        self.experiences: list[dict] = []
        #: rule id -> whether it acted, for the tick currently in flight. REFLECT
        #: reads this to credit the realised reward to the right arm.
        self._eligible_this_tick: dict[str, bool] = {}
        self.last_mined_tick = 0
        self.last_reviewed_tick = 0
        self.suppressions = 0
        self.promotions = 0
        self._load_experiences()

    # ── genome ───────────────────────────────────────────────────────

    def set_genome(self, genome: dict) -> None:
        """Adopt the evolved policy parameters (Appendix C)."""
        genome = genome or {}
        if "policy_weight" in genome:
            try:
                self.store.weight = max(0.0, min(1.0, float(genome["policy_weight"])))
            except (TypeError, ValueError):
                logger.debug("Ignoring unusable policy_weight")
        if "policy_min_support" in genome:
            try:
                self.miner.min_support = max(5, int(genome["policy_min_support"]))
            except (TypeError, ValueError):
                logger.debug("Ignoring unusable policy_min_support")

    # ── the planner's hooks ──────────────────────────────────────────

    def delta(self, state, action: str) -> float:
        """Preference contribution to a plan's score (Appendix J, step 4)."""
        return self.store.delta(state, action)

    def apply_rules(self, state, plans, tick: int = 0):
        """Step 5: rules suppress and promote. Returns the surviving plans.

        Trial rules act only on their ABAB "applied" blocks; on the withheld
        blocks they are recorded as eligible and left inert, which is what makes
        the two arms comparable. Active rules always act.
        """
        self._eligible_this_tick = {}
        if not plans:
            return list(plans)

        before = plans[0].action if plans else None
        acting, inert = self._rules_for(state, tick)
        for rule in inert:
            self._eligible_this_tick.setdefault(rule.id, False)

        surviving = []
        for plan in plans:
            action = plan.action
            if action is None:
                surviving.append(plan)
                continue
            suppressed = False
            for rule in acting:
                if not rule.matches(state, action):
                    continue
                self._eligible_this_tick[rule.id] = True
                if rule.effect == SUPPRESS:
                    if plan.safety_critical:
                        # §M3.5: the policy may not switch off the things that
                        # keep the system alive, however good the evidence.
                        continue
                    suppressed = True
                    self.suppressions += 1
                elif rule.effect == PREFER:
                    plan.score += PREFER_BONUS
                    self.promotions += 1
            if not suppressed:
                surviving.append(plan)

        # An eligible-but-inert rule still counts as eligible for the trial.
        for rule in inert:
            if any(rule.matches(state, plan.action) for plan in plans):
                self._eligible_this_tick[rule.id] = False

        surviving.sort(key=lambda plan: (-plan.score, plan.objective))
        after = surviving[0].action if surviving else None
        self.shadow.note_decision(after, before)
        return surviving

    def _rules_for(self, state, tick: int) -> tuple[list[Rule], list[Rule]]:
        """Which rules may act this tick, and which are held back."""
        acting, inert = [], []
        for rule in self.lifecycle.ordered():
            if not rule.matches_state(state):
                continue
            if rule.status == ACTIVE:
                # Active rules keep a monitoring holdout for life, or nothing
                # could ever establish that they had stopped working.
                (acting if self.shadow.acts_while_active(rule, tick)
                 else inert).append(rule)
            elif rule.status == TRIAL:
                (acting if self.shadow.applies_this_tick(rule, tick)
                 else inert).append(rule)
        return acting, inert

    # ── the experience side ──────────────────────────────────────────

    def observe(self, state, action: str, reward: float, success: bool,
                tick: int = 0, experience_id: str = "") -> None:
        """Close one experience: update preferences, log it, score the trials.

        This is the whole of "result → evaluation → knowledge" as far as this
        contour is concerned; everything else it does is downstream of the rows
        recorded here.
        """
        if not action or state is None:
            return
        self.store.update(state, action, reward)

        state_key = state.key() if hasattr(state, "key") else str(state)
        row = {"tick": int(tick), "state": state_key, "action": str(action),
               "success": bool(success), "reward": round(float(reward), 5)}
        if experience_id:
            row["id"] = str(experience_id)
        self.experiences.append(row)
        if len(self.experiences) > MAX_EXPERIENCE_ROWS:
            self.experiences = self.experiences[-MAX_EXPERIENCE_ROWS:]

        for rule_key, acted in self._eligible_this_tick.items():
            self.shadow.record(rule_key, acted, reward, tick)
        self._eligible_this_tick = {}

    # ── mining and review ────────────────────────────────────────────

    def mine(self, tick: int = 0, safety_critical=()) -> list[Rule]:
        """Run one generation of the miner and admit what survives."""
        self.last_mined_tick = int(tick)
        candidates = self.miner.mine(self.experiences, tick)
        admitted = self.lifecycle.admit(candidates, tick,
                                        safety_critical=safety_critical)
        if admitted:
            logger.info("Policy admitted %d rule(s) to trial at tick %d",
                        len(admitted), tick)
        return admitted

    def review(self, tick: int = 0) -> dict:
        """Conclude finished trials and re-judge active rules (§M3.5)."""
        self.last_reviewed_tick = int(tick)
        outcomes = {ACTIVE: 0, RETIRED: 0, REFUTED: 0}

        for rule in self.lifecycle.on_trial():
            if not self.shadow.trial_finished(rule, tick):
                continue
            applied, withheld = self.shadow.arms(rule.id)
            verdict = self.lifecycle.conclude_trial(rule, applied, withheld, tick)
            outcomes[verdict] = outcomes.get(verdict, 0) + 1
            # Clear the arms either way. A later review has to re-judge the rule
            # on evidence gathered *since* it activated; carrying the trial's
            # samples forward would let the result that activated it keep
            # outvoting everything the world says afterwards.
            self.shadow.forget(rule.id)

        for rule in self.lifecycle.active():
            if tick - (rule.activated_tick or 0) < cfg.POLICY_REVIEW_TICKS:
                continue
            if not self.shadow.enough_to_review(rule.id):
                # Nothing to judge on yet. Postponing keeps the rule; retiring
                # it here would mean withdrawing a properly activated rule for
                # want of evidence, which is the opposite of what the review is
                # for. The window stays open, so the next call tries again.
                continue
            applied, withheld = self.shadow.arms(rule.id)
            verdict = self.lifecycle.review(rule, applied, withheld, tick)
            outcomes[verdict] = outcomes.get(verdict, 0) + 1
            self.shadow.forget(rule.id)
            if verdict == ACTIVE:
                # Restart the clock so a surviving rule is re-examined again,
                # rather than being re-judged on every tick from here on.
                rule.activated_tick = tick
        return outcomes

    # ── reading ──────────────────────────────────────────────────────

    def active_rules(self) -> list[Rule]:
        return self.lifecycle.active()

    def suppressed_actions(self, state) -> list[str]:
        """Which actions an active rule forbids in this state."""
        return sorted({rule.action for rule in self.lifecycle.active()
                       if rule.effect == SUPPRESS and rule.matches_state(state)})

    def behaviour_delta_rate(self) -> float:
        return self.shadow.observed_delta_rate()

    # ── persistence ──────────────────────────────────────────────────

    def _load_experiences(self) -> None:
        if not self._experience_path.exists():
            return
        rows = []
        try:
            with self._experience_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue          # a torn line costs its own row only
                    if isinstance(row, dict) and row.get("action"):
                        rows.append(row)
        except Exception:
            logger.warning("Failed to read the policy experience log",
                           exc_info=True)
            return
        self.experiences = rows[-MAX_EXPERIENCE_ROWS:]

    def save(self) -> None:
        self.store.save()
        self.lifecycle.save()
        self.shadow.flush()
        try:
            self._experience_path.parent.mkdir(parents=True, exist_ok=True)
            self._experience_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n"
                        for row in self.experiences),
                encoding="utf-8")
        except Exception:
            logger.warning("Failed to write the policy experience log",
                           exc_info=True)

    # ── reporting ────────────────────────────────────────────────────

    def publish_metrics(self, tick: int) -> None:
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        try:
            self.telemetry.record(M.POLICY_BEHAVIOUR_DELTA_RATE,
                                  self.shadow.observed_delta_rate(), tick)
            self.telemetry.record(M.POLICY_ACTIVE_RULES,
                                  len(self.lifecycle.active()), tick)
            self.telemetry.record(M.POLICY_REFUTED,
                                  self.lifecycle.refutations, tick)
            self.telemetry.record(M.POLICY_CANDIDATES,
                                  len(self.lifecycle.on_trial()), tick)
            if self.shadow.regret is not None:
                self.telemetry.record(M.POLICY_REGRET, self.shadow.regret, tick)
            else:
                self.telemetry.record(M.POLICY_REGRET, 0.0, tick)
        except Exception:
            logger.exception("Policy metric publication failed")

    def rules_report(self) -> list[dict]:
        """Every rule with the evidence behind it — the panel of §M10.1.

        All statuses, not only the active ones. A refuted rule is the record
        of something the system tried and was wrong about, and an operator
        looking at why behaviour changed needs to see what was rejected as
        much as what was kept.
        """
        return [rule.to_dict() for rule in self.lifecycle.ordered()]

    def effect_report(self) -> dict:
        """Did the policy change behaviour, and did the change pay (§M3.6)?"""
        return {
            "behaviour_delta_rate": self.behaviour_delta_rate(),
            "shadow": self.shadow.status(),
            "active_rules": len(self.lifecycle.active()),
            "suppressions": self.suppressions,
            "promotions": self.promotions,
            "by_status": {status: len(self.lifecycle.by_status(status))
                          for status in sorted(
                              {rule.status for rule in
                               self.lifecycle.ordered()})},
        }

    def status(self) -> dict:
        return {
            "preferences": self.store.status(),
            "miner": self.miner.status(),
            "rules": self.lifecycle.status(),
            "shadow": self.shadow.status(),
            "experiences": len(self.experiences),
            "suppressions": self.suppressions,
            "promotions": self.promotions,
            "last_mined_tick": self.last_mined_tick,
            "last_reviewed_tick": self.last_reviewed_tick,
        }
