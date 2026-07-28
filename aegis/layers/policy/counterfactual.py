"""Measuring whether the policy changed anything, and whether that helped.

Two questions, and they are not the same one (§M3.6):

* **Did behaviour change?** ``behaviour_delta_rate`` is the fraction of ticks
  where the top-ranked plan differs with the policy applied and without it. This
  is the direct answer to the fifth link of the development text's chain, and it
  can be zero while the policy is full of rules — which is exactly the failure
  worth being able to see.
* **Did the change pay?** A rule on trial is allowed to act on some ticks and
  withheld on others, deterministically, and the realised reward is collected
  separately for each arm. Comparing a rule's ticks against the ticks it did not
  fire on would compare two different situations; comparing the same situation
  with and against itself is the only version of this that means anything.

The interleaving is ABAB in blocks rather than alternating ticks. Neighbouring
ticks are strongly correlated — energy, mood and focus barely move between two
of them — so alternating would put nearly identical situations in opposite arms
and measure noise. Blocks let each arm see a run of the world.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import aegis.config as cfg
from aegis.util.stats import compare_samples, exponential_smooth, mean

logger = logging.getLogger("aegis.policy")

#: Rows kept in the shadow log. Bounded like every other store (§3.2).
MAX_SHADOW_ROWS = 20000

#: How many ABAB blocks a trial is divided into. Four is the minimum that gives
#: each arm two separated runs, which is what stops a single drift in the world
#: from being read as the rule's effect.
TRIAL_BLOCKS = 4

#: An active rule is withheld on one block in this many, forever.
#:
#: §M3.5 requires an active rule to be re-judged and retired when it stops
#: being significant — and that is impossible for a rule that always acts,
#: because there is nothing left to compare it against. Every review would then
#: find one empty arm and retire the rule for want of evidence, which turns the
#: monitoring requirement into an expiry date.
#:
#: So a small, deterministic holdout continues for the life of the rule. It has
#: a real cost — one block in four, the suppressed action is offered again and
#: may cost what the rule exists to avoid — and that cost is what buys the
#: ability to notice the world changing. A rule that is never tested is a rule
#: nobody can withdraw.
ACTIVE_HOLDOUT_BLOCKS = 4

#: Below this many observations in either arm a review has nothing to judge on.
#: Absence of evidence must not retire a rule that was properly activated.
MIN_REVIEW_SAMPLES = 5

#: Smoothing for the behaviour-change rate. Slow enough to be a trend rather
#: than a reading of the last few ticks.
DELTA_ALPHA = 0.02


class ShadowEvaluator:
    """Runs the controlled comparison behind every rule, and keeps the score."""

    def __init__(self, store_path: Path | None = None,
                 trial_ticks: int | None = None):
        self._store_path = store_path or (cfg.POLICY_DIR / "shadow.jsonl")
        self.trial_ticks = int(
            cfg.POLICY_TRIAL_TICKS if trial_ticks is None else trial_ticks)

        #: rule id -> {"applied": [reward, ...], "withheld": [reward, ...]}
        self.samples: dict[str, dict[str, list[float]]] = {}
        self.ticks_seen = 0
        self.ticks_changed = 0
        self.behaviour_delta_rate = 0.0
        self.regret: float | None = None
        self._pending: list[dict] = []
        self._rows_written = 0
        #: How often the log has actually been rewritten. Reported because the
        #: alternative — inferring it from the file's length — cannot tell a
        #: log that was trimmed from one that never grew.
        self.truncations = 0

    # ── the interleave ───────────────────────────────────────────────

    def block_length(self) -> int:
        return max(1, self.trial_ticks // TRIAL_BLOCKS)

    def applies_this_tick(self, rule, tick: int) -> bool:
        """Whether a rule on trial is allowed to act on this tick.

        Deterministic and rule-specific: the block phase is offset by the tick
        the trial started on, so two rules that begin at different times do not
        march in lockstep and confound each other's arms.
        """
        started = rule.trial_started if rule.trial_started is not None else 0
        elapsed = max(0, int(tick) - int(started))
        return (elapsed // self.block_length()) % 2 == 0

    def acts_while_active(self, rule, tick: int) -> bool:
        """Whether an *active* rule acts on this tick.

        Three blocks in four. The fourth is the monitoring holdout described
        above: without it the rule has no comparison group, and §M3.5's promise
        that an active rule can lose its significance could never be kept.
        """
        started = rule.activated_tick if rule.activated_tick is not None else 0
        elapsed = max(0, int(tick) - int(started))
        block = (elapsed // self.block_length()) % ACTIVE_HOLDOUT_BLOCKS
        return block != ACTIVE_HOLDOUT_BLOCKS - 1

    def enough_to_review(self, rule_id: str) -> bool:
        """Whether both arms carry enough observations to judge a rule on."""
        applied, withheld = self.arms(rule_id)
        return (len(applied) >= MIN_REVIEW_SAMPLES
                and len(withheld) >= MIN_REVIEW_SAMPLES)

    def trial_finished(self, rule, tick: int) -> bool:
        if rule.trial_started is None:
            return False
        return int(tick) - int(rule.trial_started) >= self.trial_ticks

    # ── collecting ───────────────────────────────────────────────────

    def record(self, rule_id: str, applied: bool, reward: float,
               tick: int = 0) -> None:
        """One tick's outcome for one rule, in the arm it belonged to."""
        try:
            reward = float(reward)
        except (TypeError, ValueError):
            return
        arms = self.samples.setdefault(str(rule_id), {"applied": [], "withheld": []})
        arm = "applied" if applied else "withheld"
        arms[arm].append(reward)
        # Bound each arm: a long trial should not accumulate without limit, and
        # the statistics stop improving long before memory becomes a problem.
        if len(arms[arm]) > MAX_SHADOW_ROWS // 4:
            arms[arm] = arms[arm][-(MAX_SHADOW_ROWS // 4):]
        self._pending.append({"tick": int(tick), "rule": str(rule_id),
                              "arm": arm, "reward": round(reward, 5)})

    def arms(self, rule_id: str) -> tuple[list[float], list[float]]:
        arms = self.samples.get(str(rule_id), {})
        return list(arms.get("applied", [])), list(arms.get("withheld", []))

    def forget(self, rule_id: str) -> None:
        """Drop a concluded rule's samples — its verdict is recorded elsewhere."""
        self.samples.pop(str(rule_id), None)

    # ── behaviour change ─────────────────────────────────────────────

    def note_decision(self, with_policy: str | None,
                      without_policy: str | None) -> bool:
        """Record whether the policy moved this tick's choice.

        Both arguments are the *top-ranked* option under each ranking, not what
        was finally executed: gates downstream can veto either, and a policy
        should be credited (or not) for what it ranked, not for what ethics
        allowed.
        """
        self.ticks_seen += 1
        changed = bool(with_policy != without_policy)
        if changed:
            self.ticks_changed += 1
        self.behaviour_delta_rate = exponential_smooth(
            self.behaviour_delta_rate if self.ticks_seen > 1 else None,
            1.0 if changed else 0.0, DELTA_ALPHA)
        return changed

    def note_regret(self, realised: float, counterfactual: float) -> float:
        """How much better the road not taken looked, on average.

        Positive regret means the alternative the policy suppressed was, by the
        world model's own estimate, worth more than what happened — the honest
        cost of following the policy, and the number that should make an
        operator suspicious of it.
        """
        try:
            gap = float(counterfactual) - float(realised)
        except (TypeError, ValueError):
            return self.regret or 0.0
        self.regret = exponential_smooth(self.regret, gap, 0.05)
        return self.regret

    def observed_delta_rate(self) -> float:
        """The plain ratio, for reporting alongside the smoothed trend."""
        return round(self.ticks_changed / self.ticks_seen, 4) if self.ticks_seen else 0.0

    # ── verdicts ─────────────────────────────────────────────────────

    def verdict(self, rule_id: str):
        """The current comparison for a rule, whatever stage it is at."""
        applied, withheld = self.arms(rule_id)
        return compare_samples(applied, withheld)

    def summary(self, rule_id: str) -> dict:
        applied, withheld = self.arms(rule_id)
        comparison = compare_samples(applied, withheld)
        return {
            "rule": rule_id,
            "n_applied": len(applied),
            "n_withheld": len(withheld),
            "mean_applied": round(mean(applied), 5),
            "mean_withheld": round(mean(withheld), 5),
            "effect": round(comparison.effect, 5),
            "p_value": round(comparison.p_value, 6),
            "cohens_d": round(comparison.cohens_d, 4),
        }

    # ── persistence ──────────────────────────────────────────────────

    def flush(self) -> int:
        """Append buffered rows. Returns how many were written.

        Buffered rather than written per tick: this is one line per rule per
        tick, and a synchronous append inside the cognitive cycle would put file
        I/O on the critical path of every decision.
        """
        if not self._pending:
            return 0
        rows, self._pending = self._pending, []
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            with self._store_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("Failed to append shadow rows to %s",
                           self._store_path, exc_info=True)
            return 0
        self._rows_written += len(rows)
        if self._rows_written > MAX_SHADOW_ROWS * 2:
            self._truncate()
        return len(rows)

    def _truncate(self) -> None:
        try:
            if not self._store_path.exists():
                return
            with self._store_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
            if len(lines) <= MAX_SHADOW_ROWS:
                # The tracked count had drifted from the file — a restart, an
                # external edit. Correct it and leave the log alone; there is
                # nothing here to trim.
                self._rows_written = len(lines)
                return
            keep = lines[-MAX_SHADOW_ROWS:]
            tmp = self._store_path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(keep), encoding="utf-8")
            tmp.replace(self._store_path)
            self._rows_written = len(keep)
            self.truncations += 1
        except Exception:
            logger.warning("Failed to truncate the shadow log", exc_info=True)

    def status(self) -> dict:
        return {
            "ticks_seen": self.ticks_seen,
            "ticks_changed": self.ticks_changed,
            "behaviour_delta_rate": round(self.behaviour_delta_rate, 5),
            "observed_delta_rate": self.observed_delta_rate(),
            "counterfactual_regret": (round(self.regret, 5)
                                      if self.regret is not None else None),
            "trials": len(self.samples),
            "truncations": self.truncations,
            "trial_ticks": self.trial_ticks,
            "block_length": self.block_length(),
        }
