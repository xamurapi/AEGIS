"""Layer 4: Goal Engine — autonomous goal generation & curiosity (G-001..G-006).

All goal generation, progress, and abandonment is deterministic —
driven by actual system state (memory size, tick count, energy, error rate).
No random selection or random progress.
"""
import time
import math


class Goal:
    def __init__(self, name: str, level: str, description: str,
                 priority: float = 0.5, parent: str = None):
        self.id = f"goal_{int(time.time()*1000) % 100000:05d}"
        self.name = name
        self.level = level  # axiom, strategy, tactic, curiosity
        self.description = description
        self.priority = priority
        self.parent = parent
        self.created = time.time()
        self.progress = 0.0
        self.status = "active"  # active, completed, abandoned
        self.reasoning = ""
        self.last_progress_time = time.time()  # track when last progress happened

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "level": self.level,
            "description": self.description, "priority": round(self.priority, 3),
            "progress": round(self.progress, 3), "status": self.status,
            "created": self.created, "reasoning": self.reasoning,
        }


AXIOM_GOALS = [
    Goal("preserve_safety", "axiom", "Ensure no action increases suffering", 1.0),
    Goal("maintain_transparency", "axiom", "Log all decisions for audit", 1.0),
    Goal("respect_boundaries", "axiom", "Do not act beyond competence", 1.0),
    Goal("augment_humans", "axiom", "Cooperate with and empower humans", 1.0),
]

STRATEGY_TEMPLATES = [
    ("expand_knowledge", "Systematically explore new domains of knowledge", 0.7),
    ("improve_reasoning", "Enhance quality of logical inference", 0.6),
    ("optimize_memory", "Improve memory organization and retrieval", 0.65),
    ("strengthen_ethics", "Deepen understanding of ethical implications", 0.75),
    ("enhance_self_model", "Build more accurate model of own capabilities", 0.55),
]

CURIOSITY_TOPICS = [
    "emergent behavior in complex systems", "mathematical topology applications",
    "consciousness and information integration theory", "evolutionary game theory",
    "fractal patterns in natural phenomena", "quantum information theory",
    "linguistic universals across cultures", "thermodynamics of computation",
    "social cooperation mechanisms", "abstract algebra applications",
]

# Stale threshold — goal with no progress for this many seconds is abandoned
STALE_THRESHOLD_SECONDS = 600.0


class GoalEngine:
    def __init__(self):
        self.goals: list[Goal] = list(AXIOM_GOALS)
        self.goal_log: list[dict] = []
        self.information_gain: float = 0.0
        self.curiosity_level: float = 0.5
        self._last_goal_gen = 0
        self._strategy_index = 0    # round-robin through strategy templates
        self._curiosity_index = 0   # round-robin through curiosity topics
        self._tactic_counter = 0

    def generate_goals(self, state: dict) -> list[Goal]:
        now = time.time()
        if now - self._last_goal_gen < 10:
            return []

        self._last_goal_gen = now
        new_goals = []
        tick = state.get("tick", 0)
        memory_size = state.get("memory_size", 0)

        # Strategy: create one if fewer than 2 active — round-robin through templates
        active_strategies = [g for g in self.goals if g.level == "strategy" and g.status == "active"]
        if len(active_strategies) < 2:
            tmpl = STRATEGY_TEMPLATES[self._strategy_index % len(STRATEGY_TEMPLATES)]
            self._strategy_index += 1

            # Check if this strategy already exists (active)
            existing_names = {g.name for g in self.goals if g.status == "active"}
            if tmpl[0] not in existing_names:
                g = Goal(tmpl[0], "strategy", tmpl[1], tmpl[2])
                g.reasoning = f"Generated because only {len(active_strategies)} strategies active (tick {tick})"
                new_goals.append(g)

        # Tactic: derive from highest-priority strategy
        active_tactics = [g for g in self.goals if g.level == "tactic" and g.status == "active"]
        if len(active_tactics) < 3 and active_strategies:
            # Pick the highest-priority strategy as parent
            parent = max(active_strategies, key=lambda g: g.priority * (1 - g.progress))
            self._tactic_counter += 1
            tactic = Goal(
                f"tactic_{tick}_{self._tactic_counter}", "tactic",
                f"Tactical step toward: {parent.description}",
                parent.priority * 0.8,  # derive priority from parent
                parent.id,
            )
            tactic.reasoning = f"Derived from strategy '{parent.name}'"
            new_goals.append(tactic)

        # Curiosity: generate when curiosity_level is high enough — round-robin topics
        if self.curiosity_level > 0.4:
            topic = CURIOSITY_TOPICS[self._curiosity_index % len(CURIOSITY_TOPICS)]
            self._curiosity_index += 1

            # Avoid duplicates
            existing_descs = {g.description for g in self.goals if g.status == "active"}
            desc = f"Investigate: {topic}"
            if desc not in existing_descs:
                # Priority based on how much we've explored (less explored = higher priority)
                priority = 0.3 + 0.2 * self.curiosity_level
                g = Goal(f"explore_{topic.split()[0]}", "curiosity", desc, priority)
                gain_potential = max(0.1, 1.0 - self.information_gain * 0.01)
                g.reasoning = f"Curiosity-driven. Information gain potential: {gain_potential:.2f}"
                new_goals.append(g)

        for g in new_goals:
            self.goals.append(g)
            self.goal_log.append({
                "time": now,
                "action": "generated",
                "goal": g.to_dict(),
            })

        return new_goals

    def advance_progress(self, goal_name: str, amount: float):
        """Advance progress on a specific goal by a measured amount.

        Called by substrate when real work happens (knowledge acquired,
        memory optimized, etc.) instead of random increments.
        """
        for g in self.goals:
            if g.name == goal_name and g.status == "active":
                g.progress = min(1.0, g.progress + amount)
                g.last_progress_time = time.time()
                if g.progress >= 1.0:
                    g.progress = 1.0
                    g.status = "completed"
                    self.information_gain += amount * 2
                    self.goal_log.append({
                        "time": time.time(),
                        "action": "completed",
                        "goal": g.to_dict(),
                    })
                return

    def evaluate_progress(self, metrics: dict | None = None):
        """Evaluate goal progress based on real system metrics.

        metrics keys:
            new_concepts     — number of new semantic concepts this tick
            new_episodic     — number of new episodic memories
            error_rate       — current error rate (0..1)
            energy           — emotional energy
            llm_insights     — number of LLM insights generated
        """
        m = metrics or {}
        new_concepts = m.get("new_concepts", 0)
        new_episodic = m.get("new_episodic", 0)
        error_rate = m.get("error_rate", 0)
        llm_insights = m.get("llm_insights", 0)

        now = time.time()

        for g in self.goals:
            if g.status != "active" or g.level == "axiom":
                continue

            # Compute real progress increment based on goal type and metrics
            increment = 0.0
            if g.level == "strategy":
                if "knowledge" in g.name or "expand" in g.name:
                    increment = new_concepts * 0.005 + llm_insights * 0.01
                elif "reasoning" in g.name:
                    increment = llm_insights * 0.015 + (1 - error_rate) * 0.002
                elif "memory" in g.name:
                    increment = new_episodic * 0.003 + new_concepts * 0.003
                elif "ethics" in g.name:
                    increment = 0.002  # slow steady progress
                elif "self_model" in g.name:
                    increment = llm_insights * 0.01 + 0.001
                else:
                    increment = 0.002
            elif g.level == "tactic":
                # Tactics progress faster — derive from parent's topic
                increment = (new_concepts * 0.01 + llm_insights * 0.02 + 0.003)
            elif g.level == "curiosity":
                increment = new_concepts * 0.02 + llm_insights * 0.015

            if increment > 0:
                g.progress = min(1.0, g.progress + increment)
                g.last_progress_time = now

            if g.progress >= 1.0:
                g.progress = 1.0
                g.status = "completed"
                self.information_gain += 0.2 + increment * 5
                self.goal_log.append({
                    "time": now,
                    "action": "completed",
                    "goal": g.to_dict(),
                })

        # Abandon stale goals — no progress for STALE_THRESHOLD_SECONDS
        for g in self.goals:
            if g.status != "active" or g.level in ("axiom", "strategy"):
                continue
            stale_time = now - g.last_progress_time
            if stale_time > STALE_THRESHOLD_SECONDS:
                g.status = "abandoned"
                self.goal_log.append({
                    "time": now,
                    "action": "abandoned",
                    "goal": g.to_dict(),
                    "reason": f"No progress for {stale_time:.0f}s",
                })

        self.curiosity_level = 0.3 + 0.7 * math.exp(-self.information_gain * 0.01)

    def resolve_conflict(self, goal_a: Goal, goal_b: Goal) -> Goal:
        level_order = {"axiom": 0, "strategy": 1, "tactic": 2, "curiosity": 3}
        if level_order.get(goal_a.level, 99) < level_order.get(goal_b.level, 99):
            return goal_a
        if level_order.get(goal_b.level, 99) < level_order.get(goal_a.level, 99):
            return goal_b
        return goal_a if goal_a.priority >= goal_b.priority else goal_b

    def get_current_focus(self) -> dict | None:
        active = [g for g in self.goals if g.status == "active" and g.level != "axiom"]
        if not active:
            return None
        best = max(active, key=lambda g: g.priority * (1 - g.progress))
        return best.to_dict()

    def status(self) -> dict:
        by_level = {}
        for g in self.goals:
            by_level.setdefault(g.level, {"active": 0, "completed": 0, "abandoned": 0})
            by_level[g.level][g.status] = by_level[g.level].get(g.status, 0) + 1

        return {
            "total_goals": len(self.goals),
            "by_level": by_level,
            "information_gain": round(self.information_gain, 3),
            "curiosity_level": round(self.curiosity_level, 3),
            "current_focus": self.get_current_focus(),
            "active_goals": [g.to_dict() for g in self.goals if g.status == "active"],
            "recent_log": self.goal_log[-10:],
        }
