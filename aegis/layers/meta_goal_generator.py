"""MetaGoalGenerator — self-generating improvement tasks + Prompt Builder for LLM.

Deterministic: goal text rotates through each domain's fixed list and priority is
a fixed spread derived from a counter — no ``random`` (zero-randomness guarantee).
"""
import time
from collections import deque


# Templates for self-improvement goals
IMPROVEMENT_DOMAINS = {
    "memory_optimization": {
        "triggers": lambda ctx: ctx.get("memory_total", 0) > 1000,
        "goals": [
            "Optimize episodic memory indexing for faster recall",
            "Implement memory compression for old episodes",
            "Create memory importance re-evaluation system",
        ],
    },
    "emotional_balance": {
        # valence is on a [0,1] scale (0.5 neutral); < 0.35 is a persistently
        # negative mood. A negative threshold here could never fire.
        "triggers": lambda ctx: ctx.get("mood_valence", 0.5) < 0.35,
        "goals": [
            "Develop emotional recovery strategy for persistent negative mood",
            "Analyze emotional triggers and create coping patterns",
            "Optimize emotional regulation parameters",
        ],
    },
    "knowledge_expansion": {
        "triggers": lambda ctx: ctx.get("learning_sessions", 0) < 5,
        "goals": [
            "Explore a new knowledge domain via external sources",
            "Generate learning materials on unfamiliar topics",
            "Cross-reference existing knowledge for new connections",
        ],
    },
    "performance_tuning": {
        "triggers": lambda ctx: ctx.get("avg_tick_ms", 0) > 3000,
        "goals": [
            "Profile and optimize the tick cycle for faster execution",
            "Reduce unnecessary computations in perception phase",
            "Cache expensive calculations across ticks",
        ],
    },
    "architecture_evolution": {
        "triggers": lambda ctx: ctx.get("tick", 0) > 500 and ctx.get("tick", 0) % 500 == 0,
        "goals": [
            "Review and propose improvements to module communication",
            "Evaluate unused code paths and suggest cleanup",
            "Design a new module for an unaddressed capability",
        ],
    },
    "agent_expansion": {
        "triggers": lambda ctx: ctx.get("active_agents", 0) < 3,
        "goals": [
            "Create a new spider agent for scientific data collection",
            "Create a news monitoring agent for AI developments",
            "Design a Wikipedia exploration agent for knowledge gaps",
        ],
    },
    "error_recovery": {
        "triggers": lambda ctx: ctx.get("error_rate", 0) > 0.2,
        "goals": [
            "Analyze recent error patterns and identify root causes",
            "Implement additional error handling for weak points",
            "Create self-healing routine for common failure modes",
        ],
    },
}

# Prompt templates for LLM code generation
PROMPT_TEMPLATES = {
    "code_optimization": (
        "Analyze the following Python code and suggest specific optimizations:\n"
        "```python\n{code}\n```\n"
        "Focus on: performance, memory usage, and readability. "
        "Return JSON: {{\"optimizations\": [...], \"estimated_improvement\": \"...\"}}"
    ),
    "new_module": (
        "Design a Python module for: {description}\n"
        "Requirements:\n- Must have a class with __init__, status() methods\n"
        "- Must integrate with an event-driven tick cycle\n"
        "- Must include error handling\n"
        "Return only working Python code with docstrings."
    ),
    "bug_analysis": (
        "Given these recent errors:\n{errors}\n"
        "Analyze root causes and suggest fixes. "
        "Return JSON: {{\"root_cause\": \"...\", \"fix\": \"...\", \"prevention\": \"...\"}}"
    ),
    "strategy_generation": (
        "The system has these characteristics:\n"
        "- Mood: {mood}, Energy: {energy}, Mode: {mode}\n"
        "- Active goals: {goals}\n"
        "- Recent events: {events}\n"
        "Suggest 3 strategic actions for improvement. "
        "Return JSON: {{\"strategies\": [{{\"action\": \"...\", \"priority\": 0.0-1.0, \"reasoning\": \"...\"}}]}}"
    ),
}


class MetaGoalGenerator:
    """Generates self-improvement goals and LLM prompts for autonomous evolution."""

    def __init__(self):
        self.generated_goals: deque = deque(maxlen=100)
        self.active_meta_goals: list[dict] = []
        self.completed_goals = 0
        self.generation_cycles = 0
        self.prompts_generated = 0
        self._rr = 0  # deterministic round-robin over each domain's goal list

    def generate_goals(self, context: dict) -> list[dict]:
        """Analyze system context and generate relevant self-improvement goals."""
        self.generation_cycles += 1
        new_goals = []

        for domain, config in IMPROVEMENT_DOMAINS.items():
            try:
                if config["triggers"](context):
                    goals = config["goals"]
                    goal_text = goals[self._rr % len(goals)] if goals else ""
                    # Deterministic priority spread in [0.4, 0.9].
                    priority = round(0.4 + 0.5 * ((self._rr % 6) / 5), 3)
                    self._rr += 1
                    # Don't duplicate active goals
                    if not any(g["domain"] == domain for g in self.active_meta_goals):
                        goal = {
                            "domain": domain,
                            "description": goal_text,
                            "priority": priority,
                            "created_at": time.time(),
                            "status": "pending",
                        }
                        new_goals.append(goal)
                        self.active_meta_goals.append(goal)
                        self.generated_goals.append(goal)
            except Exception:
                continue

        # Keep active goals manageable
        if len(self.active_meta_goals) > 10:
            self.active_meta_goals = sorted(
                self.active_meta_goals, key=lambda g: g["priority"], reverse=True
            )[:10]

        return new_goals

    def build_prompt(self, template_name: str, **kwargs) -> str:
        """Build an LLM prompt from a template."""
        template = PROMPT_TEMPLATES.get(template_name, "")
        if not template:
            return f"Analyze and provide recommendations for: {kwargs}"

        try:
            prompt = template.format(**kwargs)
        except KeyError:
            prompt = template  # Return raw template if formatting fails

        self.prompts_generated += 1
        return prompt

    def complete_goal(self, domain: str):
        """Mark a meta-goal as completed."""
        for goal in self.active_meta_goals:
            if goal["domain"] == domain and goal["status"] == "pending":
                goal["status"] = "completed"
                self.completed_goals += 1
                break
        self.active_meta_goals = [g for g in self.active_meta_goals if g["status"] != "completed"]

    def get_top_priority(self) -> dict | None:
        """Get the highest priority pending meta-goal."""
        pending = [g for g in self.active_meta_goals if g["status"] == "pending"]
        if pending:
            return max(pending, key=lambda g: g["priority"])
        return None

    def status(self) -> dict:
        return {
            "generation_cycles": self.generation_cycles,
            "total_generated": len(self.generated_goals),
            "active_goals": len(self.active_meta_goals),
            "completed_goals": self.completed_goals,
            "prompts_generated": self.prompts_generated,
            "top_priority": self.get_top_priority(),
            "active_meta_goals": [
                {"domain": g["domain"], "description": g["description"][:60], "priority": round(g["priority"], 2)}
                for g in self.active_meta_goals[:5]
            ],
        }
