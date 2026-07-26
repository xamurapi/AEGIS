"""Unit tests for the Substrate's pure/deterministic helper methods.

These exercise the reward/confidence/importance signals, checkpoint restore,
uptime formatting, full_status assembly and the stop() guard directly, without
running the async tick loop. Network/LLM/sandbox are neutralized so nothing
touches the real world.
"""
import asyncio
import json

from aegis.layers.substrate import Substrate


def _make_substrate():
    s = Substrate()

    async def _noop_agents():
        return []

    async def _noop_learn(*a, **k):
        return {"success": False}

    s.agent_system.run_due_agents = _noop_agents
    s.external_learning.learn_from_source = _noop_learn
    s.llm.enabled = False
    s.environment.step = lambda: {"reward": 0.0, "solved": False, "task": None}
    # Pin health to "healthy": HealthMonitor.check() reads REAL cpu/memory via
    # psutil, so under full-suite load it reports "critical", MetaRegulation
    # switches to emergency mode and the tick legitimately skips learning /
    # ethics blocks self-modification — a machine-load dependency, not a defect.
    s.health.check = lambda: {"status": "healthy", "warnings": [],
                              "critical": [], "metrics": {}}
    return s


def test_compute_reward_in_unit_range():
    s = _make_substrate()
    r = s._compute_reward()
    assert 0.0 <= r <= 1.0


def test_compute_reward_uses_benchmark_when_available():
    s = _make_substrate()
    s._last_benchmark_score = 0.8
    # environment has no steps -> reward = 0.7*bench + 0.3*env(0) = 0.56
    s.environment.rolling_reward = lambda: 0.0
    r = s._compute_reward()
    assert abs(r - 0.56) < 1e-6


def test_compute_reward_fallback_before_first_eval():
    s = _make_substrate()
    s._last_benchmark_score = None
    s.environment.total_steps = 0
    r = s._compute_reward()  # legacy synthetic estimate branch
    assert 0.0 <= r <= 1.0


def test_compute_confidence_bounds():
    s = _make_substrate()
    c = s._compute_confidence()
    assert 0.1 <= c <= 0.99


def test_compute_importance_bounds_and_signals():
    s = _make_substrate()
    base = s._compute_importance()
    assert 0.1 <= base <= 1.0
    # An LLM insight this tick raises importance.
    s._tick_llm_insights = 1
    assert s._compute_importance() > base


def test_format_uptime():
    assert Substrate._format_uptime(0) == "00:00:00"
    assert Substrate._format_uptime(3661) == "01:01:01"
    assert Substrate._format_uptime(59) == "00:00:59"


def test_is_llm_tick_respects_disabled_llm():
    s = _make_substrate()
    s.llm.enabled = False
    assert s._is_llm_tick() is False


def test_full_status_has_all_sections():
    s = _make_substrate()
    st = s.full_status()
    for key in ("substrate", "emotions", "memory", "goals", "ethics",
                "world_model", "cognitive_graph", "evolution",
                "goal_intelligence", "feedback_loop", "reward_signal"):
        assert key in st


def test_save_and_restore_checkpoint_round_trip(tmp_path):
    s = _make_substrate()
    s._checkpoint_path = tmp_path / "latest.json"
    s.tick_count = 123
    s.self_mod.current_version = "1.2.3"
    s._save_checkpoint()
    assert s._checkpoint_path.exists()

    s2 = _make_substrate()
    s2._checkpoint_path = tmp_path / "latest.json"
    s2._restore_checkpoint()
    assert s2.tick_count == 123
    assert s2.self_mod.current_version == "1.2.3"


def test_restore_checkpoint_survives_corrupt_file(tmp_path):
    s = _make_substrate()
    s._checkpoint_path = tmp_path / "latest.json"
    s._checkpoint_path.write_text("{ broken json", encoding="utf-8")
    s.tick_count = 7
    s._restore_checkpoint()  # must not raise; leaves state as-is
    assert s.tick_count == 7


def test_checkpoint_write_is_atomic_and_valid_json(tmp_path):
    s = _make_substrate()
    s._checkpoint_path = tmp_path / "latest.json"
    s.tick_count = 5
    s._save_checkpoint()
    data = json.loads(s._checkpoint_path.read_text(encoding="utf-8"))
    assert data["tick_count"] == 5
    # No leftover temp file.
    assert not (tmp_path / "latest.json.tmp").exists()


def test_stop_blocked_for_unauthorized_reason():
    s = _make_substrate()
    # A non-allowlisted reason must be refused by self-preservation.
    ok = s.stop(reason="random_watchdog")
    assert ok is False
    assert s.running is False  # never started, stays stopped


def test_stop_allowed_for_human_command(tmp_path):
    s = _make_substrate()
    s._checkpoint_path = tmp_path / "latest.json"
    s.running = True
    ok = s.stop(reason="human_command")
    assert ok is True
    assert s.running is False


def test_single_tick_updates_counters_and_health():
    async def run():
        s = _make_substrate()
        before = s.tick_count
        await s.tick()
        assert s.tick_count == before + 1
        assert s.last_tick_duration >= 0.0
        # llm_thinking must be cleared after the cycle.
        assert s.llm_thinking is False
    asyncio.run(run())
