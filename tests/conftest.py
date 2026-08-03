"""Shared test configuration.

Socket-exhaustion fix (Windows WinError 10055): the async tests call
``asyncio.run(...)`` per test. On Windows each ``asyncio.run`` builds a fresh
event loop, and every loop opens a self-pipe socket pair; across ~900 async
tests those sockets pile up in TIME_WAIT and the OS runs out of socket buffer
space ("An operation on a socket could not be performed because the system
lacked sufficient buffer space").

Fix: reuse ONE event loop for the whole test session (one self-pipe, reused)
and cancel any leftover tasks after each run so nothing leaks between tests.
This frees the sockets and makes the full suite run green in a single pass.
"""
import sys
import asyncio
from pathlib import Path

import pytest

# Prefer the Selector loop on Windows — it is lighter than the Proactor loop for
# the short, socket-free coroutines these tests run.
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

_SESSION_LOOP: asyncio.AbstractEventLoop | None = None


def _session_loop() -> asyncio.AbstractEventLoop:
    global _SESSION_LOOP
    if _SESSION_LOOP is None or _SESSION_LOOP.is_closed():
        _SESSION_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_SESSION_LOOP)
    return _SESSION_LOOP


def _shared_run(coro):
    """Drop-in replacement for asyncio.run that reuses the session loop and
    cleans up leftover tasks afterwards (so one test's detached tasks cannot
    leak into the next)."""
    loop = _session_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


# Route every asyncio.run(...) in the test session through the shared loop.
asyncio.run = _shared_run


def pytest_sessionfinish(session, exitstatus):
    """Close the single session loop at the very end, releasing its socketpair."""
    global _SESSION_LOOP
    if _SESSION_LOOP is not None and not _SESSION_LOOP.is_closed():
        _SESSION_LOOP.close()
        _SESSION_LOOP = None


# ── isolated persistent state ────────────────────────────────────────
# Every store reads its directory from a module-level constant imported at
# import time, so redirecting persistence means patching those constants — not
# aegis.config, which was already read. Without this a Substrate built in a test
# loads whatever the last run left on disk, which makes "two identical runs"
# impossible to write and lets tests contaminate each other.

_STATE_DIRS = (
    ("aegis.layers.memory", "MEMORY_DIR", "memory"),
    ("aegis.layers.world_model", "WORLD_MODEL_DIR", "world_model"),
    ("aegis.layers.cognitive_graph", "COGNITIVE_GRAPH_DIR", "cognitive_graph"),
    ("aegis.layers.evolution_engine", "EVOLUTION_DIR", "evolution"),
    ("aegis.layers.goal_intelligence", "GOAL_INTEL_DIR", "goal_intelligence"),
    ("aegis.layers.feedback_loop", "FEEDBACK_DIR", "feedback"),
    ("aegis.layers.substrate", "CHECKPOINTS_DIR", "checkpoints"),
    ("aegis.layers.substrate", "EVAL_DIR", "eval"),
    ("aegis.layers.substrate", "CODE_BACKUPS_DIR", "code_backups"),
    ("aegis.layers.dataset_builder", "WEIGHT_DATASETS_DIR", "datasets"),
    ("aegis.telemetry.store", "TELEMETRY_DIR", "telemetry"),
    # The contours added by the development spec read their directory from
    # `aegis.config` at construction time rather than binding a module-level
    # constant, so the config module is what has to be redirected for them.
    ("aegis.config", "MOTIVATION_DIR", "motivation"),
    ("aegis.config", "POLICY_DIR", "policy"),
    ("aegis.config", "REASONING_DIR", "reasoning"),
    ("aegis.config", "DISCOVERY_DIR", "discovery"),
    ("aegis.config", "META_DIR", "metacognition"),
    ("aegis.config", "CORTEX_DIR", "cortex"),
    ("aegis.config", "WORLD_MODEL_DIR", "world_model"),
    ("aegis.config", "EVOLUTION_DIR", "evolution"),
)


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Point every persistent store at a private directory.

    Yields the root path so a test can inspect what was written.
    """
    import importlib

    root = tmp_path / "state"
    for module_name, constant, subdir in _STATE_DIRS:
        module = importlib.import_module(module_name)
        if not hasattr(module, constant):
            continue
        target = root / subdir
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(module, constant, target, raising=False)

    # LLM lifetime stats and state backups take their paths at construction.
    import aegis.llm as llm_mod
    monkeypatch.setattr(llm_mod, "TOKEN_STATS_FILE", root / "token_stats.json",
                        raising=False)

    import aegis.layers.substrate as substrate_mod
    from aegis.layers.state_backup import StateBackup
    backup_dir = root / "backups"
    monkeypatch.setattr(substrate_mod, "StateBackup",
                        lambda *a, **k: StateBackup(backup_dir=backup_dir),
                        raising=False)

    yield root


# ── evolution stays off unless a test asks for it ────────────────────
# A generation evaluates ten variants, each building a fresh system in another
# process to run a benchmark and a rollout in it. Any test that ticks past
# `EVO_EVERY_N_TICKS` would otherwise start one — and a suite that runs real
# generations measures the machine rather than the code. Measured: 137 python
# processes and a suite that stopped finishing.
#
# The tests that are *about* evolution drive `run_generation` directly, so they
# are unaffected; `test_evolution_is_off_in_the_suite` keeps this honest.

@pytest.fixture(autouse=True)
def _evolution_off(monkeypatch):
    import aegis.config as cfg

    monkeypatch.setattr(cfg, "EVO_ENABLED", False)
