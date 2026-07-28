"""The evaluation pool (spec M9.1).

Four promises, and each one is a defect somewhere else if it is broken:

* results come back in **submission order**, or evolution's selection depends on
  machine load and two runs of the same experiment disagree;
* a hung worker is **killed**, or one non-terminating variant stops a generation
  forever;
* no subprocess slot is taken **without a lease**, or the budget is decorative;
* shutdown **does not wait**, or stopping the system hangs on a benchmark.

Most tests here run through the inline path deliberately: spawning a process per
assertion on Windows costs about a second and would buy nothing, because the
ordering, budgeting and error handling are the same code either way. The last
two do use real processes, and they are not marked slow — "it runs elsewhere"
and "a hung worker is killed" are exactly the claims that cannot be checked in
this process, so skipping them by default would leave the pool's two hardest
guarantees unverified.
"""
import os
import time

import pytest

from aegis.eval.pool import EvaluationPool, PoolResult, RESERVED_CORES, default_workers
from aegis.layers.motivation.resources import ResourceCost, ResourceManager


# ── module-level so a child process can import them ──────────────────

def square(value):
    return value * value


def explode(value):
    raise ValueError(f"no: {value}")


def slow(value):
    time.sleep(value)
    return value


def process_id(value):
    return os.getpid()


@pytest.fixture
def pool():
    return EvaluationPool(workers=2, task_timeout=5.0)


# ── ordering ─────────────────────────────────────────────────────────

def test_results_come_back_in_submission_order(pool):
    results = pool.map(square, [5, 1, 4, 2, 3])
    assert [r.index for r in results] == [0, 1, 2, 3, 4]
    assert [r.value for r in results] == [25, 1, 16, 4, 9]


def test_an_empty_batch_is_nothing(pool):
    assert pool.map(square, []) == []
    assert pool.stats.batches == 0


def test_a_failing_item_costs_itself_and_nothing_else(pool):
    results = pool.map(explode, [1])
    assert not results[0].ok
    assert "ValueError" in results[0].error
    assert pool.stats.failed == 1


def test_one_failure_does_not_stop_the_batch():
    calls = []

    def sometimes(value):
        calls.append(value)
        if value == 2:
            raise RuntimeError("boom")
        return value

    pool = EvaluationPool(workers=1, task_timeout=5.0)
    pool._ensure_executor = lambda: None          # force the inline path
    results = pool.map(sometimes, [1, 2, 3])
    assert [r.ok for r in results] == [True, False, True]
    assert calls == [1, 2, 3]


def test_a_result_reports_its_own_shape():
    result = PoolResult(index=3, value=9, latency_ms=12.5)
    assert result.ok
    assert result.as_dict() == {"index": 3, "ok": True, "error": "",
                                "timed_out": False, "latency_ms": 12.5}
    assert not PoolResult(index=0, error="x").ok
    assert not PoolResult(index=0, timed_out=True).ok


# ── the budget ───────────────────────────────────────────────────────

def test_a_refused_lease_runs_the_work_here_instead(tmp_path):
    """Deferring a benchmark forever is not a safe degradation. The work still
    happens — it just happens where its cost is visible."""
    resources = ResourceManager(store_path=tmp_path / "budgets.json")
    resources.budgets["subprocess_slots"].limit = 0
    pool = EvaluationPool(workers=2, resources=resources)

    results = pool.map(square, [2, 3])
    assert [r.value for r in results] == [4, 9]
    assert pool.stats.inline == 2
    assert pool.status()["inline"] == 2


def test_a_granted_lease_is_settled(tmp_path):
    resources = ResourceManager(store_path=tmp_path / "budgets.json")
    pool = EvaluationPool(workers=2, resources=resources)
    pool._ensure_executor = lambda: None
    pool.map(square, [1, 2])
    # Held slots go back: a pool that leaked them would strangle the next batch.
    assert resources.budgets["subprocess_slots"].held == 0
    assert resources.open_leases() == []


def test_a_broken_resource_manager_does_not_stop_the_work():
    class _Boom:
        def reserve(self, *a, **k):
            raise RuntimeError("resource manager down")

    pool = EvaluationPool(workers=1, resources=_Boom())
    pool._ensure_executor = lambda: None
    results = pool.map(square, [4])
    assert results[0].value == 16


def test_no_resource_manager_means_no_reservation(pool):
    assert pool._reserve(2, "test") is None
    pool._settle(None)              # must not raise


# ── worker count ─────────────────────────────────────────────────────

def test_the_machine_keeps_two_cores(monkeypatch):
    """`min(configured, cores − 2)`, and never below one.

    The reservation is what keeps the tick budget meetable: a pool that
    saturates every core makes §3.4 unmeetable, which is the thing the pool
    exists to protect.
    """
    import aegis.eval.pool as module

    monkeypatch.setattr(module.cfg, "EVAL_POOL_WORKERS", 16)
    monkeypatch.setattr(module.os, "cpu_count", lambda: 12)
    assert default_workers() == 10                 # 12 − 2, under the cap of 16

    monkeypatch.setattr(module.cfg, "EVAL_POOL_WORKERS", 3)
    assert default_workers() == 3                  # the cap binds instead

    monkeypatch.setattr(module.os, "cpu_count", lambda: 2)
    assert default_workers() == 1                  # never zero or negative

    monkeypatch.setattr(module.os, "cpu_count", lambda: None)
    assert default_workers() == 1                  # an unknown machine
    assert RESERVED_CORES == 2


def test_a_worker_count_is_never_zero():
    assert EvaluationPool(workers=0).workers == 1
    assert EvaluationPool(workers=-5).workers == 1


# ── failure of the pool itself ───────────────────────────────────────

def test_a_pool_that_cannot_start_falls_back_inline(monkeypatch):
    import aegis.eval.pool as module

    def _refuse(*a, **k):
        raise OSError("no processes available")

    monkeypatch.setattr(module, "ProcessPoolExecutor", _refuse)
    pool = EvaluationPool(workers=2)
    results = pool.map(square, [3, 4])
    assert [r.value for r in results] == [9, 16]
    assert pool.stats.inline == 2


def test_a_pool_that_dies_on_submit_falls_back_inline(monkeypatch):
    class _Dying:
        def submit(self, *a, **k):
            raise RuntimeError("pool is broken")

        def shutdown(self, **k):
            pass

    pool = EvaluationPool(workers=2)
    pool._executor = _Dying()
    pool._ensure_executor = lambda: pool._executor
    results = pool.map(square, [3])
    assert results[0].value == 9
    assert pool.stats.inline == 1


def test_shutdown_is_safe_when_nothing_is_running(pool):
    pool.shutdown()
    pool.shutdown()
    assert pool.status()["running"] is False


def test_a_shutdown_that_raises_is_absorbed():
    class _Awkward:
        def shutdown(self, **k):
            raise RuntimeError("will not close")

    pool = EvaluationPool(workers=1)
    pool._executor = _Awkward()
    pool.shutdown()                 # swallowed, logged
    assert pool._executor is None


def test_status_reports_the_batch(pool):
    pool._ensure_executor = lambda: None
    pool.map(square, [1, 2, 3])
    status = pool.status()
    assert status["submitted"] == 3 and status["completed"] == 3
    assert status["batches"] == 1
    assert status["workers"] == 2


# ── real processes ───────────────────────────────────────────────────

def test_work_really_runs_in_another_process():
    pool = EvaluationPool(workers=2, task_timeout=60.0)
    try:
        results = pool.map(process_id, [None, None])
        assert all(r.ok for r in results)
        assert all(r.value != os.getpid() for r in results)
    finally:
        pool.shutdown()


def test_a_hung_worker_is_killed_and_the_batch_continues():
    """A timeout has to be a limit, not a suggestion: the future is still
    running in its worker, and killing the pool is the only way to reclaim it."""
    pool = EvaluationPool(workers=1, task_timeout=1.0)
    try:
        results = pool.map(slow, [30, 0])
        assert results[0].timed_out
        assert "exceeded" in results[0].error
        assert results[1].ok and results[1].value == 0
        assert pool.stats.kills >= 1
    finally:
        pool.shutdown()


# ── the pooled path, driven by a fake executor ───────────────────────
# Real processes prove that the pool *works*; they are far too slow to prove
# that it works in every ordering. A controllable executor exercises the same
# `_run_pooled` code with futures resolved by hand, which is where the indexing
# and restart arithmetic actually lives.

class _Controlled:
    """An executor whose futures the test resolves."""

    def __init__(self, outcomes):
        # outcome per submission: ("value", v) | ("raise", exc) | ("hang",)
        self.outcomes = list(outcomes)
        self.submissions = 0
        self.shutdowns = 0
        self.shutdown_kwargs = []
        self._processes = {}

    def submit(self, function, item):
        from concurrent.futures import Future

        future = Future()
        outcome = self.outcomes[self.submissions] \
            if self.submissions < len(self.outcomes) else ("value", item)
        self.submissions += 1
        if outcome[0] == "value":
            future.set_result(outcome[1])
        elif outcome[0] == "raise":
            future.set_exception(outcome[1])
        # "hang": left unresolved, so `result(timeout=...)` raises TimeoutError
        return future

    def shutdown(self, **kwargs):
        self.shutdowns += 1
        self.shutdown_kwargs.append(dict(kwargs))


def _with_executor(pool, executor):
    pool._executor = executor
    pool._ensure_executor = lambda: pool._executor


def test_the_pooled_path_keeps_submission_order():
    pool = EvaluationPool(workers=2, task_timeout=0.05)
    _with_executor(pool, _Controlled([("value", 10), ("value", 20), ("value", 30)]))
    results = pool.map(square, ["a", "b", "c"])
    assert [(r.index, r.value) for r in results] == [(0, 10), (1, 20), (2, 30)]


def test_a_timeout_reports_the_item_that_hung_and_resumes_after_it():
    """The restart has to continue *past* the hung item.

    Resuming at or before it would re-submit the thing that just hung, and one
    non-terminating variant would stop a generation forever.
    """
    pool = EvaluationPool(workers=1, task_timeout=0.05)
    first = _Controlled([("value", 1), ("hang",), ("value", 3)])
    _with_executor(pool, first)

    rebuilt = _Controlled([("value", 33)])

    def _rebuild():
        pool._executor = pool._executor or rebuilt
        return pool._executor

    pool._ensure_executor = _rebuild
    results = pool.map(square, ["a", "b", "c"])

    assert [r.index for r in results] == [0, 1, 2]
    assert results[0].value == 1
    assert results[1].timed_out and "exceeded" in results[1].error
    assert results[2].value == 33            # the item AFTER the hang, once
    assert rebuilt.submissions == 1


def test_a_timeout_kills_the_pool():
    pool = EvaluationPool(workers=1, task_timeout=0.05)
    executor = _Controlled([("hang",)])
    _with_executor(pool, executor)
    pool._ensure_executor = lambda: pool._executor

    pool.map(square, ["a"])
    assert pool.stats.kills == 1
    assert executor.shutdowns == 1
    assert pool.stats.timed_out == 1
    # Not waiting is the whole point: the worker is hung, so waiting for it is
    # waiting forever, and leaving queued futures would run work nobody wants.
    assert executor.shutdown_kwargs == [{"wait": False, "cancel_futures": True}]


def test_shutdown_does_not_wait_either():
    pool = EvaluationPool(workers=1)
    executor = _Controlled([])
    pool._executor = executor
    pool.shutdown()
    assert executor.shutdown_kwargs == [{"wait": False, "cancel_futures": True}]


def test_a_timeout_is_reported_with_the_limit_it_exceeded():
    pool = EvaluationPool(workers=1, task_timeout=0.25)
    _with_executor(pool, _Controlled([("hang",)]))
    result = pool.map(square, ["a"])[0]
    assert result.error == "exceeded 0.25s"
    assert result.latency_ms == pytest.approx(250.0)


def test_a_raising_future_is_reported_without_killing_the_pool():
    pool = EvaluationPool(workers=2, task_timeout=1.0)
    executor = _Controlled([("raise", ValueError("nope")), ("value", 7)])
    _with_executor(pool, executor)
    results = pool.map(square, ["a", "b"])
    assert not results[0].ok and "ValueError: nope" in results[0].error
    assert results[1].value == 7
    assert pool.stats.kills == 0


def test_latency_is_recorded_in_milliseconds(monkeypatch):
    """`(end − start) · 1000`. The pool's own status is read to decide whether
    evaluation is affordable, so a factor of a thousand there is a budget
    decision made on a wrong number."""
    import aegis.eval.pool as module

    ticks = iter([1.0, 1.5])
    monkeypatch.setattr(module, "CLOCK",
                        type("C", (), {"monotonic": staticmethod(lambda: next(ticks))})())
    pool = EvaluationPool(workers=1, task_timeout=5.0)
    _with_executor(pool, _Controlled([("value", 1)]))
    result = pool.map(square, ["a"])[0]
    assert result.latency_ms == pytest.approx(500.0)
    assert pool.status()["avg_latency_ms"] == pytest.approx(500.0)


def test_the_average_latency_is_a_mean(monkeypatch):
    """Two samples, not one: with a single observation a sum divided by the
    count and a sum multiplied by it are the same number."""
    import aegis.eval.pool as module

    ticks = iter([1.0, 1.5, 2.0, 3.5])
    monkeypatch.setattr(module, "CLOCK",
                        type("C", (), {"monotonic": staticmethod(lambda: next(ticks))})())
    pool = EvaluationPool(workers=2, task_timeout=5.0)
    _with_executor(pool, _Controlled([("value", 1), ("value", 2)]))
    results = pool.map(square, ["a", "b"])
    assert [r.latency_ms for r in results] == [pytest.approx(500.0),
                                               pytest.approx(1500.0)]
    assert pool.status()["avg_latency_ms"] == pytest.approx(1000.0)


def test_the_inline_fallback_numbers_items_from_where_it_took_over():
    """A fallback that restarted its indices at zero would report two results
    for index 0 and none for the item it actually ran."""
    pool = EvaluationPool(workers=1, task_timeout=0.05)
    executor = _Controlled([("value", 1), ("hang",)])
    _with_executor(pool, executor)
    pool._ensure_executor = lambda: pool._executor      # never rebuilt

    results = pool.map(len, ["a", "bb", "ccc"])
    assert [r.index for r in results] == [0, 1, 2]
    assert results[2].value == 3             # ran inline, kept its own index


def test_the_reservation_asks_for_no_more_slots_than_there_is_work(tmp_path):
    from aegis.layers.motivation.resources import ResourceManager

    resources = ResourceManager(store_path=tmp_path / "budgets.json",
                                limits={"subprocess_slots": 16})
    pool = EvaluationPool(workers=8, resources=resources)

    # Fewer items than workers: ask for the items.
    lease = pool._reserve(2, "test")
    assert lease is not None and lease.cost.subprocess_slots == 2
    resources.commit(lease, lease.cost)

    # More items than workers: ask for the workers. Reserving one slot per item
    # would refuse a hundred-variant batch that the pool would happily run eight
    # at a time.
    lease = pool._reserve(99, "test")
    assert lease is not None and lease.cost.subprocess_slots == 8
    resources.commit(lease, lease.cost)


def test_telemetry_failure_does_not_break_a_batch():
    class _Boom:
        def record(self, *a, **k):
            raise RuntimeError("sink down")

    pool = EvaluationPool(workers=1, telemetry=_Boom())
    pool._ensure_executor = lambda: None
    assert pool.map(square, [3])[0].value == 9


def test_the_inline_path_measures_latency_in_milliseconds_too(monkeypatch):
    """`_run_inline` has its own timing, and it is the path a refused lease
    takes — so it is the one that runs when the budget is tight, which is
    exactly when somebody is reading these numbers."""
    import aegis.eval.pool as module

    ticks = iter([1.0, 1.5, 2.0, 3.5])
    monkeypatch.setattr(module, "CLOCK",
                        type("C", (), {"monotonic": staticmethod(lambda: next(ticks))})())
    pool = EvaluationPool(workers=1)
    pool._ensure_executor = lambda: None
    results = pool.map(len, ["a", "bb"])
    assert [r.latency_ms for r in results] == [pytest.approx(500.0),
                                               pytest.approx(1500.0)]
    assert pool.status()["avg_latency_ms"] == pytest.approx(1000.0)


def test_settling_nothing_never_reaches_the_resource_manager():
    """A lease that was never granted has nothing to commit, and calling commit
    with it would ask the manager to settle `None`."""
    commits = []

    class _Watching:
        def reserve(self, *a, **k):
            return None

        def commit(self, lease, cost):
            commits.append(lease)

    pool = EvaluationPool(workers=1, resources=_Watching())
    pool._settle(None)
    assert commits == []


def test_a_failing_commit_is_absorbed(tmp_path):
    class _Awkward:
        def commit(self, lease, cost):
            raise RuntimeError("ledger down")

    pool = EvaluationPool(workers=1, resources=_Awkward())
    lease = type("L", (), {"cost": "whatever"})()
    pool._settle(lease)             # swallowed, logged, the batch still returns


def test_a_lease_with_no_manager_to_settle_it_is_not_committed():
    """The other half of the guard, and a different situation: a lease exists
    but the manager is gone. Folded into one `or` with the missing-lease case,
    neither could be told from the other."""
    pool = EvaluationPool(workers=1, resources=None)
    pool._settle(type("L", (), {"cost": "whatever"})())
    assert pool.settled == 0


def test_a_settled_lease_is_counted(tmp_path):
    from aegis.layers.motivation.resources import ResourceManager

    resources = ResourceManager(store_path=tmp_path / "budgets.json")
    pool = EvaluationPool(workers=1, resources=resources)
    pool._ensure_executor = lambda: None
    pool.map(len, ["a"])
    assert pool.settled == 1
    assert pool.status()["settled"] == 1


# ── killing workers ──────────────────────────────────────────────────

class _Process:
    def __init__(self, alive=True, raises=False):
        self._alive = alive
        self._raises = raises
        self.killed = False

    def is_alive(self):
        if self._raises:
            raise OSError("cannot query this process")
        return self._alive

    def kill(self):
        self.killed = True


def test_killing_nothing_is_not_an_error():
    pool = EvaluationPool(workers=1)
    pool._kill()
    assert pool.stats.kills == 0        # there was no pool to kill


def test_a_kill_terminates_the_live_workers():
    """Shutting the executor down does not stop a worker that is mid-task; only
    killing the process does, and that is the difference between a timeout and
    a suggestion."""
    alive, finished = _Process(alive=True), _Process(alive=False)
    executor = _Controlled([])
    executor._processes = {1: alive, 2: finished}

    pool = EvaluationPool(workers=1)
    pool._executor = executor
    pool._kill()

    assert alive.killed and not finished.killed
    assert pool.stats.kills == 1
    assert pool._executor is None


def test_a_worker_that_cannot_be_killed_does_not_stop_the_sweep():
    stubborn, ordinary = _Process(raises=True), _Process(alive=True)
    executor = _Controlled([])
    executor._processes = {1: stubborn, 2: ordinary}

    pool = EvaluationPool(workers=1)
    pool._executor = executor
    pool._kill()
    assert ordinary.killed          # the second one was still reached


def test_a_shutdown_that_raises_during_a_kill_is_absorbed():
    class _Awkward:
        _processes = {}

        def shutdown(self, **kwargs):
            raise RuntimeError("will not close")

    pool = EvaluationPool(workers=1)
    pool._executor = _Awkward()
    pool._kill()
    assert pool.stats.kills == 1
    assert pool._executor is None


def test_an_executor_is_reused_across_batches():
    pool = EvaluationPool(workers=1)
    executor = _Controlled([("value", 1), ("value", 2)])
    pool._executor = executor
    pool.map(square, ["a"])
    pool.map(square, ["b"])
    assert executor.submissions == 2      # not rebuilt between batches
