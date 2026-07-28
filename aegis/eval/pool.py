"""The evaluation pool: heavy work off the cognitive cycle (spec M9.1).

Evaluating ten genome variants, or a benchmark, or a reasoning arena, takes
seconds to minutes. None of it may happen inside a tick — §3.4 gives ACT twenty
milliseconds — so it happens here, in separate processes, and the tick goes on
without it.

Four properties the rest of the system depends on:

* **Deterministic order.** Results come back indexed by submission, never by
  completion. A population sorted by "whichever worker finished first" would
  make evolution's selection depend on machine load, and two runs of the same
  experiment would disagree — which is exactly the class of defect that made the
  A/B harnesses unrepeatable.
* **A hard timeout.** A worker that hangs is killed, not waited for. Since a
  running process cannot be reclaimed from the parent any other way, a timeout
  tears the pool down and rebuilds it for whatever is left; the item that hung
  is reported as a timeout rather than silently missing.
* **A lease or nothing.** Subprocess slots are a budgeted resource (M4), so the
  pool asks before it spawns. Refused, it degrades to running the work in this
  process — slower, still correct, never unbudgeted.
* **Cancellable.** Shutdown does not wait for outstanding work.

Callables and arguments must be picklable, which in practice means module-level
functions. That is a real constraint and it is the right one: anything that has
to close over live substrate state has no business running in another process.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.layers.motivation.resources import ResourceCost

logger = logging.getLogger("aegis.eval.pool")

#: Leave the machine two cores. The cognitive cycle runs on one of them, and a
#: pool that saturates every core makes the tick budget unmeetable — the thing
#: the pool exists to protect.
RESERVED_CORES = 2


@dataclass
class PoolResult:
    """One item's outcome, whatever happened to it."""

    index: int
    value: object = None
    error: str = ""
    timed_out: bool = False
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error and not self.timed_out

    def as_dict(self) -> dict:
        return {"index": self.index, "ok": self.ok, "error": self.error,
                "timed_out": self.timed_out, "latency_ms": round(self.latency_ms, 3)}


@dataclass
class PoolStats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    timed_out: int = 0
    inline: int = 0             # items run in-process for want of a lease
    batches: int = 0
    kills: int = 0
    latencies: list[float] = field(default_factory=list)


def default_workers() -> int:
    """How many workers this machine should run."""
    cores = os.cpu_count() or 1
    return max(1, min(int(cfg.EVAL_POOL_WORKERS), cores - RESERVED_CORES))


class EvaluationPool:
    """Runs picklable work in separate processes, in order, under a budget."""

    def __init__(self, workers: int | None = None,
                 task_timeout: float | None = None,
                 resources=None, telemetry=None):
        self.workers = max(1, int(default_workers() if workers is None else workers))
        self.task_timeout = float(
            cfg.EVAL_POOL_TASK_TIMEOUT if task_timeout is None else task_timeout)
        self.resources = resources
        self.telemetry = telemetry
        self.stats = PoolStats()
        #: Leases actually closed. Reported because "the budget was returned"
        #: cannot be inferred from a balance that would look the same if the
        #: pool had never asked for anything.
        self.settled = 0
        self._executor: ProcessPoolExecutor | None = None

    # ── the executor ─────────────────────────────────────────────────

    def _ensure_executor(self) -> ProcessPoolExecutor | None:
        if self._executor is None:
            try:
                self._executor = ProcessPoolExecutor(max_workers=self.workers)
            except Exception:
                # No process pool available (a restricted sandbox, an exhausted
                # handle table). Inline execution is the honest fallback: the
                # work still happens, it just does not happen elsewhere.
                logger.warning("Could not start an evaluation pool — running inline",
                               exc_info=True)
                self._executor = None
        return self._executor

    def _kill(self) -> None:
        """Tear the pool down without waiting. Used after a timeout.

        A future that timed out is still running in its worker, and there is no
        way to reclaim that process from here. Killing the pool is what makes
        the timeout a limit rather than a suggestion.
        """
        executor, self._executor = self._executor, None
        if executor is None:
            return
        self.stats.kills += 1
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            logger.warning("Pool shutdown after a timeout failed", exc_info=True)
        # `shutdown` sets `_processes` to None rather than removing it, so a
        # plain default would hand back None and the sweep would raise inside
        # the very path that exists to recover from a hang.
        for process in list((getattr(executor, "_processes", None) or {}).values()):
            try:
                if process.is_alive():
                    process.kill()
            except Exception:
                logger.debug("Could not kill a pool worker", exc_info=True)

    def shutdown(self) -> None:
        """Stop the pool, abandoning anything outstanding."""
        executor, self._executor = self._executor, None
        if executor is None:
            return
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            logger.warning("Pool shutdown failed", exc_info=True)

    # ── running work ─────────────────────────────────────────────────

    def map(self, function, items, purpose: str = "evaluation") -> list[PoolResult]:
        """Run ``function`` over ``items``, returning results in input order."""
        items = list(items)
        if not items:
            return []
        self.stats.batches += 1
        self.stats.submitted += len(items)

        lease = self._reserve(len(items), purpose)
        if lease is None and self.resources is not None:
            # Budget refused. The work still has to happen — deferring a
            # benchmark forever is not a safe degradation — but it happens here,
            # where its cost lands on this tick and is visible.
            results = self._run_inline(function, items)
            self._publish()
            return results

        try:
            results = self._run_pooled(function, items)
        finally:
            self._settle(lease)
        self._publish()
        return results

    def _run_pooled(self, function, items) -> list[PoolResult]:
        results: list[PoolResult] = []
        pending = list(enumerate(items))
        while pending:
            executor = self._ensure_executor()
            if executor is None:
                results.extend(self._run_inline(function, [item for _, item in pending],
                                                offset=pending[0][0]))
                break
            futures = []
            try:
                for index, item in pending:
                    futures.append((index, executor.submit(function, item)))
            except Exception:
                # The pool died between the check and the submit.
                logger.warning("Submitting to the evaluation pool failed",
                               exc_info=True)
                self._kill()
                results.extend(self._run_inline(function,
                                                [item for _, item in pending],
                                                offset=pending[0][0]))
                break

            # Collected in submission order, so the answer does not depend on
            # which worker happened to finish first.
            restart_from = None
            for position, (index, future) in enumerate(futures):
                started = CLOCK.monotonic()
                try:
                    value = future.result(timeout=self.task_timeout)
                except FutureTimeout:
                    results.append(PoolResult(
                        index=index, timed_out=True,
                        error=f"exceeded {self.task_timeout:g}s",
                        latency_ms=self.task_timeout * 1000))
                    self.stats.timed_out += 1
                    self._kill()
                    restart_from = position + 1
                    break
                except Exception as exc:
                    results.append(PoolResult(index=index, error=f"{type(exc).__name__}: {exc}"))
                    self.stats.failed += 1
                else:
                    latency = (CLOCK.monotonic() - started) * 1000
                    results.append(PoolResult(index=index, value=value,
                                              latency_ms=latency))
                    self.stats.completed += 1
                    self.stats.latencies.append(latency)

            pending = ([pending[position] for position in range(restart_from, len(pending))]
                       if restart_from is not None else [])
        results.sort(key=lambda result: result.index)
        return results

    def _run_inline(self, function, items, offset: int = 0) -> list[PoolResult]:
        results = []
        for position, item in enumerate(items):
            started = CLOCK.monotonic()
            try:
                value = function(item)
            except Exception as exc:
                results.append(PoolResult(index=offset + position,
                                          error=f"{type(exc).__name__}: {exc}"))
                self.stats.failed += 1
            else:
                latency = (CLOCK.monotonic() - started) * 1000
                results.append(PoolResult(index=offset + position, value=value,
                                          latency_ms=latency))
                self.stats.completed += 1
                self.stats.latencies.append(latency)
            self.stats.inline += 1
        return results

    # ── the budget ───────────────────────────────────────────────────

    def _reserve(self, count: int, purpose: str):
        if self.resources is None:
            return None
        slots = max(1, min(self.workers, count))
        try:
            return self.resources.reserve(
                ResourceCost(subprocess_slots=slots), f"pool/{purpose}", 0.5)
        except Exception:
            logger.exception("Reserving pool slots failed")
            return None

    def _settle(self, lease) -> None:
        # Two different reasons, two statements. Folded into one `or` they were
        # indistinguishable — either branch reached the same call and the
        # try/except absorbed the difference, so a mutant that swapped the
        # operator changed nothing observable and survived every test.
        if lease is None:
            return                       # nothing was granted, nothing to close
        if self.resources is None:
            logger.debug("A lease exists with no resource manager to settle it")
            return
        self.settled += 1
        try:
            self.resources.commit(lease, lease.cost)
        except Exception:
            logger.exception("Settling the pool lease failed")

    # ── reporting ────────────────────────────────────────────────────

    def _publish(self) -> None:
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        try:
            self.telemetry.record(M.RES_SPENT, self.stats.completed, 0,
                                  tags={"kind": "pool_tasks"})
        except Exception:
            logger.exception("Pool telemetry record failed")

    def status(self) -> dict:
        latencies = self.stats.latencies[-100:]
        return {
            "workers": self.workers,
            "task_timeout": self.task_timeout,
            "running": self._executor is not None,
            "submitted": self.stats.submitted,
            "completed": self.stats.completed,
            "failed": self.stats.failed,
            "timed_out": self.stats.timed_out,
            "inline": self.stats.inline,
            "batches": self.stats.batches,
            "kills": self.stats.kills,
            "settled": self.settled,
            "avg_latency_ms": (round(sum(latencies) / len(latencies), 3)
                               if latencies else 0.0),
        }
