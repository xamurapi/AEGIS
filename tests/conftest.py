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
