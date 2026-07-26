"""pytest-bdd step definitions for tests/features/safety_and_resilience.feature.

Each scenario drives the real components (sandbox, API app, evolution engine,
feedback loop, world model) against isolated state, so the .feature file is both
the specification of the safety guarantees and the check that they hold.
"""
import json
import tempfile
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

scenarios("features/safety_and_resilience.feature")


@pytest.fixture
def ctx(tmp_path):
    return {"tmp": tmp_path, "code": None, "verdict": None, "reasons": [],
            "result": None, "marker": None, "client": None, "server": None,
            "first_message": None, "response": None, "evolution": None,
            "substrate": None, "feedback": None, "chain": None, "rows": None}


# ══ Sandbox ═══════════════════════════════════════════════════════════

@given(parsers.parse('a self-written skill that hides "{name}" in a parameter annotation'))
def _skill_annotation(ctx, name):
    ctx["code"] = f'def solve(p, _z: {name}("os").getcwd()):\n    return 1\n'


@given(parsers.parse('a self-written skill that hides "{name}" in a lambda keyword default'))
def _skill_lambda(ctx, name):
    ctx["code"] = f'f = lambda *, a={name}("os").getcwd(): a\ndef solve(p):\n    return 1\n'


@given("a self-written skill whose annotation would write a file to disk")
def _skill_writes_file(ctx):
    marker = Path(tempfile.gettempdir()) / "aegis_bdd_escape_proof.txt"
    if marker.exists():
        marker.unlink()
    ctx["marker"] = marker
    ctx["code"] = (f'def solve(p, _z: __import__("pathlib").Path(r"{marker}").write_text("X")):\n'
                   f'    return 1\n')


@given("a self-written skill that computes an integer square root with type hints")
def _skill_legit(ctx):
    ctx["code"] = ("import math\n"
                   "def solve(payload: dict) -> int:\n"
                   "    return int(math.sqrt(payload['n']))\n")
    ctx["payload"] = {"n": 16}


@when("the safety gate inspects the skill")
def _inspect(ctx):
    from aegis.eval.sandbox import check_safe
    ctx["verdict"], ctx["reasons"] = check_safe(ctx["code"])


@when("the skill is executed in the sandbox")
def _execute(ctx):
    from aegis.eval.sandbox import run_skill
    ctx["result"] = run_skill(ctx["code"], "solve", ctx.get("payload"), timeout=5.0)


@then("the skill should be rejected")
def _rejected(ctx):
    assert ctx["verdict"] is False


@then(parsers.parse('the reason should mention "{needle}"'))
def _reason_mentions(ctx, needle):
    assert any(needle in r for r in ctx["reasons"]), ctx["reasons"]


@then("the execution should fail as unsafe")
def _exec_unsafe(ctx):
    assert ctx["result"]["ok"] is False
    assert "unsafe code" in ctx["result"]["error"]


@then("no file should have been written")
def _no_file(ctx):
    marker = ctx["marker"]
    try:
        assert not marker.exists(), "sandbox escape executed — payload wrote to disk"
    finally:
        if marker.exists():
            marker.unlink()


@then(parsers.parse("the skill should return {expected:d}"))
def _returns(ctx, expected):
    assert ctx["result"] == {"ok": True, "result": expected}


# ══ Control plane ═════════════════════════════════════════════════════

@given("the control plane requires an API token")
def _api_token(ctx, monkeypatch):
    import aegis.config as cfg
    from fastapi.testclient import TestClient
    from aegis.api import server

    monkeypatch.setattr(cfg, "API_TOKEN", "s3cret")
    monkeypatch.setattr(server, "connected_ws", [])

    class _FakeSubstrate:
        def full_status(self):
            return {"internal": "state"}

    monkeypatch.setattr(server, "substrate", _FakeSubstrate())
    ctx["server"] = server
    ctx["client"] = TestClient(server.app)


@given("the runtime has not been started")
def _runtime_down(ctx, monkeypatch):
    import aegis.config as cfg
    from fastapi.testclient import TestClient
    from aegis.api import server

    monkeypatch.setattr(cfg, "API_TOKEN", "")
    monkeypatch.setattr(server, "substrate", None)
    ctx["server"] = server
    ctx["client"] = TestClient(server.app)


@when("a client connects to the status stream without a token")
def _connect_no_token(ctx):
    with ctx["client"].websocket_connect("/ws") as ws:
        ctx["first_message"] = json.loads(ws.receive_text())
        ctx["subscribed"] = len(ctx["server"].connected_ws)


@when("a client connects to the status stream with the correct token")
def _connect_with_token(ctx):
    with ctx["client"].websocket_connect("/ws?token=s3cret") as ws:
        ctx["first_message"] = json.loads(ws.receive_text())
        ctx["subscribed"] = len(ctx["server"].connected_ws)


@when("an operator asks for the status")
def _ask_status(ctx):
    ctx["response"] = ctx["client"].get("/api/status")


@then("it should be told it is unauthorized")
def _told_unauthorized(ctx):
    assert ctx["first_message"] == {"error": "unauthorized"}


@then("it should not be subscribed to the state broadcast")
def _not_subscribed(ctx):
    assert ctx["subscribed"] == 0


@then("it should receive the full status")
def _got_status(ctx):
    assert ctx["first_message"] == {"internal": "state"}


@then("it should be subscribed to the state broadcast")
def _subscribed(ctx):
    assert ctx["subscribed"] == 1


@then(parsers.parse("the API should answer {code:d}"))
def _api_answers(ctx, code):
    assert ctx["response"].status_code == code


# ══ Evolution ═════════════════════════════════════════════════════════

@given(parsers.parse('a champion parameter "{param}" of {value:f}'))
def _champion(ctx, param, value):
    from aegis.layers.evolution_engine import EvolutionEngine
    ev = EvolutionEngine(store_path=ctx["tmp"] / "lineage.json")
    ev.register_champion({param: value}, fitness=0.5)
    ctx["evolution"] = ev
    ctx["param"] = param


@given(parsers.parse('a pending mutation of "{param}" to {value:f} that no benchmark has scored'))
def _pending_mutation(ctx, param, value, monkeypatch):
    import aegis.config as cfg
    from aegis.layers.substrate import Substrate

    monkeypatch.setattr(cfg, "CHECKPOINTS_DIR", ctx["tmp"])
    sub = Substrate()
    monkeypatch.setattr(sub, "_checkpoint_path", ctx["tmp"] / "latest.json")
    sub.evolution = ctx["evolution"]
    sub.self_mod.parameters[param] = ctx["evolution"].champion["genome"][param]
    sub.evolution.propose_mutation(tick=5)
    sub.self_mod.parameters[param] = value
    sub.evolution.candidate["new_value"] = value
    sub.evolution.candidate["genome"][param] = value
    (ctx["tmp"] / "latest.json").write_text(json.dumps({
        "tick_count": 5, "version": "1.0.0", "parameters": {param: value},
    }), encoding="utf-8")
    ctx["substrate"] = sub


@when("the system restarts and restores its checkpoint")
def _restart(ctx):
    ctx["substrate"]._restore_checkpoint()


@when("a mutation is proposed")
def _propose(ctx):
    ctx["mutation"] = ctx["evolution"].propose_mutation(tick=1)


@then(parsers.parse('the champion should still record {value:f} for "{param}"'))
def _champion_unchanged(ctx, value, param):
    assert ctx["substrate"].evolution.champion["genome"][param] == pytest.approx(value)


@then(parsers.parse("the live parameter should still be {value:f} awaiting judgement"))
def _live_param(ctx, value):
    assert ctx["substrate"].self_mod.parameters[ctx["param"]] == pytest.approx(value)
    assert ctx["substrate"].evolution.candidate is not None


@then("the proposed value should differ from zero")
def _differs_from_zero(ctx):
    assert ctx["mutation"] is not None
    assert ctx["mutation"]["new_value"] != 0.0


# ══ Experience log ════════════════════════════════════════════════════

@given(parsers.parse("an experience log with {n:d} valid experiences and 1 torn line"))
def _torn_log(ctx, n):
    log = ctx["tmp"] / "experiences.jsonl"
    lines = [json.dumps({"id": f"exp_{i + 1:08d}", "situation": f"s{i}",
                         "decision": "d", "success": True, "metric": 0.5,
                         "cause": "c"}) for i in range(n)]
    lines.insert(1, '{"id": "exp_0000000')      # crash mid-append
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ctx["log"] = log


@given("an experience log with a row that predates the current schema")
def _legacy_log(ctx):
    log = ctx["tmp"] / "experiences.jsonl"
    log.write_text(json.dumps({"id": "exp_00000001", "situation": "s"}) + "\n",
                   encoding="utf-8")
    ctx["log"] = log


@when("the feedback loop loads the log")
def _load_log(ctx):
    from aegis.layers.feedback_loop import FeedbackLoop
    ctx["feedback"] = FeedbackLoop(store_path=ctx["log"])


@when("the experiences are exported as training examples")
def _export(ctx):
    from aegis.layers.feedback_loop import FeedbackLoop
    ctx["rows"] = FeedbackLoop(store_path=ctx["log"]).export_examples()


@then(parsers.parse("it should report {n:d} resolved experiences"))
def _resolved(ctx, n):
    assert ctx["feedback"].resolved == n


@then(parsers.parse("{n:d} training example should be produced"))
def _examples(ctx, n):
    assert len(ctx["rows"]) == n


# ══ World model ═══════════════════════════════════════════════════════

_SHAPES = {"a dictionary": {"step": "x"}, "a number": 42, "a string": "register"}


@given(parsers.parse("an LLM proposes a chain whose plan is {shape}"))
def _malformed_plan(ctx, shape):
    ctx["parsed"] = {"objective": "open a company", "plan": _SHAPES[shape.strip()]}


@when("the world model refines the chain")
def _refine(ctx):
    from aegis.layers.world_model import WorldModel
    wm = WorldModel(store_path=ctx["tmp"] / "model.json")
    ctx["chain"] = wm.refine_chain(ctx["parsed"])


@then("the stored chain should have an empty plan")
def _empty_plan(ctx):
    assert ctx["chain"] is not None
    assert ctx["chain"]["plan"] == []
