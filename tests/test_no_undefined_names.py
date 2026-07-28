"""The static gate that would have caught the ACT-phase ``NameError``.

``phases/act.py`` used ``Event`` and ``Layer`` without importing them, on a
branch only a *successful* curiosity call reaches. The suite was green, the
module sat at 88% coverage, and the phase aborted halfway through every time
that call succeeded. Neither coverage nor mutation testing can see a defect like
that: there was no wrong assertion and nothing to mutate — the name simply was
not there.

So the gate asks a different question, and asks it of every module: is every
name this file reads bound somewhere it can see?
"""
import subprocess
import sys
from pathlib import Path

from scripts.check_undefined_names import scan, undefined_in

ROOT = Path(__file__).resolve().parent.parent


def test_the_package_has_no_undefined_names():
    report = scan("aegis")
    assert report == {}, "\n".join(
        f"{f}:{line}: undefined name {name!r}"
        for f, findings in sorted(report.items()) for name, line in findings)


def test_the_tests_and_scripts_have_none_either():
    assert scan("tests", "scripts") == {}


def test_the_gate_reports_a_name_that_was_never_imported(tmp_path):
    """The exact shape of the defect, reduced."""
    module = tmp_path / "leaky.py"
    module.write_text(
        "def go(bus):\n"
        "    return bus.publish(Event(source=Layer.SUBSTRATE))\n",
        encoding="utf-8")
    assert undefined_in(module) == [("Event", 2), ("Layer", 2)]


def test_an_imported_name_is_bound(tmp_path):
    module = tmp_path / "clean.py"
    module.write_text(
        "from aegis.event_bus import Event, Layer\n\n"
        "def go(bus):\n"
        "    return bus.publish(Event(source=Layer.SUBSTRATE))\n",
        encoding="utf-8")
    assert undefined_in(module) == []


def test_every_binding_form_counts_as_a_definition(tmp_path):
    """Comprehensions, walrus, except-as, match and star-args must not be
    reported — a checker with false positives is one that gets switched off."""
    module = tmp_path / "forms.py"
    module.write_text(
        "import json\n"
        "\n"
        "CONST = 1\n"
        "\n"
        "\n"
        "class Thing:\n"
        "    field = CONST\n"
        "\n"
        "\n"
        "def go(a, /, b, *rest, c=1, **kw):\n"
        "    squares = [x * x for x in rest]\n"
        "    pairs = {k: v for k, v in kw.items()}\n"
        "    if (total := sum(squares)) > 0:\n"
        "        pass\n"
        "    try:\n"
        "        json.dumps(pairs)\n"
        "    except ValueError as exc:\n"
        "        return exc\n"
        "    match a:\n"
        "        case [first, *tail]:\n"
        "            return first, tail\n"
        "        case {'k': value, **extra}:\n"
        "            return value, extra\n"
        "        case other:\n"
        "            return other, b, c, total, Thing\n"
        "\n"
        "\n"
        "double = lambda n: n * 2\n",
        encoding="utf-8")
    assert undefined_in(module) == []


def test_a_global_declaration_binds(tmp_path):
    module = tmp_path / "glob.py"
    module.write_text(
        "def setup():\n"
        "    global HANDLE\n"
        "    HANDLE = 1\n"
        "\n"
        "def use():\n"
        "    return HANDLE\n",
        encoding="utf-8")
    assert undefined_in(module) == []


def test_each_missing_name_is_reported_once(tmp_path):
    module = tmp_path / "repeat.py"
    module.write_text(
        "def go():\n"
        "    return Missing() or Missing() or Missing()\n",
        encoding="utf-8")
    assert undefined_in(module) == [("Missing", 2)]


def test_the_script_exits_non_zero_on_a_finding(tmp_path):
    """The CI contract: a finding is a failed build, not a printed warning."""
    module = tmp_path / "bad.py"
    module.write_text("def go():\n    return Absent\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_undefined_names.py"),
         str(module)],
        capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 1
    assert "undefined name 'Absent'" in result.stdout


def test_the_script_exits_zero_on_a_clean_package():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_undefined_names.py")],
        capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 0
