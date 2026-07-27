"""Versioned prompt templates (spec M8.2).

Templates live on disk because a stored reasoning strategy names one, and the
arena compares strategies across runs. If the text behind a name changed
silently, every historical win rate would describe a prompt that no longer
exists.
"""
import pytest

from aegis.cortex import prompts
from aegis.cortex import schemas as S


# ── the catalogue ────────────────────────────────────────────────────

def test_every_template_the_dsl_names_exists():
    # Appendix E's built-in strategies refer to these by name.
    required = {"solve_direct", "repair_with_error", "solve_part", "combine_parts",
                "write_expression", "transfer_by_analogy", "propose_action",
                "propose_alternative", "assess_sufficiency"}
    assert required <= set(prompts.available())


def test_every_core_template_exists():
    required = {"system", "state_eval", "decision", "reflection", "curiosity",
                "skill_code", "coding_solution", "code_change", "param_adjust",
                "plan_rerank", "strategy_synthesis", "hypothesis_scan",
                "genome_proposal", "chain_refine"}
    assert required <= set(prompts.available())


def test_available_is_sorted_and_deduplicated():
    names = prompts.available()
    assert names == sorted(set(names))


def test_an_unknown_template_is_a_loud_error():
    with pytest.raises(prompts.PromptError) as excinfo:
        prompts.load("no_such_template")
    assert "no_such_template" in str(excinfo.value)


def test_loading_returns_the_file_contents():
    assert "AEGIS" in prompts.load("system")


# ── rendering ────────────────────────────────────────────────────────

def test_placeholders_are_substituted():
    rendered = prompts.render("decision", options="1. alpha\n2. beta")
    assert "1. alpha" in rendered
    assert "{options}" not in rendered


def test_json_braces_survive_rendering():
    # Every one of these templates contains a JSON example; str.format would
    # choke on the first brace.
    rendered = prompts.render("decision", options="x")
    assert '"chosen"' in rendered
    assert "{" in rendered and "}" in rendered


def test_a_missing_value_leaves_the_placeholder_visible(caplog):
    # Rendering "None" into a prompt is far harder to notice than a placeholder
    # that is still there.
    rendered = prompts.render("decision")
    assert "{options}" in rendered


def test_a_missing_value_is_logged(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="aegis.cortex.prompts"):
        prompts.render("decision")
    assert any("options" in record.getMessage() for record in caplog.records)


def test_extra_values_are_ignored():
    assert prompts.render("decision", options="x", unused="y")


def test_rendering_an_unknown_template_raises():
    with pytest.raises(prompts.PromptError):
        prompts.render("not_a_template", x=1)


# ── versioning ───────────────────────────────────────────────────────

def test_a_template_has_a_stable_version():
    assert prompts.version("system") == prompts.version("system")


def test_different_templates_have_different_versions():
    assert prompts.version("system") != prompts.version("decision")


def test_versions_covers_every_template():
    assert set(prompts.versions()) == set(prompts.available())


def test_a_version_is_a_short_hex_digest():
    digest = prompts.version("system")
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


# ── the templates and the schemas agree ──────────────────────────────

@pytest.mark.parametrize("template,schema", [
    ("state_eval", "state_eval"),
    ("decision", "decision"),
    ("reflection", "reflection"),
    ("curiosity", "curiosity"),
    ("skill_code", "skill_code"),
    ("code_change", "code_change"),
    ("param_adjust", "param_adjust"),
    ("plan_rerank", "plan_rerank"),
    ("chain_refine", "chain_refine"),
])
def test_each_template_asks_for_every_key_its_schema_requires(template, schema):
    text = prompts.load(template)
    for key in S.schema_for(schema).get("required", []):
        assert f'"{key}"' in text, f"{template} never mentions required key {key!r}"


def test_the_answer_schema_matches_the_solving_templates():
    for template in ("solve_direct", "repair_with_error", "solve_part",
                     "combine_parts", "transfer_by_analogy"):
        assert '"answer"' in prompts.load(template)


def test_the_sufficiency_template_asks_for_its_verdict_key():
    assert '"sufficient"' in prompts.load("assess_sufficiency")


def test_the_strategy_template_carries_the_grammar_placeholder():
    # The synthesizer injects the operation table; without it the model would
    # be inventing operations the interpreter cannot execute.
    assert "{grammar}" in prompts.load("strategy_synthesis")


def test_the_genome_template_warns_that_out_of_range_values_are_discarded():
    assert "discarded" in prompts.load("genome_proposal")


def test_the_rerank_template_forbids_inventing_a_candidate():
    text = prompts.load("plan_rerank")
    assert "may not invent" in text
