# language: en
Feature: Five higher-order cognitive systems
  As AEGIS, the universal управляющий интеллект, I must reason about causes,
  connect my knowledge, evolve only real improvements, act from motivation,
  and learn from real outcomes — the five systems the specification requires.

  Background:
    Given a fresh AEGIS instance with the five systems

  # ── System 1: World Model ────────────────────────────────────────────
  Scenario: The World Model learns cause and effect and builds a plan
    When I observe "open_company" causing "legal_review" 3 times successfully
    And I build a causal chain for the objective "open_company"
    Then the chain should predict "legal_review" as a likely effect
    And the chain should contain at least one plan step

  Scenario: The World Model warns about actions that tend to fail
    When I observe "risky_launch" causing "failure" 4 times unsuccessfully
    Then querying risks for "risky" should surface a high failure rate

  # ── System 2: Cognitive Graph ────────────────────────────────────────
  Scenario: The Cognitive Graph connects concepts and finds a path
    When I add concepts "a", "b" and "c" linked a-b and b-c
    Then there should be a path from "a" to "c"

  # ── System 3: Evolution Engine ───────────────────────────────────────
  Scenario: A mutation that improves the benchmark is kept
    Given a champion genome with fitness 0.50
    When I propose a mutation and the benchmark scores 0.70
    Then the mutation should be accepted as the new champion

  Scenario: A mutation that does not improve the benchmark is rolled back
    Given a champion genome with fitness 0.50
    When I propose a mutation and the benchmark scores 0.40
    Then the mutation should be rejected and the parameter reverted

  # ── System 4: Goal Intelligence ──────────────────────────────────────
  Scenario: Motivation shifts toward what pays off
    When I choose the objective "explore_topic" and receive reward 1.0 ten times
    Then the utility of "explore_topic" should rise above its default

  # ── System 5: Real-world Feedback Loop ───────────────────────────────
  Scenario: An experience captures the cause of the outcome
    When I record a situation "low energy" with decision "rest"
    And the real result comes back as failure with metric 0.10
    Then the stored experience should explain why it failed
    And it should be exportable as a training example
