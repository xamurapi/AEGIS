# language: en
Feature: Decisions made by comparing plans
  AEGIS used to decide by ranking its own wishes: take the goal with the
  highest priority times remaining progress. Nothing in that consulted what it
  had learned about which courses of action actually work. With a world model
  that can price a course of action, a decision becomes a comparison of plans —
  and the planner's only power is to say what looks best. Every gate that can
  refuse runs after it, in a fixed order.

  Background:
    Given a running AEGIS instance with a planner

  # ── the planner proposes ─────────────────────────────────────────────
  Scenario: A tick produces a plan naming a real action
    When the system takes a tick
    Then a plan should have been built
    And the plan should name an action from the registry
    And the action should hold a resource lease

  Scenario: Evidence changes which course of action is chosen
    Given the world model has learned that "rest" pays well
    And the world model has learned that "dream" pays badly
    When the system scores both plans
    Then the better-paying plan should score higher

  Scenario: A plan explains itself from the numbers that produced it
    When the system takes a tick
    Then the plan should carry a rationale naming its objective and action

  # ── the gates dispose ────────────────────────────────────────────────
  Scenario: Without a lease nothing is executed
    Given every resource budget is exhausted
    When the system takes a tick
    Then no plan should have been executed
    And the refusal should be recorded as a resource block

  Scenario: Ethics is the last gate and cannot be argued with
    Given the ethics core refuses every action
    When the system takes a tick
    Then no plan should have been executed
    And the refusal should be recorded as an ethics block
    And no lease should still be held

  # ── the cortex may permute, never extend ─────────────────────────────
  Scenario: A model cannot introduce an action outside the shortlist
    Given a cortex that answers with an index outside the shortlist
    When the system takes a tick
    Then the planner's own choice should stand

  # ── the loop closes ──────────────────────────────────────────────────
  Scenario: The forecast is recorded before the action, not after
    When the system takes a tick
    Then a forecast should exist for the chosen action

  Scenario: The promise is measured against what was realised
    When the system takes several ticks
    Then the gap between promised and realised value should be measured
