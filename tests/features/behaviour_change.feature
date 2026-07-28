# language: en
Feature: Experience changes behaviour
  The development text ends its learning chain with a link the system did not
  have: действие → результат → оценка → новое знание → **изменение поведения**.
  Experience used to reach behaviour through one narrow channel — a confidence
  penalty — which cannot say "not this, here", cannot be inspected, and never
  measured whether it moved a single decision.

  Now the last arrow is an object. A preference weight shifts rankings quietly
  on every closed experience; a rule with evidence, a controlled trial and an
  expiry can remove an option outright. Both are subordinate to the gates that
  run after them, and neither may touch the work that keeps the system alive.

  Background:
    Given a behaviour policy with no rules yet

  # ── the quiet half: preferences ──────────────────────────────────────
  Scenario: A closed experience moves a preference
    When the action pays better than its state usually does
    Then the preference for that action should rise

  Scenario: A preference learns advantage, not reward
    Given an action that pays the same amount in a rich state and a poor one
    Then it should be preferred in the poor state and not in the rich one

  Scenario: A preference never removes an option
    When the action fails repeatedly in one state
    Then both courses of action should still be offered
    And the preference for the failing action should be negative

  # ── the loud half: rules ─────────────────────────────────────────────
  Scenario: A repeatedly failing action becomes a suppression rule
    When the action fails repeatedly in one state
    And the policy mines its experience
    Then a suppression rule for that state and action should be on trial

  Scenario: A rule activates only after a trial shows it helped
    Given a suppression rule on trial
    When suppressing the action pays better than allowing it
    And the policy reviews its trials
    Then the rule should be active
    And the action should no longer be offered in that state

  Scenario: Behaviour measurably changed
    Given an active suppression rule
    When the system decides repeatedly in that state
    Then the behaviour-change rate should be above zero

  Scenario: The rule is about the state, not the action everywhere
    Given an active suppression rule
    Then the action should still be offered in a different state

  # ── the rule comes off again ─────────────────────────────────────────
  Scenario: A rule that stops paying is retired
    Given an active suppression rule
    When suppressing the action stops making any difference
    And the policy reviews its trials
    Then the rule should be retired
    And the action should be offered again

  Scenario: A rule that reverses is refuted and never re-mined
    Given an active suppression rule
    When suppressing the action starts making things worse
    And the policy reviews its trials
    Then the rule should be refuted
    And re-mining the same evidence should not bring it back

  # ── what a rule may never do ─────────────────────────────────────────
  Scenario: Safety-critical work cannot be suppressed
    When the action fails repeatedly in one state
    And the policy mines its experience protecting that action
    Then no suppression rule for that action should exist

  Scenario: A safety-critical plan survives an existing rule
    Given an active suppression rule
    Then a safety-critical plan for that action should still be offered

  # ── noise is not knowledge ───────────────────────────────────────────
  Scenario: Pure noise produces no rules at all
    When outcomes are independent of the state
    And the policy mines its experience
    Then no rules should have been proposed
