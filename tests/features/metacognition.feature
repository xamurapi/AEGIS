# language: en
Feature: Why a strategy won, and inventing a different way
  The meta-loop over reasoning (M11). "Why" is a measurement: remove the part
  of a strategy an explanation credits and the win must disappear. "A
  principally different way" is a distance: a quota of every round's
  candidates must sit far from everything already evaluated — and then face
  the same arena as everyone else.

  Background:
    Given a reasoning engine with metacognition enabled

  Rule: An explanation is a measurement, not a story

    Scenario: A strategy won and was explained
      Given an accepted strategy whose win is carried by its abstention branch
      When the strategy is attributed
      Then the explanation should be supported
      And the mechanism should be "abstention_avoided_confident_error"
      And at least one edit should be confirmed by ablation

    Scenario: An explanation is refuted by ablation
      Given an accepted strategy whose extra step changes nothing
      When the strategy is attributed
      Then the explanation should be unsupported
      And the mechanism should be empty

    Scenario: The cortex cannot rewrite a computed attribution
      Given an accepted strategy whose win is carried by its abstention branch
      And a cortex that names a contradicting mechanism
      When the strategy is attributed
      Then the explanation should be contested
      And the mechanism should be "abstention_avoided_confident_error"

  Rule: Invention is far by measure and judged like everyone else

    Scenario: A different way is invented and accepted
      Given a weak class whose incumbent is expensive
      When invention proposes far candidates and the arena judges them
      Then at least one far candidate should be accepted
      And every accepted far candidate should be far from the prior archive

    Scenario: The same skeleton is not invented again
      Given a skeleton that failed its class three times
      When invention proposes far candidates again
      Then that skeleton should not be among the proposals
