# language: en
Feature: Motivation that costs something
  A goal that everyone agrees is valuable and that nothing is spent on is not a
  motive, it is an opinion. The chain the spec asks for is
  goal → value → priority → resource → action, and the fourth link is what makes
  the rest real: an action with no lease does not run.

  Rule: Nothing runs without a lease

    Scenario: A lease is granted when the budget allows it
      Given a resource manager with a full budget
      When a lease is requested for 1000 tokens
      Then the lease should be granted
      And the tokens should be counted as reserved

    Scenario: A lease is refused when the budget is exhausted
      Given a resource manager with no token budget at all
      When a lease is requested for 1000 tokens
      Then the lease should be refused

    Scenario: Releasing a lease gives the budget back
      Given a resource manager with a full budget
      When a lease is requested for 1000 tokens
      And the lease is released
      Then the reserved tokens should be back to zero

    Scenario: What was actually spent is what is charged
      Given a resource manager with a full budget
      When a lease is requested for 1000 tokens
      And only 200 tokens are actually used
      Then 200 tokens should be recorded as spent

  Rule: Safety keeps a floor nothing can take

    Scenario: Safety-critical work is affordable when ordinary work is not
      Given a resource manager with almost no budget left
      Then ordinary work should not be affordable
      But safety-critical work should still be affordable

  Rule: Waiting raises priority, so nothing starves

    Scenario: A long-waiting candidate overtakes a fresher one of equal value
      Given a priority scheduler
      When a candidate has been waiting 500 ticks
      And an identical candidate has just arrived
      Then the long-waiting candidate should be ordered first

    Scenario: Value still counts for more than waiting alone
      Given a priority scheduler
      When a candidate of high value has just arrived
      And a candidate of no value has been waiting 10 ticks
      Then the valuable candidate should be ordered first

  Rule: Budget follows return, but never falls to nothing

    Scenario: An activity that pays nothing loses share but keeps a floor
      Given a ROI tracker
      When an activity spends repeatedly and returns nothing
      And the budget is reallocated
      Then its share should be above zero

    Scenario: An activity that pays well gains share
      Given a ROI tracker
      When one activity returns well and another returns nothing
      And the budget is reallocated
      Then the paying activity should hold the larger share
