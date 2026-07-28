# language: en
Feature: Thinking is a strategy, and a strategy is data
  A reasoning strategy is a declarative pipeline drawn from a fixed vocabulary,
  not Python. Some strategies are written by a language model, so the grammar
  and the interpreter — not good intentions — are what keep a synthesised
  strategy from doing something nobody sanctioned.

  Background:
    Given a reasoning engine with the built-in strategies

  Rule: A strategy the interpreter cannot run never enters the library

    Scenario: An operation outside the vocabulary is refused at the door
      When a strategy using the operation "EXEC" is offered
      Then it should be refused
      And the library should not contain it

    Scenario: A strategy that is only a rename of another is refused
      When a strategy identical in shape to "direct" is offered
      Then it should be refused

    Scenario: Every built-in strategy is admissible
      Then every built-in strategy should pass validation

  Rule: The interpreter's limits cannot be argued with

    Scenario: A loop whose condition never goes false still stops
      When a strategy loops forever on 12 steps of budget
      Then it should stop within 13 steps

    Scenario: A gene cannot buy more budget than the configuration allows
      When the step budget is asked to be 999 against a maximum of 4
      Then no more than 4 steps should run

    Scenario: A strategy reaches nothing that was not handed to it
      When a strategy computes "__import__('os').getcwd()" with no sandbox
      Then nothing should have been evaluated

  Rule: A strategy may check its own answer, never the right answer

    Scenario: The benchmark's grader is not available to a strategy
      When a strategy verifies its answer with the checker "task"
      Then the verification should report that the task carries no check

    Scenario: A guessed answer is distinguishable from a reasoned one
      When a task with missing data is worked by "abstain_on_low_confidence"
      Then the engine should abstain
      And the answer should count as correct

    Scenario: Guessing on that task is a confident error
      When a task with missing data is worked by "direct"
      Then the engine should record a confident error

  Rule: Experience decides which strategy suits which class of problem

    Scenario: Selection is spread while a class is unmeasured
      When 200 problems from one family are worked
      Then more than one strategy should have been tried on it

    Scenario: The worst measured class becomes the weakness
      When 24 problems with missing data are worked by "direct"
      Then the top weakness should be "missing_data"

    Scenario: A class with too little evidence is not called weak
      When 4 problems with missing data are worked by "direct"
      Then there should be no weakness
