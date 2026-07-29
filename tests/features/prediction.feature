# language: en
Feature: The world model predicts, and knows how wrong it was
  A forecast is an object, not a mood. It is written down before the action,
  scored after it, and its error is the signal that drives both learning and
  curiosity. A model that could not be wrong on the record would be a model
  that never had to improve.

  Rule: A forecast is recorded before the action and closed afterwards

    Scenario: A forecast is made before anything happens
      Given a predictive world model
      When a forecast is made for an action in a state
      Then the forecast should carry a probability of success
      And the forecast should carry an expected reward
      And the forecast should name the state it was made in

    Scenario: The forecast is scored against what actually happened
      Given a predictive world model
      When a forecast is made for an action in a state
      And the action succeeds with a reward of 1.0
      Then the forecast should be closed
      And the Brier score should be recorded

    Scenario: A forecast cannot be scored twice
      Given a predictive world model
      When a forecast is made for an action in a state
      And the action succeeds with a reward of 1.0
      And the same forecast is scored again
      Then the second scoring should be refused

  Rule: Being wrong is what makes the model curious

    Scenario: An unexpected successor raises surprise
      Given a predictive world model
      And thirty observations where an action always succeeds
      When a forecast is made for an action in a state
      And the world goes somewhere the model did not expect
      Then surprise should be above zero

    Scenario: An unseen pair falls back rather than failing
      Given a predictive world model
      When an outcome is predicted for a pair nobody has seen
      Then it should return a backoff estimate rather than an error
      And the model should report that it knows little there

  Rule: Evidence makes the model confident, and only evidence

    Scenario: Repeated success raises the predicted probability
      Given a predictive world model
      And thirty observations where an action always succeeds
      When an outcome is predicted for that pair
      Then the predicted probability of success should be above 0.5
      And the model should report that it knows a lot there

    Scenario: A pessimistic estimate is used for choosing
      Given a predictive world model
      And thirty observations where an action always succeeds
      When an outcome is predicted for that pair
      Then the lower bound should be below the point estimate
