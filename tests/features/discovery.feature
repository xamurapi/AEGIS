# language: en
Feature: Knowledge nobody supplied
  The system reads its own telemetry, proposes what might be true, writes the
  relationship down as a formula, commits to a test before running it, and
  records the answer. A refutation is kept as carefully as a confirmation:
  it is knowledge, and it is what stops the same appealing pattern from being
  rediscovered forever.

  Rule: A real law is found, written down and confirmed

    Scenario: A planted relationship becomes a registered discovery
      Given telemetry in which reward is 2.5 times surprise minus brier squared
      When the engine scans for hypotheses
      Then it should propose at least one hypothesis
      When the engine fits a model
      Then the formula should contain "aegis.wm.surprise"
      And the formula should contain "brier^2"
      And the model should explain at least 90 percent of held-out variance

    Scenario: The formula survives data recorded after the plan was frozen
      Given telemetry in which reward is 2.5 times surprise minus brier squared
      When the engine scans for hypotheses
      And the engine fits a model
      And the engine preregisters the experiment
      And another 300 ticks of the same relationship are recorded
      And the engine runs the observational experiment
      Then the discovery should be supported

  Rule: Noise produces nothing

    Scenario: A thousand comparisons of unrelated series register no discovery
      Given telemetry of six series unrelated to reward
      When the engine scans and fits repeatedly
      Then it should have made at least 1000 comparisons
      And no discovery should be supported

  Rule: A plan cannot be changed once the data starts arriving

    Scenario: Altering the analysis after freezing invalidates the result
      Given telemetry in which reward is 2.5 times surprise minus brier squared
      When the engine scans for hypotheses
      And the engine fits a model
      And the engine preregisters the experiment
      And the analysis is changed after the plan was frozen
      And the engine runs the observational experiment
      Then the result should be invalid

  Rule: A self-experiment gives back what it borrowed

    Scenario: An intervention on a parameter nobody whitelisted never starts
      Given a discovery engine
      When an intervention is attempted on "ETHICAL_THRESHOLD_AUTO"
      Then the intervention should not start
      And the parameter should be untouched

    Scenario: Critical health stops an intervention where it stands
      Given a discovery engine
      When an intervention is started on "explore_bonus"
      And one tick of the series runs
      Then the parameter should be at its experimental level
      When health goes critical
      Then the intervention should be aborted
      And the parameter should be back at its original value

  Rule: Knowledge that changes nothing is a report

    Scenario: A confirmed discovery reaches the systems it is about
      Given a discovery engine
      And a discovery confirmed in two separate windows
      When the engine applies what it has learned
      Then the world model should have been told
      And the discovery should record where it was applied

    Scenario: A discovery whose application made things worse goes back for re-testing
      Given a discovery engine
      And a discovery confirmed in two separate windows
      When the engine applies what it has learned
      And the metric falls afterwards
      Then the discovery should be proposed again

  Rule: A refutation is permanent

    Scenario: A refuted hypothesis is never proposed again
      Given a discovery engine
      And a hypothesis that has been refuted
      When the same hypothesis is proposed again
      Then it should be refused
