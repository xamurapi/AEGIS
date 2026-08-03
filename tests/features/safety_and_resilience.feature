# language: en
Feature: Safety and resilience of the cognitive core
  AEGIS writes and runs its own code, exposes a control plane, evolves its own
  parameters and learns from a log it appends to continuously. Each of those is
  a place where a defect becomes a security or data-loss incident, so the
  guarantees below are specified as behaviour and checked on every run.

  # ── Sandbox: self-written skills must never escape ───────────────────
  Rule: Self-written skill code cannot execute anything outside the allowlist

    Scenario: An exploit hidden in a type annotation is rejected
      Given a self-written skill that hides "__import__" in a parameter annotation
      When the safety gate inspects the skill
      Then the skill should be rejected
      And the reason should mention "__import__"

    Scenario: An exploit hidden in a lambda keyword default is rejected
      Given a self-written skill that hides "__import__" in a lambda keyword default
      When the safety gate inspects the skill
      Then the skill should be rejected

    Scenario: The hidden payload never runs
      Given a self-written skill whose annotation would write a file to disk
      When the skill is executed in the sandbox
      Then the execution should fail as unsafe
      And no file should have been written

    Scenario: Ordinary typed skill code still runs
      Given a self-written skill that computes an integer square root with type hints
      When the skill is executed in the sandbox
      Then the skill should return 4

    Scenario: An attribute reached by string rather than by syntax is rejected
      Given a self-written skill that reaches the interpreter through a string
      When the safety gate inspects the skill
      Then the skill should be rejected
      And the reason should mention "attrgetter"

    Scenario: The string-based escape never reaches the child process
      Given a self-written skill that reaches the interpreter through a string
      When the skill is executed in the sandbox
      Then the execution should fail as unsafe
      And nothing about this machine should have been returned

  # ── Control plane: internal state is not public ──────────────────────
  Rule: Internal state reaches only authenticated operators

    Scenario: An unauthenticated dashboard client is refused and gets no state
      Given the control plane requires an API token
      When a client connects to the status stream without a token
      Then it should be told it is unauthorized
      And it should not be subscribed to the state broadcast

    Scenario: An authenticated dashboard client receives state
      Given the control plane requires an API token
      When a client connects to the status stream with the correct token
      Then it should receive the full status
      And it should be subscribed to the state broadcast

    Scenario: The API reports honestly that the runtime is not started
      Given the runtime has not been started
      When an operator asks for the status
      Then the API should answer 503

  # ── Evolution: only measured improvements survive ────────────────────
  Rule: A parameter change is kept only when a benchmark scored it

    Scenario: A restart does not adopt a mutation that was never judged
      Given a champion gene "w_ev" of 1.00
      And a pending mutation of "w_ev" to 1.90 that no benchmark has scored
      When the system restarts and restores its checkpoint
      Then the running configuration should still be 1.00 for "w_ev"
      And the mutation should still be awaiting judgement

    Scenario: A gene sitting at the bottom of its range is still explored
      Given a champion gene "w_ev" of 0.00
      When mutations are proposed over several generations
      Then that gene should have moved off zero

  # ── Experience log: learning data survives partial writes ────────────
  Rule: A damaged experience log degrades, it does not vanish

    Scenario: A torn line does not destroy the rest of the history
      Given an experience log with 2 valid experiences and 1 torn line
      When the feedback loop loads the log
      Then it should report 2 resolved experiences

    Scenario: An experience written by an older version is still exportable
      Given an experience log with a row that predates the current schema
      When the experiences are exported as training examples
      Then 1 training example should be produced

  # ── World model: untrusted LLM output cannot break planning ──────────
  Rule: A malformed model answer is coerced, never propagated

    Scenario Outline: A malformed plan does not break chain refinement
      Given an LLM proposes a chain whose plan is <shape>
      When the world model refines the chain
      Then the stored chain should have an empty plan

      Examples:
        | shape       |
        | a dictionary|
        | a number    |
        | a string    |
