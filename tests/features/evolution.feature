# language: en
Feature: Ten variants, judged, and only the best kept
  Evolution here is a generation, not a guess. Ten variants are composed by
  fixed rules, judged on a validation split they were not tuned on, and the
  winner is confirmed on a test split that selection never sees. What cannot
  be changed at all is not in the genome.

  Rule: A generation is composed by rule, not by chance

    Scenario: A generation has the size the configuration asks for
      Given a population of ten
      When a generation is composed from two elites
      Then it should hold 10 genomes

    Scenario: The elites carry forward unchanged
      Given a population of ten
      When a generation is composed from two elites
      Then the first two genomes should be the elites themselves

    Scenario: The same generation is composed the same way twice
      Given a population of ten
      When a generation is composed from two elites
      And an identical population composes the same generation
      Then the two generations should be identical

  Rule: Selection is total, so a champion never depends on list order

    Scenario: Variants are ranked best first
      Given a population of ten
      When four variants are judged with fitnesses 0.1, 0.9, 0.5 and 0.3
      Then the best should be the one scoring 0.9

    Scenario: A tie is broken by identity rather than by position
      Given a population of ten
      When two variants are judged with the same fitness
      Then the ranking should be the same whichever order they arrived in

  Rule: The genome cannot reach what must not change

    Scenario: No gene names an immutable parameter
      Given the genome schema
      Then no gene should name a parameter from the immutable set

    Scenario: Every gene stays inside its declared range
      Given the genome schema
      When a genome is pushed past every bound
      Then every gene should be clamped back into its range

    Scenario: A gene the schema does not declare is ignored
      Given the genome schema
      When a genome carrying an undeclared gene is loaded
      Then the undeclared gene should not survive
