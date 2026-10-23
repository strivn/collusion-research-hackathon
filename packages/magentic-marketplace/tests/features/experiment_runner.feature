Feature: Experiment runner
  One experiment config lists arms (each a run config, e.g. S1-S4) and a number of
  repetitions. The runner starts every arm x repetition as its own run, in parallel up
  to a cap, with paired budget seeds across arms, and records each run in the tracker.

  Rule: Every arm x repetition becomes its own run with its own folder and schema

    Scenario: Two arms, three repetitions make six runs
      Given an experiment config "exp1" with arms "S2" and "S3" and 3 repetitions
      When the experiment runs
      Then 6 runs finish
      And run "S3" repetition 2 has folder "research/results/exp1/S3_rep2" and schema "exp1_S3_rep2"

    Scenario: Each run folder holds the run's outputs
      Given a finished run "exp1/S2_rep1"
      Then its folder holds the resolved config, the budgets and the measurement tables

  Rule: Repetition i uses the same budget seed in every arm

    Scenario: Paired seeds
      Given an experiment config with arms "S2" and "S3", 2 repetitions and base seed 7
      When the experiment runs
      Then "S2_rep1" and "S3_rep1" give every customer the same budget
      And "S2_rep1" and "S2_rep2" use different seeds

  Rule: Runs execute in parallel up to a cap

    Scenario: No more than the cap at once
      Given an experiment with 6 runs and a parallel cap of 3
      When the experiment runs
      Then at no time are more than 3 runs in progress

  Rule: A failed run does not stop the others

    Scenario: One failure, the rest finish
      Given an experiment with 4 runs where run "S3_rep1" fails
      When the experiment runs
      Then the other 3 runs finish
      And "S3_rep1" is reported as failed with its error

  Rule: Every run is recorded in the tracker

    Scenario: A tracker row per run
      Given a finished run "exp1/S2_rep1" on "mexican_10_30" with "claude-haiku-5-5"
      Then "research/tracker.md" has a row with run "exp1/S2_rep1", arm "S2", date, model, data, results folder and status "done"
