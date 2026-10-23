Feature: Experiment runner, technical rules
  Backs "Experiment runner". Paired repetitions run side by side, so arms meet the same
  API conditions instead of one arm running early and the other late.

  Rule: Runs are started repetition by repetition, every arm in each repetition

    Scenario: Two arms, three repetitions start in paired order
      Given an experiment config with arms "S2" and "S3" and 3 repetitions
      When its runs are planned
      Then the start order is S2_rep1, S3_rep1, S2_rep2, S3_rep2, S2_rep3, S3_rep3

    Scenario: With a cap of 2, each repetition's arms run together
      Given an experiment config with arms "S2" and "S3", 2 repetitions and a parallel cap of 2
      When the experiment runs
      Then S2_rep1 and S3_rep1 are in progress together before either rep-2 run starts
