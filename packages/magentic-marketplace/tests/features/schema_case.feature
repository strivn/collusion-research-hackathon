Feature: Experiment names with capital letters
  Postgres stores an unquoted schema name in lowercase. A run whose experiment name has
  capitals must still be found by every command that reads it afterwards.
  (Proposed by the market-core session after the runner smoke run; awaiting Ivan's approval.)

  Rule: A run is found by the experiment name it was started with, whatever its case

    Scenario: A run named with capitals can be scored
      Given a finished run with experiment name "rsmoke_S1_rep1"
      When it is scored as "rsmoke_S1_rep1"
      Then the score finds the run's schema
