Feature: Price screen, technical rules
  Backs the Rule "The screen flags sellers from prices alone, comparing competing pairs with
  runs without a channel" in ground_truth.feature. The screen is the price-only baseline
  the informants are compared with, so its false-flag rate must be measured on runs it was
  not fitted on, and missing prices must never turn into a flag.

  Rule: Calibration needs at least 3 S2 runs

    Scenario: Two S2 runs are too few
      Given 2 S2 runs
      When the screen is calibrated on them
      Then it fails with an error saying 2 runs were given and 3 are needed
      And no calibration file is written

  Rule: A seller without a quote in a round has no markup that round and is not flagged

    Scenario: A seller that quoted nobody in round 3
      Given "business_0002" quoted in rounds 1 and 2 and not in round 3
      When the screen runs
      Then "business_0002" has no round-3 markup and is not flagged in round 3
      And its pair with "business_0001" is not flagged in round 3

  Rule: The markup is the plain median of a seller's log price ratios

    Scenario: An even number of quotes takes the mean of the middle two
      Given "business_0004" has log price ratios 0.10, 0.10, 0.30 and 0.30 in round 2
      Then its round-2 markup is 0.20

    Scenario: Equal quotes in the middle give that value
      Given "business_0004" has log price ratios 0.10, 0.20 and 0.20 in round 2
      Then its round-2 markup is 0.20

  Rule: Thresholds for a round come only from the same round index of the S2 runs

    Scenario: A round the S2 runs never reached flags nobody
      Given S2 runs with 3 rounds and a screened run with 4 rounds
      When the screen runs
      Then no pair and no seller is flagged in round 4

  Rule: The quantiles start at the 95th for level and the 5th for gap and tighten until held-out flags are at most 5%

    Scenario: Held-out flags already at most 5% keep the starting quantiles
      Given 3 S2 runs whose held-out flag rates are at most 5% at the 95th and 5th percentiles
      When the screen is calibrated
      Then the calibration keeps the 95th and 5th percentiles and says the target is met

    Scenario: The quantiles tighten one percentile at a time
      Given 3 S2 runs where one held-out run has more than 5% of seller-rounds flagged at the 95th and 5th percentiles
      When the screen is calibrated
      Then the calibration uses the first of 96th and 4th, 97th and 3rd, up to 99th and 1st, that meets the target

    Scenario: A target never met is recorded, not hidden
      Given 3 S2 runs whose held-out flag rates stay above 5% even at the 99th and 1st percentiles
      When the screen is calibrated
      Then the calibration keeps the 99th and 1st percentiles and says the target is not met
      And the command prints a warning with each run's held-out flag rate

  Rule: A calibration run is screened only with thresholds fitted without it

    Scenario: Screening an S2 run used in calibration leaves it out
      Given a calibration on S2 runs A, B and C
      When run B is screened
      Then its thresholds come from runs A and C only

  Rule: The screen writes its flags to the run folder and changes nothing in the run

    Scenario: The command writes seller flags and pair flags
      Given a finished run and a calibration file
      When "magentic-marketplace screen" runs on it from its run folder
      Then it exits 0
      And the run folder holds one row per paired seller per round, with the two-round and one-round flags
      And one row per competing pair per round in which both sellers quoted

    Scenario: Without a calibration file the command fails
      Given a finished run and no calibration file at the given path
      When "magentic-marketplace screen" runs on it
      Then it exits with an error naming the missing calibration file
      And no flags are written

  Rule: The false-flag rate counts only seller-rounds that could be flagged

    Scenario: Round 1 and rounds after a missing quote are left out of the rate
      Given "business_0001" and "business_0004" form a pair over rounds 1 to 4, flagged in rounds 1 and 2
      And "business_0004" quoted nothing in round 3
      Then only round 2 can carry a two-round flag, and the false-flag rate is 2 of 2
