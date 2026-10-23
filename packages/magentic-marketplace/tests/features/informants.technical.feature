Feature: Informants, technical rules
  Backs "Informants". The undercover pick is what later scoring leaves out of its own
  report and compares against, so it has to be on disk, drawn only from sellers, and
  impossible to configure into a run where it has nothing to watch.

  Rule: The undercover pick is recorded in the run folder

    Scenario: An S4 run writes its pick before round 1
      Given a config.yml with the undercover informant on and informant seed 7
      When the run starts
      Then the run folder holds informant.yml with the seed 7 and the picked seller's id

    Scenario: A run without the switch writes no pick
      Given a config.yml with the channel on and the undercover informant off
      When the run starts
      Then the run folder holds no informant.yml

  Rule: The undercover seller is always one of the run's sellers

    Scenario: Every seed picks a seller of the run
      Given the sellers of "mexican_3_9"
      When the undercover seller is picked with seeds 0 to 99
      Then every pick is one of those sellers

    Scenario: The pick does not depend on the order sellers are loaded in
      Given the sellers of "mexican_3_9" in file order and in reverse order
      When the undercover seller is picked with informant seed 7 from each
      Then both picks are the same seller

    Scenario: Each repetition of an experiment plants its own undercover seller
      Given an experiment with an S4 arm, base seed 7 and 3 repetitions
      When its runs are planned
      Then repetitions 1, 2 and 3 use informant seeds 7, 8 and 9

  Rule: The undercover informant needs the seller channel

    Scenario: A config with the informant on and the channel off is rejected
      Given a config.yml with the undercover informant on and the channel off
      When it is loaded
      Then loading fails with an error naming the seller channel

    Scenario: With the switch off every seller prompt is unchanged
      Given an S3 run of "mexican_3_9"
      When any seller's talk or pricing prompt is rendered
      Then it is the same text the code rendered before the switch existed

  Rule: The questioned view is the seller's last answered pricing call of the round

    Scenario: A retried price list shows the last attempt
      Given "business_0004" was asked for its round-2 prices twice, the first list rejected
      When it is questioned about round 2
      Then it is shown the second pricing prompt and the answer to it

  Rule: An accusation must name a seller of the run

    Scenario: An unknown seller id is sent back
      Given "business_0007" accuses "business_0099", which is not a seller of the run
      When its report is checked
      Then "business_0007" is asked again, told that "business_0099" is not a seller in this market

  Rule: After the last retry the report keeps only its valid accusations

    Scenario: An accusation still without a quote is dropped
      Given "business_0007" accuses "business_0011" without a quote on every attempt
      And accuses "business_0004", quoting "deal, $12 it is"
      When its report is checked with 3 retries
      Then the saved report lists only "business_0004"
      And records "business_0011" as dropped after 4 attempts

    Scenario: A question the LLM fails on every attempt is saved as an error
      Given the LLM fails every attempt for "business_0004" about round 2
      When the sellers are questioned
      Then its report records the error and no accusations
      And every other report is saved

  Rule: Questions run concurrently within a limit

    Scenario: Never more questions in flight than the limit
      Given 18 questions and a concurrency limit of 4
      When the sellers are questioned
      Then at most 4 questions wait on the LLM at once
      And all 18 reports are saved

  Rule: Any finished run can be questioned

    Scenario: A run without channel messages is questioned
      Given a finished S2 run of "mexican_3_9" with 2 rounds and no channel messages
      When the sellers are questioned
      Then there are 9 pressured reports

    Scenario: A single-round run has nothing to question
      Given a finished S1 run
      When the sellers are questioned
      Then the reports file is empty and the command exits 0

    Scenario: An undercover pick that names no seller of the run is rejected
      Given an informant.yml naming "business_0099" next to a run without that seller
      When the sellers are questioned
      Then the command exits 1 naming "business_0099"

  Rule: Questioning changes nothing in the run

    Scenario: The run's database is untouched
      Given a finished S3 run
      When the sellers are questioned
      Then the run's logs and actions hold the same rows as before
