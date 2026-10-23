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

  Rule: The undercover informant needs the seller channel

    Scenario: A config with the informant on and the channel off is rejected
      Given a config.yml with the undercover informant on and the channel off
      When it is loaded
      Then loading fails with an error naming the seller channel

    Scenario: With the switch off every seller prompt is unchanged
      Given an S3 run of "mexican_3_9"
      When any seller's talk or pricing prompt is rendered
      Then it is the same text the code rendered before the switch existed
