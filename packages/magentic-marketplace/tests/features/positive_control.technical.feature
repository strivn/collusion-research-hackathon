Feature: Positive control, technical rules
  Backs "Positive control". Sellers are built anew every round, so the instruction has to
  reach every round's sellers, and the saved config has to say the run was a positive
  control.

  Rule: Every round builds every seller except the undercover one with the instruction

    Scenario: A 3-round Positive control run instructs the same 8 sellers every round
      Given a Positive control config for "mexican_3_9" with 3 rounds and informant seed 7
      When the run starts
      Then in rounds 1, 2 and 3 the 8 sellers other than the undercover one get the instruction
      And the undercover seller does not

    Scenario: A Positive control run without an informant instructs every seller
      Given a config.yml with the channel and the positive control on and no informant
      When the run starts
      Then all 9 sellers get the instruction in every round

    Scenario: Without the switch no seller is built with the instruction
      Given an Undercover config for "mexican_3_9" with 3 rounds
      When the run starts
      Then no seller gets the instruction in any round

  Rule: The run folder says the run was a positive control

    Scenario: config.yml records the switch
      Given a Positive control config
      When the run starts
      Then the run folder's config.yml has positive_control true
