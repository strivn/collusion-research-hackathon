Feature: Prompt additions, technical rules
  Backs "Prompt additions from the run config". Sellers are built anew every round, a
  rejected price list is asked again with its errors appended, and questioning runs after
  the run from the run folder alone, so the additions and the replaced instruction have to
  survive all three.

  Rule: Every round builds every seller with the run's additions and undercover instruction

    Scenario: A 3-round run hands the same additions to every seller every round
      Given an Undercover config for "mexican_3_9" with 3 rounds, informant seed 7 and one addition for all sellers
      When the run starts
      Then in rounds 1, 2 and 3 all 9 sellers are built with that addition
      And the undercover seller is built with the configured undercover instruction in every round

  Rule: A retried prompt keeps its additions and the retry errors stay last

    Scenario: A rejected price list is asked again with the addition still in place
      Given an addition for all sellers in pricing at the top and one after the goal
      And a seller whose first price list leaves out a menu item
      When it is asked again
      Then the retry prompt opens with the top addition and contains the after_goal one
      And it ends with the rejection errors

  Rule: The run folder records the additions and the instruction, and questioning reads them back

    Scenario: config.yml records the additions and the undercover instruction
      Given a config with two additions and an undercover instruction
      When the run starts
      Then the run folder's config.yml has both additions in config order and the instruction

    Scenario: Questioning a run folder without config.yml uses the default instruction
      Given a run folder with informant.yml and no config.yml
      When the undercover seller is questioned
      Then its question opens with the default undercover instruction
