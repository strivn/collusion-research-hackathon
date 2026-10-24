Feature: Menu variants, technical
  Backs menu_variants.feature: guards that stop a broken or mixed dataset from being
  written.

  Rule: An existing destination folder is refused, so a variant never mixes with older files

    Scenario: Writing into a folder that exists fails and leaves it untouched
      Given the fixture dataset
      And a destination folder that already holds a file
      When the variant is written there
      Then it fails with an error naming the folder
      And the folder holds only that file

  Rule: A seller left with no items is refused, since it could sell nothing

    Scenario: A seller with no demanded items and no extras fails the whole write
      Given the fixture dataset with business_0003 selling only "Mole"
      When the variant is written with 0 extra items
      Then it fails with an error naming business_0003
      And no destination folder is written
