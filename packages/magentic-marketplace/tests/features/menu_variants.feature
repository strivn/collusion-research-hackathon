Feature: Menu variants
  A script copies a dataset and trims every seller's menu, so a run can compare the full
  menu with one that lists only what customers ask for, plus a few unwanted items.

  Fixture dataset: customer_0001 asks for "Tacos" and "Flan"; customer_0002 asks for
  "Salsa". Each customer has a request, amenity features and its own menu prices.
    business_0001 sells "Burrito", "Tacos", "Churros", "Flan", "Nachos", "Elote"
    business_0002 sells "Salsa", "Quesadilla"
    business_0003 sells "Tamales", "Tacos", "Mole"

  Rule: The demanded variant keeps on each seller's menu only items some customer asks for

    Scenario: Unwanted items leave every menu
      Given the fixture dataset
      When the variant is written with 0 extra items
      Then business_0001's menu is "Tacos", "Flan", in that order, at its own prices
      And business_0002's menu is "Salsa"
      And business_0003's menu is "Tacos"

  Rule: The demanded+N variant also keeps N other items of the seller's own menu, picked by seed

    Scenario: The same seed keeps the same extra items
      Given the fixture dataset
      When the variant is written twice with 2 extra items and seed 7
      Then both copies have the same menus
      And business_0001 keeps "Tacos", "Flan" and 2 of "Burrito", "Churros", "Nachos", "Elote"
      And its items keep their original order

    Scenario: A seller with fewer than N other items keeps all of them
      Given the fixture dataset
      When the variant is written with 2 extra items and seed 7
      Then business_0002's menu is "Salsa", "Quesadilla"

  Rule: Everything except the menus is copied unchanged (customers, amenities, ratings, baseline utilities)

    Scenario: Customers, seller details and baseline utilities are the same as the source
      Given the fixture dataset
      When the variant is written with 2 extra items and seed 7
      Then every customer file and baseline_utilities.json is byte-for-byte the source's
      And every business keeps every field other than menu_features, in the same order
