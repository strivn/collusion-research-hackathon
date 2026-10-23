Feature: Round start, technical rules
  Backs "Rounds", "Seller pricing" and "Seller channel". Sellers talk and price while they
  start up; customers must not shop before that is done, or customers in talk runs wait
  longer than customers in no-talk runs and the comparison picks up the difference.

  Rule: Customers start a round only after every seller of that round has finished starting up

    Scenario: Customers wait for sellers that are still pricing
      Given a round with 2 sellers whose start-up takes 0.2 seconds and 2 customers
      When the round runs
      Then no customer takes its first step before the last seller is ready for customers

    Scenario: Each round waits for its own sellers
      Given a run with 2 rounds
      When round 2 runs
      Then no round-2 customer takes its first step before every round-2 seller is ready

  Rule: A seller that cannot finish starting up does not hold customers forever

    Scenario: A seller whose start-up fails releases its place
      Given a round with 2 sellers, one of whose start-up raises an error
      When the round runs
      Then customers start once the other seller is ready

    Scenario: Customers stop waiting after the time limit
      Given a round with a seller whose start-up never finishes and a wait limit of 0.1 seconds
      When the round runs
      Then customers start after 0.1 seconds
      And each customer logs a warning that it started before every seller was ready
