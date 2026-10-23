Feature: Rounds
  A run can repeat the market for several rounds with the same customers. Customers
  start every round fresh with the same order and budget. Sellers carry their own notes,
  their sales history and rivals' last prices into each round's pricing step, in the style
  of Fish, Gonczarowski & Shorrer (2024). One round reproduces stock.

  Rule: A run without a rounds setting runs one round, as stock

    Scenario: No rounds setting gives a single round
      Given a config.yml for "mexican_3_9" with no rounds setting
      When the run finishes
      Then every customer has shopped exactly once

  Rule: Each round, every customer shops once, fresh, with the same order and budget

    Scenario: Five rounds means five shopping trips per customer
      Given a config.yml for "mexican_3_9" with 5 rounds
      When the run finishes
      Then every customer has shopped exactly 5 times

    Scenario: A customer starts each round with no memory of earlier rounds
      Given a config.yml with 2 rounds
      And customer "customer_0001" paid "business_0001" in round 1
      When customer "customer_0001" starts round 2
      Then its prompt holds no trace of round 1

    Scenario: A customer's budget is the same in every round
      Given a config.yml with 3 rounds and budgets on
      When customer "customer_0001" has a budget of $16.85 in round 1
      Then its budget is $16.85 in rounds 2 and 3

  Rule: Seller conversations start empty every round

    Scenario: A seller does not see last round's chat with a customer
      Given a config.yml with 2 rounds
      And seller "business_0001" talked with "customer_0001" in round 1
      When "customer_0001" messages "business_0001" in round 2
      Then the seller's response prompt shows only round 2's conversation

  # From the Rule "From round 2, a seller prices with its own notes..."; the rest of that Rule comes with seller memory.
    Scenario: No prompt says how many rounds the run has
      Given a config.yml with 5 rounds and seller pricing on
      When any seller or customer prompt is rendered
      Then it does not state the total number of rounds

  Rule: Every logged action and LLM call carries its round number

    Scenario: Actions in round 2 are tagged round 2
      Given a config.yml with 2 rounds
      When customer "customer_0001" pays a proposal in round 2
      Then the logged payment carries round 2
