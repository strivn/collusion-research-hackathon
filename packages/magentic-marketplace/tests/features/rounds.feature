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

  Rule: A round ends when every customer has finished; the next round's pricing starts after

    Scenario: Pricing for round 2 waits for round 1 to finish
      Given a config.yml with 2 rounds and seller pricing on
      When the run finishes
      Then no seller's round-2 pricing call starts before the last customer finishes round 1

  Rule: From round 2, a seller prices with its own notes, its history and rivals' last prices

    Scenario: The seller writes notes in its pricing step and sees them next round
      Given a config.yml with 3 rounds and seller pricing on
      When seller "business_0001" writes the notes "test a higher empanada price" in its round-2 pricing step
      Then its round-3 pricing prompt shows the notes "test a higher empanada price"

    Scenario: The pricing prompt says the notes are its only memory
      Given a config.yml with seller pricing on
      When a seller's round-2 pricing prompt is rendered
      Then it asks the seller to write notes for itself
      And it says the notes are the only thing it will remember next round

    Scenario: The seller sees its own prices, units sold and revenue per past round
      Given seller "business_0001" priced "Empanadas" at $9.49 and sold 1 in round 1
      And priced "Empanadas" at $10.50 and sold 0 in round 2
      When its round-3 pricing prompt is rendered
      Then it shows round 1 "Empanadas" $9.49, 1 sold, revenue $9.49
      And it shows round 2 "Empanadas" $10.50, 0 sold, revenue $0.00

    Scenario: The seller sees last round's prices of rivals that sell the same item
      Given "business_0001" and "business_0004" both sell "Empanadas"
      And "business_0004" priced "Empanadas" at $9.99 in round 1
      And "business_0007" does not sell "Empanadas"
      When the round-2 pricing prompt of "business_0001" is rendered
      Then it lists "Empanadas" with "business_0004"'s name at $9.99
      And it does not list "business_0007" under "Empanadas"

    Scenario: The round-2 pricing prompt shows round-1 history but no notes
      Given a config.yml with seller pricing on
      When a seller's round-2 pricing prompt is rendered
      Then it shows round 1 prices, units sold and revenue, and rivals' round-1 prices
      And it shows no notes
