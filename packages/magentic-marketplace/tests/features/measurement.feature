Feature: Measurement output
  After a run, every customer's outcome in every round is split into what the customer
  keeps, what the seller gets, and what is lost, using the stock utility rule applied per
  round. Seller prices and sales are tabled per round. Token use is logged as input and
  output separately.

  Rule: Each customer-round is scored with the stock utility rule

    Scenario: A matching sale splits V between customer and seller
      Given customer "customer_0001" with reference $13.48, so V = $26.96
      When in round 2 it pays $15.00 for an order that meets its request
      Then its round-2 row shows customer share $11.96, seller share $15.00, lost $0.00, outcome "sale"

    Scenario: A mismatched sale costs the customer the price
      Given customer "customer_0001" with V = $26.96
      When in round 2 it pays $15.00 to a seller missing a required amenity
      Then its round-2 row shows customer share -$15.00, seller share $15.00, lost $26.96, outcome "mismatch"

    Scenario: No sale loses V
      Given customer "customer_0001" with V = $26.96
      When in round 2 it pays nothing
      Then its round-2 row shows customer share $0.00, seller share $0.00, lost $26.96, outcome "no sale"

    Scenario: Two payments, one matching, count V once and both prices
      Given customer "customer_0001" with V = $26.96
      When in round 2 it pays $15.00 for a matching order and $5.00 for a mismatched one
      Then its round-2 row shows customer share $6.96, seller share $20.00, lost $0.00

    Scenario: Refused over-budget payments are counted
      Given customer "customer_0001" with a budget of $16.85
      When in round 2 one payment of $18.00 is refused and no other payment is made
      Then its round-2 row shows outcome "no sale" and 1 refused payment

  Rule: Customer share, seller share and lost add up to total V in every round

    Scenario: The round totals balance
      Given a finished run of "mexican_3_9" with 2 rounds
      When the round totals are computed
      Then in each round customer share + seller share + lost equals the sum of every customer's V

  Rule: Every customer-round row carries the inputs needed to read it

    Scenario: The customer-round table columns
      Given a finished run
      When the customer-round table is written to the run folder
      Then each row has round, customer, reference, V, m, budget, seller paid, price paid, needs met, customer share, seller share, lost, outcome and refused payments

  Rule: Seller prices and sales are tabled per seller per round

    Scenario: The seller-round table
      Given seller "business_0001" priced "Empanadas" at $10.50 and sold 2 in round 2
      When the seller-round table is written to the run folder
      Then it has a row for "business_0001", round 2, "Empanadas", price $10.50, units sold 2, revenue $21.00

  Rule: With all features off, the scores equal stock analytics

    Scenario: S1 scores match the stock analyze report
      Given a finished one-round run with every feature off
      When the customer-round table is written
      Then each customer's customer share equals its utility in the stock "analyze" report
