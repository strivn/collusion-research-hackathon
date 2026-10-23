Feature: Measurement output
  After a run, every customer's outcome in every round is split into what the customer
  keeps, what the seller gets, and what is lost, using the stock utility rule applied per
  round. Seller prices and sales are tabled per round. Token use is logged as input and
  output separately.

  Rule: Each LLM call logs input and output tokens separately

    Scenario: Token counts are split
      When any agent makes an LLM call
      Then its log entry holds input tokens and output tokens as separate numbers
