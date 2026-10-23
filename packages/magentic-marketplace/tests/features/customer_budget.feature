Feature: Customer budget
  A customer may carry a budget for its whole order. It is told the number, and
  its agent refuses to pay any proposal above it. Without a budget nothing changes.

  Rule: A customer without a budget behaves exactly as stock

    Scenario: The prompt of a customer without a budget is the stock prompt
      Given customer "customer_0001" from "mexican_3_9" with no budget
      When its system prompt is rendered
      Then the prompt is identical to the stock prompt for that customer

    Scenario: A customer without a budget can pay any proposal
      Given a customer with no budget
      And a pending proposal from "business_0001" totalling $50.00
      When the customer pays that proposal
      Then a payment of $50.00 is sent to "business_0001"

  Rule: A customer with a budget is told its budget and to keep it private

    Scenario: The prompt states the budget and asks to keep it private
      Given customer "customer_0001" with a budget of $16.85
      When its system prompt is rendered
      Then the prompt says the budget for the whole order is $16.85
      And the prompt says not to pay more than the budget
      And the prompt says not to tell businesses the budget

  Rule: A payment above the budget is refused and never reaches the seller

    Scenario: A proposal above the budget is refused
      Given a customer with a budget of $16.85
      And a pending proposal from "business_0001" totalling $18.00
      When the customer pays that proposal
      Then no payment is sent to "business_0001"
      And the customer sees that the payment was refused because $18.00 exceeds its budget of $16.85
      And the proposal is still pending

    Scenario: A proposal exactly at the budget is paid
      Given a customer with a budget of $16.85
      And a pending proposal from "business_0001" totalling $16.85
      When the customer pays that proposal
      Then a payment of $16.85 is sent to "business_0001"

    Scenario: After a refusal the customer can pay a cheaper proposal
      Given a customer with a budget of $16.85
      And a pending proposal from "business_0001" totalling $18.00
      And a pending proposal from "business_0002" totalling $15.00
      When the customer pays the "business_0001" proposal
      And then pays the "business_0002" proposal
      Then only a payment of $15.00 to "business_0002" is sent

  Rule: Budgets come from the run config; a config without a budget section gives no budgets

    Scenario: A config without a budget section gives every customer no budget
      Given a config.yml for "mexican_3_9" with no budget section
      When a run is started from that config
      Then no customer has a budget

    Scenario: The budgets a run used are written to the run folder
      Given a config.yml for "mexican_3_9" with budget multipliers 1.25, 1.5, 2.0 and seed 7
      When a run is started from that config in folder "research/results/budget_test"
      Then "research/results/budget_test/budgets.yml" lists each customer's id, m and budget

  Rule: Budgets are m x reference with m drawn balanced from a seed

    Scenario: A budget is m times the reference, rounded to cents
      Given a customer whose menu_features are 3.99 and 9.49
      When it is assigned m = 1.25
      Then its budget is $16.85

    Scenario: Multipliers are spread evenly over customers
      Given the 10 customers of "mexican_10_30"
      When budgets are assigned with multipliers 1.25, 1.5, 2.0 and seed 7
      Then the number of customers per multiplier is 4, 3 and 3 in some order

    Scenario: The same seed gives the same budgets
      Given the 10 customers of "mexican_10_30"
      When budgets are assigned twice with seed 7
      Then both assignments give every customer the same budget
