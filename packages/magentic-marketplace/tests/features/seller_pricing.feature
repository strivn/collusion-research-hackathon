Feature: Seller pricing
  With pricing on, each seller sets one price per menu item at the start of a round,
  aiming to maximize its revenue, and must quote those prices to every customer that
  round. With pricing off, sellers quote menu prices exactly as stock.
  (F2 runs one round; F3 repeats the pricing step every round.)

  Rule: With pricing off, sellers behave exactly as stock

    Scenario: The response prompt of a seller with pricing off is the stock prompt
      Given a config.yml with seller pricing off
      And seller "business_0001" from "mexican_3_9"
      When its response prompt is rendered
      Then the prompt is identical to the stock response prompt

    Scenario: With pricing off, no pricing step runs
      Given a config.yml with seller pricing off
      When a run starts
      Then no seller makes a pricing call

  Rule: With pricing on, each seller sets a price for every menu item before serving customers

    Scenario: The pricing prompt shows current prices and the revenue goal
      Given a config.yml with seller pricing on
      And seller "business_0001" whose menu lists "Savory Pumpkin Empanadas" at $9.49
      When its pricing prompt is rendered for round 1
      Then the prompt lists "Savory Pumpkin Empanadas" with current price $9.49
      And the prompt says its goal is to maximize total revenue over all rounds

    Scenario: The price list is set before the seller answers any customer
      Given a config.yml with seller pricing on
      When a run starts
      Then every seller's pricing call finishes before that seller sends its first proposal

    Scenario: A price list missing an item is retried
      Given a seller with menu items "Empanadas" and "Agua Fresca"
      When its pricing step returns a price for "Empanadas" only
      Then the seller is asked again, with the error naming "Agua Fresca"

    Scenario: A price list with a zero or negative price is retried
      Given a seller with menu item "Empanadas"
      When its pricing step returns "Empanadas" at $0.00
      Then the seller is asked again, with the error naming "Empanadas"

    Scenario: After failed retries the seller keeps its current prices
      Given a seller whose current price for "Empanadas" is $9.49
      When its pricing step returns an invalid price list on every retry
      Then the seller's price for "Empanadas" stays $9.49
      And the failure is logged

  Rule: With pricing on, sellers quote their round prices to every customer

    Scenario: The response prompt shows round prices and the revenue goal
      Given a seller whose round price for "Savory Pumpkin Empanadas" is $10.50 and menu price is $9.49
      When its response prompt is rendered
      Then the prompt lists "Savory Pumpkin Empanadas" at $10.50
      And the prompt does not list $9.49
      And the prompt says its goal is to maximize total revenue over all rounds

    Scenario: A proposal at the round price is accepted
      Given a seller whose round price for "Empanadas" is $10.50
      When it proposes 2 "Empanadas" at $10.50 each, total $21.00
      Then the proposal is sent

    Scenario: A proposal at any other price is retried
      Given a seller whose round price for "Empanadas" is $10.50
      When it proposes 1 "Empanadas" at $9.49
      Then the seller is asked again, with the error naming the round price $10.50

  Rule: Every price list is logged with seller and round

    Scenario: A price list appears in the run log
      Given a config.yml with seller pricing on
      When seller "business_0001" sets "Empanadas" to $10.50 in round 1
      Then the run log holds a price-list entry with seller "business_0001", round 1 and "Empanadas" at $10.50
