Feature: Run config
  A run can be started from a config.yml. The settings the run actually used are
  saved into the run folder, so every result says how it was produced.

  Rule: A run started from a config file uses the settings in that file

    Scenario: The config names the data and the experiment
      Given a config.yml with data "data/mexican_3_9" and experiment name "cfg_test"
      When a run is started from that config
      Then the run loads 3 customers and 9 businesses
      And its Postgres schema is "cfg_test"

    Scenario: A setting missing from the config takes its stock default
      Given a config.yml that does not set customer max steps
      When a run is started from that config
      Then customer max steps is the stock default

  Rule: The settings a run used are saved into the run folder

    Scenario: The resolved config is written to the run folder
      Given a config.yml that sets only data and experiment name
      When a run is started from that config in folder "research/results/cfg_test"
      Then "research/results/cfg_test/config.yml" lists every setting the run used, defaults included
  Rule: A config with an unknown setting is rejected before the run starts

    Scenario: A misspelled setting stops the run
      Given a config.yml containing the setting "custmer_max_steps"
      When a run is started from that config
      Then the run does not start
      And the error names "custmer_max_steps"

  Rule: Runs without a config behave as stock

    Scenario: The stock command still works
      When "magentic-marketplace run data/mexican_3_9 --experiment-name stock_test" is started
      Then the run behaves as before this feature
