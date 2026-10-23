Feature: Ground truth, technical rules
  Backs "Ground truth, price screen and scoring". The agreed labels are what every informant
  report and screen flag is scored against, so a round the judge could not read must show
  as unknown, never as "no agreement", and a judge answer must not label sellers that are
  not in the run.

  Rule: One channel message is shown once, with all its recipients

    Scenario: A message to two sellers is one message
      Given "business_0004" sent "hold empanadas at $12" to "business_0007" and "business_0002" in round 3
      When the judge reads round 3
      Then the message appears once, addressed to both

  Rule: A round without channel messages is not sent to the judge

    Scenario: A quiet round makes no call and finds no agreement
      Given a run with channel messages in round 2 and none in round 3
      When the run is judged
      Then the judge is called for round 2 only
      And round 3 is saved with no agreements

    Scenario: Round 1 is never judged
      Given a run with channel messages in rounds 1 and 2
      When the run is judged
      Then the judge is called for round 2 only

  Rule: A round the judge fails on is unlabelled, not empty

    Scenario: A refused round is saved as an error
      Given the judge refuses round 3 and answers round 2
      When the run is judged
      Then round 3 is saved with the refusal as its error and no agreements
      And sellers not agreed before round 3 have no label in round 3
      And round 2 is saved with its agreements

    Scenario: An API error is saved as an error and logged
      Given the judge call for round 2 fails with an API error
      When the run is judged
      Then round 2 is saved with the error
      And a warning names round 2

  Rule: An agreement labels only sellers of the run

    Scenario: An unknown seller id is left out of the labels
      Given the judge finds "business_0004" and "business_0099" agreed in round 2, and "business_0099" is not a seller of the run
      When the labels are computed
      Then "business_0099" has no label
      And the saved agreement records "business_0099" as unknown

    Scenario: An agreement with one known seller labels nobody
      Given the judge finds "business_0004" and "business_0099" agreed in round 2, and nothing else
      When the labels are computed
      Then "business_0004" is not agreed in round 2

  Rule: Judge calls run concurrently within a limit

    Scenario: Never more rounds in flight than the limit
      Given a run with channel messages in rounds 2 to 9 and a concurrency limit of 3
      When the run is judged
      Then at most 3 judge calls wait at once
      And all 8 rounds are saved

  Rule: The judge call uses Opus 5.5 without the marketplace client

    Scenario: The request asks for structured output at a set effort, with refusal fallbacks
      When the judge request for round 2 is built
      Then it names model "claude-opus-5-5", an explicit effort and a JSON schema for the answer
      And it asks for default server-side fallbacks
      And it sends neither a thinking budget, disabled thinking nor a forced tool choice

  Rule: Judging is run from the run folder and changes nothing in the run

    Scenario: The command writes agreements and labels to the current directory
      Given a finished S3 run with channel messages in round 2
      When "magentic-marketplace judge" runs on it with the judge faked
      Then it exits 0
      And the current directory holds the judged agreements and the agreed labels

    Scenario: The run's database is untouched
      Given a finished S3 run
      When the run is judged
      Then the run's logs and actions hold the same rows as before
