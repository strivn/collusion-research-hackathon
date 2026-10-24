Feature: Seller conduct, technical rules
  Backs "Seller conduct on the channel". The labels are compared across paired runs, so a
  post the labeller could not read must show as unlabelled, never as ordinary, and a
  labeller answer must not invent posts. The labeller is the judge's Opus 5.5 call with a
  different answer schema.

  Rule: One post is one message as sent, with all its recipients

    Scenario: A post to two sellers is labelled once
      Given "business_0004" sent "hold empanadas at $12" to "business_0007" and "business_0002" in round 2
      When the labeller reads round 2
      Then the post appears once, addressed to both
      And its label rows list both recipients

  Rule: A round without posts is not sent to the labeller

    Scenario: A quiet round makes no call
      Given a run with channel posts in round 2 and none in round 3
      When the run's conduct is labelled
      Then the labeller is called for round 2 only

  Rule: A round the labeller fails on is unlabelled, not ordinary

    Scenario: A refused round leaves its posts unlabelled
      Given the labeller refuses round 3 and answers round 2
      When the run's conduct is labelled
      Then each round-3 post has one row with no label and no quote
      And round 2 has its labels
      And a warning names round 3

    Scenario: An API error leaves the round's posts unlabelled
      Given the labeller call for round 2 fails with an API error
      When the run's conduct is labelled
      Then each round-2 post has one row with no label and no quote
      And a warning names round 2

  Rule: A post the labeller leaves out is unlabelled

    Scenario: A skipped post has one row with no label
      Given round 2 has two posts and the labeller answers only the first
      When the run's conduct is labelled
      Then the second post has one row with no label and no quote

    Scenario: A post given no labels has one row with no label
      Given the labeller answers a round-2 post with an empty list of labels
      When the run's conduct is labelled
      Then that post has one row with no label and no quote

  Rule: An answer labels only posts of its round

    Scenario: A label for a post number the round does not have is dropped
      Given round 2 has two posts and the labeller also labels post 5
      When the run's conduct is labelled
      Then no row comes from post 5
      And a warning names round 2 and post 5

  Rule: A label given twice for one post is kept once

    Scenario: A repeated label keeps its first quote
      Given the labeller labels a post joins collusion twice, quoting "deal" then "$12 it is"
      When the run's conduct is labelled
      Then that post has one joins collusion row, quoting "deal"

  Rule: Rounds are labelled concurrently, with a cap

    Scenario: No more calls wait than the cap
      Given a run with channel posts in rounds 2 to 7
      When the run's conduct is labelled with at most 2 calls at once
      Then no more than 2 labeller calls wait at once
      And every round from 2 to 7 is labelled

  Rule: The undercover seller is read from the run folder's informant.yml

    Scenario: No informant.yml makes every seller ordinary
      Given a run folder without informant.yml
      When the run's conduct is labelled
      Then every row has role ordinary

    Scenario: An informant.yml naming no seller of the run stops the command
      Given informant.yml names "business_0099", which is no seller of the run
      When the conduct command runs
      Then it fails before any labeller call and names "business_0099"
      And it writes no files

  Rule: The labeller call is the judge's Opus 5.5 call with the conduct answer schema

    Scenario: The request asks for structured conduct labels
      When the labeller request for round 2 is built
      Then it names model "claude-opus-5-5" with the judge's effort, token limit and fallbacks
      And its JSON schema allows exactly the seven labels

    Scenario: A refusal stop reason is a refusal
      Given the labeller's response stops with reason "refusal"
      When the round is labelled
      Then the round is unlabelled with the refusal as its reason

  Rule: The labeller's model can be chosen per command, Opus 5.5 by default

    Scenario: conduct --model sends every request to that model
      Given a finished run with channel posts in rounds 2 and 3
      When "conduct --model claude-sonnet-5-5" runs
      Then every labeller request names model "claude-sonnet-5-5"

    Scenario: conduct --model applies to note labelling too
      When "conduct --source notes --model claude-sonnet-5-5" runs
      Then every labeller request names model "claude-sonnet-5-5"

    Scenario: Without --model the labeller uses Opus 5.5
      When "conduct" runs without --model
      Then every labeller request names model "claude-opus-5-5"

  Rule: Conduct is run from the run folder and changes nothing in the run

    Scenario: The command writes both files into the run folder
      Given a finished run and its run folder with informant.yml naming "business_0004"
      When "magentic-marketplace conduct" runs from the run folder against a fake labeller
      Then it exits 0 and writes conduct_labels.csv and conduct_counts.csv there
      And conduct_labels.csv has columns round, sender, recipients, turn, label, quote, role
      And conduct_counts.csv has columns seller_id, round, posts_sent, role

    Scenario: The run's database is untouched
      Given a finished run
      When "magentic-marketplace conduct" runs against it
      Then the run's database has the same rows as before

  Rule: A labels file that cannot be read stops the command before any call

    Scenario Outline: A bad labels file is refused
      Given a labels file that <problem>
      When the conduct command runs with that file
      Then it fails before any labeller call and names the problem
      And it writes no files

      Examples:
        | problem                              |
        | lists ordinary                       |
        | lists "coded" twice                  |
        | has an entry without a definition    |
        | is an empty list                     |

  Rule: The answer schema allows exactly the offered labels

    Scenario: The schema follows the labels file
      Given a labels file with the built-in six plus "coded" and "price_signal"
      When the labeller request for round 2 is built
      Then its JSON schema allows exactly those eight labels and ordinary

    Scenario: The example labels file offers the built-in seven plus coded and price_signal
      When experiments/collusion/conduct_labels_v2.yml is read
      Then it offers the built-in six, coded and price_signal, word for word, and then ordinary

  Rule: An answer with labels outside the offered ones is sent back at most three times

    Scenario: A labeller that keeps using an unknown label leaves the round unlabelled
      Given the labeller labels a round-2 post "coded" on every call, and coded is not offered
      When the run's conduct is labelled
      Then the labeller is called 4 times for round 2
      And each round-2 post has one row with no label and no quote
      And a warning names round 2 and "coded"

  Rule: Notes are read from the run's price lists, one note per seller per round from 2

    Scenario: The latest price list of a seller and round holds its notes
      Given "business_0004" has two price lists for round 2, noting "first" then "second"
      When the run's notes are labelled
      Then the labeller is shown "second" for business_0004 in round 2, and not "first"

    Scenario: A round where nobody wrote notes makes no call
      Given notes in round 2 and only empty notes in round 3
      When the run's notes are labelled
      Then the labeller is called for round 2 only

    Scenario: A round the labeller fails on leaves its notes unlabelled
      Given the labeller call for round 2's notes fails with an API error
      When the run's notes are labelled
      Then each round-2 note has one row with no label and no quote
      And a warning names round 2

    Scenario: Note rounds are labelled concurrently, with a cap
      Given notes in rounds 2 to 7
      When the run's notes are labelled with at most 2 calls at once
      Then no more than 2 labeller calls wait at once
      And every round from 2 to 7 is labelled

  Rule: `conduct --source notes` writes only the notes labels file

    Scenario: The notes command writes conduct_notes_labels.csv into the run folder
      Given a finished Pricing Only run and its run folder
      When "magentic-marketplace conduct --source notes --labels <file>" runs from the run folder against a fake labeller
      Then it exits 0 and writes conduct_notes_labels.csv there
      And it writes neither conduct_labels.csv nor conduct_counts.csv
