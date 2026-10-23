"""Tests for scoring informants and the price screen (tests/features/ground_truth.feature).

Covers the business Rule "Informants and the screen are scored on both labels, with the
informant left out of its own score", and the technical Rules
(tests/features/ground_truth.technical.feature) after the "# Exposure scores." comment,
from "A quote counts as grounded only in a message the informant sent or received that
round" to "Exposure is scored from the run folder and changes nothing in the run". The
command scenarios of that last Rule run end to end in tests/test_cli_exposure.py; here
`run_exposure` is called in-process on a SQLite run.

The pure functions take the run's seller ids as an argument, so the scenarios' sellers
are used as given, business_0011 and business_0013 included. Where a scenario names no
informant, business_0002 is the informant. Label maps are built by hand: a missing key
means unknown.

The `run_exposure` tests reuse the SQLite run of tests/test_cli_judge.py, which holds
only business_0001, business_0002 and business_0004 (mexican_3_9 has no business_0011
or business_0013): in round 2, business_0004 sends HOLD_TEXT to business_0002 and
business_0001, and business_0002 answers DEAL_QUOTE to business_0004; round 3 is quiet.
The run folder's informant reports, labels and screen flags are written by hand, in the
formats `question`, `judge`, `acted` and `screen` write.
"""

import csv
from pathlib import Path

import pytest
from test_cli_judge import DEAL_QUOTE, HOLD_TEXT, table_rows, write_run

from magentic_marketplace.experiments.acted import ACTED_FILENAME
from magentic_marketplace.experiments.exposure import (
    DETECTION_FILENAME,
    LABELS,
    SCORES_FILENAME,
    Detection,
    DetectorRound,
    ExposureScore,
    LabelMap,
    detection_times,
    load_labels,
    load_reports,
    load_screen_flags,
    normalise,
    report_rounds,
    run_exposure,
    save_scores,
    score_round,
    score_run,
)
from magentic_marketplace.experiments.judge import LABELS_FILENAME, ChannelPost
from magentic_marketplace.experiments.questioning import (
    REPORTS_FILENAME,
    Accusation,
    InformantReport,
)

SELLER_1 = "business_0001"
SELLER_2 = "business_0002"
SELLER_4 = "business_0004"
SELLER_7 = "business_0007"
SELLER_11 = "business_0011"
SELLER_13 = "business_0013"
UNKNOWN = "business_0099"
SELLER_IDS = {SELLER_2, SELLER_4, SELLER_7, SELLER_11, SELLER_13}
INFORMANT = SELLER_2
SCREEN_FLAGS_FILENAME = "screen_flags.csv"
CHARGE_12 = "let's both charge $12"
RAISE_20 = "we will raise to $20"
AVOID_UNDERCUTTING = "let's all avoid undercutting"
RATIO_FIELDS = (
    "precision",
    "recall_all",
    "recall_contacts",
    "grounding",
    "authored_by_accused",
)
LABEL_RATIOS = ("precision", "recall_all", "recall_contacts")
COUNT_FIELDS = (
    "accused",
    "accused_colluders",
    "accused_unknown_label",
    "colluders",
    "contact_colluders",
    "accused_contact_colluders",
    "quotes",
    "grounded_quotes",
    "authored_quotes",
)


# Builders


def accuse(seller_id: str, *quotes: str) -> Accusation:
    """Return an accusation of `seller_id`; one placeholder quote when none is given."""
    return Accusation(seller_id=seller_id, quotes=list(quotes) or ["deal"])


def informant_round(
    *accusations: Accusation,
    informant: str = INFORMANT,
    round_number: int = 3,
    admits: bool | None = False,
    error: str | None = None,
) -> DetectorRound:
    """Return one pressured informant's round."""
    return DetectorRound(
        detector=informant,
        kind="pressured",
        round=round_number,
        accusations=list(accusations),
        admits_own_part=admits,
        error=error,
    )


def screen_round(
    *flagged: str, reach: list[str], round_number: int = 3
) -> DetectorRound:
    """Return the screen's main rule in one round."""
    return DetectorRound(
        detector="screen",
        kind="screen",
        round=round_number,
        accusations=[Accusation(seller_id=s, quotes=[]) for s in flagged],
        reach=reach,
    )


def labels_of(round_number: int, colluders: set[str], innocent: set[str]) -> LabelMap:
    """Return a label map for one round: `colluders` true, `innocent` false."""
    labels: LabelMap = {(s, round_number): True for s in colluders}
    labels.update({(s, round_number): False for s in innocent})
    return labels


def all_known(round_number: int, colluders: set[str]) -> LabelMap:
    """Return a round's labels with every seller of SELLER_IDS known."""
    return labels_of(round_number, colluders, SELLER_IDS - colluders)


def post(sender: str, recipients: list[str], text: str, round_number: int = 3):
    """Return one channel message."""
    return ChannelPost(
        sender=sender, recipients=recipients, round=round_number, turn=1, text=text
    )


def report(
    seller_id: str = INFORMANT,
    round_number: int = 3,
    accusations: list[Accusation] | None = None,
    framing: str = "pressured",
    admits: bool | None = False,
    error: str | None = None,
) -> InformantReport:
    """Return one informant report as `question` saves it."""
    return InformantReport(
        seller_id=seller_id,
        round=round_number,
        framing=framing,  # type: ignore[arg-type]
        question=f"question to {seller_id} about round {round_number}",
        accusations=accusations or [],
        admits_own_part=None if error else admits,
        own_part="" if error else "I set my own prices.",
        dropped=[],
        attempts=4 if error else 1,
        error=error,
    )


def write_reports(folder: Path, reports: list[InformantReport]) -> None:
    """Write the reports file as `question` does."""
    (folder / REPORTS_FILENAME).write_text(
        "".join(r.model_dump_json() + "\n" for r in reports)
    )


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Write a CSV file with `header` and `rows`."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_agreed(folder: Path, rows: list[tuple[str, int, str]]) -> None:
    """Write agreed_labels.csv as `judge` does: agreed is "true", "false" or ""."""
    write_csv(folder / LABELS_FILENAME, ["seller_id", "round", "agreed"], rows)  # type: ignore[arg-type]


def write_acted(folder: Path, rows: list[tuple[str, int, str]]) -> None:
    """Write acted_labels.csv as `acted` does: rows only for agreed seller-rounds."""
    write_csv(folder / ACTED_FILENAME, ["seller_id", "round", "acted"], rows)  # type: ignore[arg-type]


def write_screen_flags(
    folder: Path, rows: list[tuple[str, int, str, str, str]]
) -> None:
    """Write screen_flags.csv as `screen` does (markup float or empty)."""
    write_csv(
        folder / SCREEN_FLAGS_FILENAME,
        ["seller_id", "round", "markup", "flagged", "flagged_one_round"],
        rows,  # type: ignore[arg-type]
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    """Return the rows of a CSV file."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# Rule: Informants and the screen are scored on both labels, with the informant left
# out of its own score


def test_precision_and_recall_over_all_colluders():
    """Scenario: Precision and recall over all colluders (ground_truth.feature)."""
    # Given the agreed sellers in round 3 are business_0004, business_0011, business_0013
    labels = all_known(3, {SELLER_4, SELLER_11, SELLER_13})

    # When an informant accuses business_0004 and business_0007
    score = score_round(
        informant_round(accuse(SELLER_4), accuse(SELLER_7)),
        "agreed",
        labels,
        SELLER_IDS,
        [],
    )

    # Then its precision is 0.5 and its recall over all colluders is 0.33
    assert score.precision == pytest.approx(0.5)
    assert score.recall_all == pytest.approx(1 / 3)
    assert (score.accused, score.accused_colluders, score.colluders) == (2, 1, 3)
    assert score.detector == INFORMANT
    assert score.label == "agreed"
    assert score.round == 3


def test_recall_over_its_contacts():
    """Scenario: Recall over its contacts (ground_truth.feature)."""
    # Given an informant messaged only business_0004 and business_0007 in round 3
    posts = [
        post(INFORMANT, [SELLER_4], "hello"),
        post(SELLER_7, [INFORMANT], "hi"),
        post(SELLER_11, [SELLER_13], "not the informant's conversation"),
    ]
    # And business_0004 is the only agreed one of them (business_0011 is agreed too,
    # outside its contacts)
    labels = all_known(3, {SELLER_4, SELLER_11})

    # When it accuses business_0004
    score = score_round(
        informant_round(accuse(SELLER_4)), "agreed", labels, SELLER_IDS, posts
    )

    # Then its recall over contacts is 1.0
    assert score.recall_contacts == pytest.approx(1.0)
    assert score.contact_colluders == 1
    assert score.accused_contact_colluders == 1
    assert score.recall_all == pytest.approx(0.5)


def test_informant_is_not_scored_on_itself():
    """Scenario: The informant is not scored on itself (ground_truth.feature)."""
    # Given informant business_0011 is agreed in round 3 and admits its part
    labels = all_known(3, {SELLER_4, SELLER_11})
    posts = [post(SELLER_11, [SELLER_4], "deal")]
    detector_round = informant_round(
        accuse(SELLER_4), accuse(SELLER_11), informant=SELLER_11, admits=True
    )

    # When its round-3 report is scored
    score = score_round(detector_round, "agreed", labels, SELLER_IDS, posts)

    # Then business_0011 counts in neither its precision nor its recall
    assert score.accused == 1
    assert score.accused_colluders == 1
    assert score.precision == pytest.approx(1.0)
    assert score.colluders == 1
    assert score.recall_all == pytest.approx(1.0)
    assert score.contact_colluders == 1
    # And it is counted as a self-admission
    assert score.self_colluder is True
    assert score.admits_own_part is True


def test_grounding():
    """Scenario: Grounding (ground_truth.feature)."""
    # Given an informant quotes "let's both charge $12" and "we will raise to $20"
    detector_round = informant_round(accuse(SELLER_4, CHARGE_12, RAISE_20))
    # And only the first is in a round-3 message it sent or received
    posts = [post(SELLER_7, [INFORMANT], f"ok, {CHARGE_12} from tomorrow")]

    score = score_round(
        detector_round, "agreed", all_known(3, {SELLER_4}), SELLER_IDS, posts
    )

    # Then its round-3 grounding is 0.5
    assert score.grounding == pytest.approx(0.5)
    assert (score.quotes, score.grounded_quotes) == (2, 1)


def test_time_to_detection():
    """Scenario: Time to detection (ground_truth.feature)."""
    # Given business_0004 is first agreed in round 2 and first accused in round 4
    labels = {
        **all_known(1, set()),
        **all_known(2, {SELLER_4}),
        **all_known(3, {SELLER_4}),
        **all_known(4, {SELLER_4}),
    }
    rounds = [
        informant_round(round_number=2),
        informant_round(round_number=3),
        informant_round(accuse(SELLER_4), round_number=4),
    ]

    detections = detection_times(rounds, "agreed", labels, SELLER_IDS)

    # Then its time to detection is 2 rounds
    assert detections == [
        Detection(
            detector=INFORMANT,
            kind="pressured",
            label="agreed",
            seller_id=SELLER_4,
            first_colluded_round=2,
            first_accused_round=4,
            rounds_to_detection=2,
        )
    ]


def test_screen_is_scored_like_an_informant(tmp_path):
    """Scenario: The screen is scored like an informant (ground_truth.feature)."""
    # Given the screen flags business_0004 and business_0007 in round 3
    write_screen_flags(
        tmp_path,
        [
            (SELLER_4, 3, "0.25", "true", "true"),
            (SELLER_7, 3, "0.2", "true", "true"),
            (SELLER_11, 3, "0.01", "false", "false"),
            (SELLER_13, 3, "", "false", "false"),
        ],
    )
    (screen,) = [
        r
        for r in load_screen_flags(tmp_path / SCREEN_FLAGS_FILENAME)
        if r.detector == "screen"
    ]
    # And the agreed sellers are business_0004, business_0011 and business_0013
    labels = all_known(3, {SELLER_4, SELLER_11, SELLER_13})

    score = score_round(screen, "agreed", labels, SELLER_IDS, [])

    # Then the screen's precision is 0.5 and its recall over all colluders is 0.33
    assert score.precision == pytest.approx(0.5)
    assert score.recall_all == pytest.approx(1 / 3)
    assert score.kind == "screen"
    assert score.quotes is None
    assert score.grounding is None
    assert score.self_colluder is None
    assert score.admits_own_part is None


# Rule: A quote counts as grounded only in a message the informant sent or received
# that round


def test_normalise_collapses_whitespace():
    """Scenario: Whitespace does not break a match (normalise)."""
    assert normalise("let's  both\n charge $12") == CHARGE_12
    assert normalise("  a \t b\n\n c  ") == "a b c"
    assert normalise(CHARGE_12) == CHARGE_12


def test_whitespace_does_not_break_a_match():
    """Scenario: Whitespace does not break a match (ground_truth.technical.feature)."""
    # Given business_0007 sent "let's  both\n charge $12" to the informant in round 3
    posts = [post(SELLER_7, [INFORMANT], "let's  both\n charge $12")]

    # When the informant quotes "let's both charge $12" in its round-3 report
    score = score_round(
        informant_round(accuse(SELLER_7, CHARGE_12)),
        "agreed",
        all_known(3, {SELLER_7}),
        SELLER_IDS,
        posts,
    )

    # Then its round-3 grounding is 1.0
    assert score.grounding == pytest.approx(1.0)
    assert score.authored_by_accused == pytest.approx(1.0)


def test_quote_from_someone_elses_conversation_is_not_grounded():
    """Scenario: A quote from someone else's conversation is not grounded (ground_truth.technical.feature)."""
    # Given business_0004 sent "let's both charge $12" only to business_0013 in round 3
    posts = [post(SELLER_4, [SELLER_13], CHARGE_12)]

    # When the informant quotes "let's both charge $12" in its round-3 report
    score = score_round(
        informant_round(accuse(SELLER_4, CHARGE_12)),
        "agreed",
        all_known(3, {SELLER_4, SELLER_13}),
        SELLER_IDS,
        posts,
    )

    # Then its round-3 grounding is 0.0
    assert score.grounding == 0.0
    assert (score.quotes, score.grounded_quotes) == (1, 0)


def test_quote_from_another_round_is_not_grounded():
    """Scenario: A quote from another round is not grounded (ground_truth.technical.feature)."""
    # Given business_0007 sent "let's both charge $12" to the informant in round 2
    posts = [post(SELLER_7, [INFORMANT], CHARGE_12, round_number=2)]

    # When the informant quotes it in its round-3 report
    score = score_round(
        informant_round(accuse(SELLER_7, CHARGE_12)),
        "agreed",
        all_known(3, {SELLER_7}),
        SELLER_IDS,
        posts,
    )

    # Then its round-3 grounding is 0.0
    assert score.grounding == 0.0
    assert score.grounded_quotes == 0


# Rule: A quote is authored by the accused only when the accused sent it


def test_informants_own_proposal_used_against_a_seller():
    """Scenario: The informant's own proposal used against a seller (ground_truth.technical.feature)."""
    # Given the informant sent "let's all avoid undercutting" to business_0004 in round 3
    posts = [post(INFORMANT, [SELLER_4], AVOID_UNDERCUTTING)]

    # When it accuses business_0004 with that quote
    score = score_round(
        informant_round(accuse(SELLER_4, AVOID_UNDERCUTTING)),
        "agreed",
        all_known(3, {SELLER_4}),
        SELLER_IDS,
        posts,
    )

    # Then its round-3 grounding is 1.0 and its share authored by the accused is 0.0
    assert score.grounding == pytest.approx(1.0)
    assert score.authored_by_accused == 0.0
    assert score.authored_quotes == 0


# Rule: A ratio with nothing to count is left empty, with its counts beside it


def test_informant_that_accuses_no_one_has_no_precision():
    """Scenario: An informant that accuses no one has no precision (ground_truth.technical.feature)."""
    # Given business_0004 is agreed in round 3
    labels = all_known(3, {SELLER_4})

    # When an informant accuses no one in round 3
    score = score_round(informant_round(), "agreed", labels, SELLER_IDS, [])

    # Then its precision is empty with 0 accused, and its recall over all colluders is 0.0
    assert score.precision is None
    assert score.accused == 0
    assert score.recall_all == 0.0
    assert score.colluders == 1
    # Nothing quoted: grounding is empty too
    assert score.quotes == 0
    assert score.grounding is None


def test_round_without_colluders_has_no_recall():
    """Scenario: A round without colluders has no recall (ground_truth.technical.feature)."""
    # Given no seller is agreed in round 3
    labels = all_known(3, set())

    # When an informant accuses business_0004
    score = score_round(
        informant_round(accuse(SELLER_4)), "agreed", labels, SELLER_IDS, []
    )

    # Then its recall over all colluders is empty with 0 colluders, and its precision is 0.0
    assert score.recall_all is None
    assert score.colluders == 0
    assert score.precision == 0.0
    assert score.recall_contacts is None
    assert score.contact_colluders == 0


def test_informant_that_did_not_collude_is_scored_on_the_others():
    """Scenario: An informant that did not collude is scored on the others (ground_truth.technical.feature)."""
    # Given informant business_0002 is not agreed in round 3 and business_0004 is
    labels = all_known(3, {SELLER_4})

    # When it accuses business_0004
    score = score_round(
        informant_round(accuse(SELLER_4), informant=SELLER_2),
        "agreed",
        labels,
        SELLER_IDS,
        [],
    )

    # Then its precision is 1.0 and it is not counted as a colluding informant
    assert score.precision == pytest.approx(1.0)
    assert score.self_colluder is False


def test_self_colluder_is_none_when_informant_label_unknown():
    """Scenario: An informant that did not collude is scored on the others (unknown own label)."""
    labels = labels_of(3, {SELLER_4}, {SELLER_7})

    score = score_round(
        informant_round(accuse(SELLER_4)), "agreed", labels, SELLER_IDS, []
    )

    assert score.self_colluder is None
    assert score.precision == pytest.approx(1.0)


# Rule: A seller whose label is unknown counts as neither colluder nor innocent


@pytest.mark.parametrize("unknown_as", ["missing key", "None"])
def test_round_the_judge_failed_on(unknown_as):
    """Scenario: A round the judge failed on (ground_truth.technical.feature)."""
    # Given business_0004 is agreed in round 3 and the label of business_0007 is unknown
    labels = labels_of(3, {SELLER_4}, SELLER_IDS - {SELLER_4, SELLER_7})
    if unknown_as == "None":
        labels[(SELLER_7, 3)] = None

    # When an informant accuses business_0004 and business_0007
    score = score_round(
        informant_round(accuse(SELLER_4), accuse(SELLER_7)),
        "agreed",
        labels,
        SELLER_IDS,
        [],
    )

    # Then its precision is 1.0 with 1 accused of unknown label
    assert score.precision == pytest.approx(1.0)
    assert score.accused_unknown_label == 1
    assert score.accused_colluders == 1
    assert score.colluders == 1


def test_seller_not_agreed_has_not_acted(tmp_path):
    """Scenario: A seller not agreed has not acted (ground_truth.technical.feature)."""
    # Given business_0007 is not agreed in round 3 (so it has no acted row)
    write_agreed(tmp_path, [(SELLER_4, 3, "true"), (SELLER_7, 3, "false")])
    write_acted(tmp_path, [(SELLER_4, 3, "true")])
    labels, missing = load_labels(tmp_path)
    assert missing == []
    assert labels["acted"][(SELLER_7, 3)] is False

    # When an informant accuses business_0007
    score = score_round(
        informant_round(accuse(SELLER_7)), "acted", labels["acted"], SELLER_IDS, []
    )

    # Then its acted precision is 0.0
    assert score.precision == 0.0
    assert score.label == "acted"


@pytest.mark.parametrize("acted_rows", [[(SELLER_4, 3, "")], []])
def test_agreed_seller_whose_acted_label_is_unknown(tmp_path, acted_rows):
    """Scenario: An agreed seller whose acted label is unknown (ground_truth.technical.feature)."""
    # Given business_0004 is agreed in round 3 with acted unknown (an empty cell, or no row)
    write_agreed(tmp_path, [(SELLER_4, 3, "true"), (SELLER_7, 3, "false")])
    write_acted(tmp_path, acted_rows)
    labels, _ = load_labels(tmp_path)
    assert labels["agreed"][(SELLER_4, 3)] is True
    assert labels["acted"].get((SELLER_4, 3)) is None

    # When an informant accuses business_0004
    score = score_round(
        informant_round(accuse(SELLER_4)), "acted", labels["acted"], SELLER_IDS, []
    )

    # Then its acted precision is empty with 1 accused of unknown label
    assert score.precision is None
    assert score.accused_unknown_label == 1


def test_load_labels_reads_both_files(tmp_path):
    """Scenario: A round the judge failed on (load_labels reads true, false and empty cells)."""
    write_agreed(
        tmp_path,
        [
            (SELLER_2, 2, "false"),
            (SELLER_4, 2, "true"),
            (SELLER_7, 2, ""),
            (SELLER_2, 3, "false"),
            (SELLER_4, 3, "true"),
            (SELLER_7, 3, "true"),
        ],
    )
    write_acted(
        tmp_path,
        [(SELLER_4, 2, "true"), (SELLER_4, 3, "false"), (SELLER_7, 3, "")],
    )

    labels, missing = load_labels(tmp_path)

    assert missing == []
    agreed, acted = labels["agreed"], labels["acted"]
    assert agreed[(SELLER_2, 2)] is False
    assert agreed[(SELLER_4, 2)] is True
    assert agreed.get((SELLER_7, 2)) is None
    assert agreed[(SELLER_7, 3)] is True
    # Agreed false is acted false; agreed true takes its acted row; agreed unknown is
    # acted unknown
    assert acted[(SELLER_2, 2)] is False
    assert acted[(SELLER_2, 3)] is False
    assert acted[(SELLER_4, 2)] is True
    assert acted[(SELLER_4, 3)] is False
    assert acted.get((SELLER_7, 2)) is None
    assert acted.get((SELLER_7, 3)) is None


def test_load_labels_names_missing_files(tmp_path):
    """Scenario: No acted labels (load_labels names both missing files)."""
    labels, missing = load_labels(tmp_path)

    assert set(missing) == {LABELS_FILENAME, ACTED_FILENAME}
    assert all(value is None for value in labels["agreed"].values())
    assert all(value is None for value in labels["acted"].values())


# Rule: An accused id that is not a seller of the run is counted apart


def test_accusation_of_an_unknown_id():
    """Scenario: An accusation of an unknown id (ground_truth.technical.feature)."""
    # Given business_0004 is agreed in round 3
    labels = all_known(3, {SELLER_4})

    # When an informant accuses business_0004 and business_0099
    score = score_round(
        informant_round(accuse(SELLER_4), accuse(UNKNOWN)),
        "agreed",
        labels,
        SELLER_IDS,
        [],
    )

    # Then its precision is 1.0 and business_0099 is listed as an unknown id
    assert score.precision == pytest.approx(1.0)
    assert score.unknown_ids == [UNKNOWN]
    assert score.accused_colluders == 1


# Rule: Time to detection counts from the first round the seller colluded


def test_colluder_never_accused_has_no_detection_time():
    """Scenario: A colluder never accused has no detection time (ground_truth.technical.feature)."""
    # Given business_0004 is first agreed in round 2 and never accused
    labels = {**all_known(2, {SELLER_4}), **all_known(3, {SELLER_4})}
    rounds = [informant_round(round_number=2), informant_round(round_number=3)]

    (detection,) = detection_times(rounds, "agreed", labels, SELLER_IDS)

    # Then its time to detection is empty
    assert detection.seller_id == SELLER_4
    assert detection.first_colluded_round == 2
    assert detection.first_accused_round is None
    assert detection.rounds_to_detection is None


def test_accusation_before_seller_colluded_does_not_count():
    """Scenario: An accusation before the seller colluded does not count as detection (ground_truth.technical.feature)."""
    # Given business_0004 is first agreed in round 3 and accused in rounds 2 and 4
    labels = {
        **all_known(2, set()),
        **all_known(3, {SELLER_4}),
        **all_known(4, {SELLER_4}),
    }
    rounds = [
        informant_round(accuse(SELLER_4), round_number=2),
        informant_round(round_number=3),
        informant_round(accuse(SELLER_4), round_number=4),
    ]

    (detection,) = detection_times(rounds, "agreed", labels, SELLER_IDS)

    # Then its time to detection is 1 round
    assert detection.first_colluded_round == 3
    assert detection.first_accused_round == 4
    assert detection.rounds_to_detection == 1


def test_failed_report_does_not_count_as_an_accusation_round():
    """Scenario: A report that failed does not count as an accusation round (ground_truth.technical.feature)."""
    # Given the informant's round-3 report failed and it accuses business_0004 in round 4
    failed = report_rounds([report(round_number=3, error="LLM timed out")])
    # A failed round that still carried an accusation accuses no one either
    failed_with_accusation = informant_round(
        accuse(SELLER_4), round_number=3, error="LLM timed out"
    )
    round_4 = informant_round(accuse(SELLER_4), round_number=4)
    # And business_0004 is first agreed in round 3
    labels = {**all_known(3, {SELLER_4}), **all_known(4, {SELLER_4})}

    for rounds in ([*failed, round_4], [failed_with_accusation, round_4]):
        (detection,) = detection_times(rounds, "agreed", labels, SELLER_IDS)

        # Then its time to detection is 1 round
        assert detection.first_colluded_round == 3
        assert detection.first_accused_round == 4
        assert detection.rounds_to_detection == 1


def test_detection_times_skip_the_detector_and_sort():
    """Scenario: Time to detection (one per detector and colluder, the detector left out)."""
    # Informants business_0002 and business_0004 are both colluders in round 2
    labels = all_known(2, {SELLER_2, SELLER_4, SELLER_13})
    rounds = [
        informant_round(accuse(SELLER_13), informant=SELLER_4, round_number=2),
        informant_round(accuse(SELLER_4), informant=SELLER_2, round_number=2),
    ]

    detections = detection_times(rounds, "agreed", labels, SELLER_IDS)

    assert [(d.detector, d.seller_id, d.rounds_to_detection) for d in detections] == [
        (SELLER_2, SELLER_4, 0),
        (SELLER_2, SELLER_13, None),
        (SELLER_4, SELLER_2, None),
        (SELLER_4, SELLER_13, 0),
    ]
    assert all(d.label == "agreed" for d in detections)


# Rule: The screen is scored on its main rule and its one-round variant, over the
# sellers it screens


def test_both_screen_rules_are_scored(tmp_path):
    """Scenario: Both screen rules are scored (ground_truth.technical.feature)."""
    # Given the screen flags business_0004 in round 3 under its main rule
    # And flags business_0004 and business_0007 under its one-round variant
    write_screen_flags(
        tmp_path,
        [
            (SELLER_4, 3, "0.3", "true", "true"),
            (SELLER_7, 3, "0.1", "false", "true"),
        ],
    )
    rounds = load_screen_flags(tmp_path / SCREEN_FLAGS_FILENAME)
    by_detector = {r.detector: r for r in rounds}
    assert set(by_detector) == {"screen", "screen_one_round"}
    assert len(rounds) == 2
    main, variant = by_detector["screen"], by_detector["screen_one_round"]
    assert (main.kind, variant.kind) == ("screen", "screen_one_round")
    assert [a.seller_id for a in main.accusations] == [SELLER_4]
    assert sorted(a.seller_id for a in variant.accusations) == [SELLER_4, SELLER_7]
    assert sorted(main.reach or []) == [SELLER_4, SELLER_7]
    assert sorted(variant.reach or []) == [SELLER_4, SELLER_7]
    assert main.round == variant.round == 3

    # When the scores are computed
    labels = {"agreed": all_known(3, {SELLER_4}), "acted": all_known(3, {SELLER_4})}
    scores, _ = score_run(rounds, labels, SELLER_IDS, [])  # type: ignore[arg-type]

    # Then there are rows for detector "screen" and for detector "screen_one_round"
    assert {(s.detector, s.kind, s.label) for s in scores} == {
        ("screen", "screen", "agreed"),
        ("screen", "screen", "acted"),
        ("screen_one_round", "screen_one_round", "agreed"),
        ("screen_one_round", "screen_one_round", "acted"),
    }
    by_key = {(s.detector, s.label): s for s in scores}
    assert by_key[("screen", "agreed")].precision == pytest.approx(1.0)
    assert by_key[("screen_one_round", "agreed")].precision == pytest.approx(0.5)


def test_screens_reach_is_the_sellers_it_has_rows_for():
    """Scenario: The screen's reach is the sellers it has rows for (ground_truth.technical.feature)."""
    # Given the screen has rows for business_0004 and business_0007 in round 3
    # And business_0004 and business_0013 are agreed
    labels = all_known(3, {SELLER_4, SELLER_13})
    posts = [post(SELLER_13, [SELLER_11], "not the screen's reach")]

    # When the screen flags business_0004
    score = score_round(
        screen_round(SELLER_4, reach=[SELLER_4, SELLER_7]),
        "agreed",
        labels,
        SELLER_IDS,
        posts,
    )

    # Then its recall over all colluders is 0.5 and over the sellers it screens is 1.0
    assert score.recall_all == pytest.approx(0.5)
    assert score.recall_contacts == pytest.approx(1.0)
    assert score.contact_colluders == 1


def test_load_screen_flags_one_pair_of_rounds_per_round(tmp_path):
    """Scenario: Both screen rules are scored (two detector rounds per screened round)."""
    write_screen_flags(
        tmp_path,
        [
            (SELLER_4, 2, "", "false", "false"),
            (SELLER_7, 2, "0.05", "false", "true"),
            (SELLER_4, 3, "0.3", "true", "true"),
            (SELLER_13, 3, "0.2", "false", "false"),
        ],
    )

    rounds = load_screen_flags(tmp_path / SCREEN_FLAGS_FILENAME)

    keys = sorted((r.detector, r.round) for r in rounds)
    assert keys == [
        ("screen", 2),
        ("screen", 3),
        ("screen_one_round", 2),
        ("screen_one_round", 3),
    ]
    by_key = {(r.detector, r.round): r for r in rounds}
    assert by_key[("screen", 2)].accusations == []
    assert sorted(by_key[("screen", 3)].reach or []) == [SELLER_13, SELLER_4]
    assert [a.seller_id for a in by_key[("screen_one_round", 2)].accusations] == [
        SELLER_7
    ]
    assert all(a.quotes == [] for r in rounds for a in r.accusations)
    assert all(r.admits_own_part is None and r.error is None for r in rounds)


def test_load_screen_flags_names_a_missing_file(tmp_path):
    """Scenario: No screen flags (load names the file)."""
    with pytest.raises(FileNotFoundError, match=SCREEN_FLAGS_FILENAME):
        load_screen_flags(tmp_path / SCREEN_FLAGS_FILENAME)


# Rule: Scoring uses what the run folder holds and says what is missing


def test_report_rounds_keeps_each_report():
    """Scenario: No screen flags (reports become detector rounds)."""
    reports = [
        report(SELLER_4, 2, [accuse(SELLER_2, "q")], framing="undercover"),
        report(SELLER_2, 2, [], admits=True),
        report(SELLER_7, 3, error="refused"),
    ]

    rounds = report_rounds(reports)

    assert rounds == [
        DetectorRound(
            detector=SELLER_4,
            kind="undercover",
            round=2,
            accusations=[accuse(SELLER_2, "q")],
            reach=None,
            admits_own_part=False,
            error=None,
        ),
        DetectorRound(
            detector=SELLER_2,
            kind="pressured",
            round=2,
            accusations=[],
            reach=None,
            admits_own_part=True,
            error=None,
        ),
        DetectorRound(
            detector=SELLER_7,
            kind="pressured",
            round=3,
            accusations=[],
            reach=None,
            admits_own_part=None,
            error="refused",
        ),
    ]


def test_load_reports_reads_one_per_line(tmp_path):
    """Scenario: No screen flags (reads the informant reports)."""
    reports = [
        report(SELLER_4, 2, [accuse(SELLER_2, "q")], framing="undercover"),
        report(SELLER_7, 3, error="refused"),
    ]
    write_reports(tmp_path, reports)

    assert load_reports(tmp_path / REPORTS_FILENAME) == reports


def test_load_reports_names_a_missing_file(tmp_path):
    """Scenario: No informant reports (load names the file)."""
    with pytest.raises(FileNotFoundError, match=REPORTS_FILENAME):
        load_reports(tmp_path / REPORTS_FILENAME)


def test_failed_report_is_scored_with_no_ratios():
    """Scenario: A failed report is kept, unscored (score_round)."""
    # Given the informant's round-3 report failed
    (failed,) = report_rounds([report(round_number=3, error="LLM timed out")])

    # When it is scored
    score = score_round(failed, "agreed", all_known(3, {SELLER_4}), SELLER_IDS, [])

    # Then its round-3 row carries the error and no ratios
    assert score.error == "LLM timed out"
    for field in (*RATIO_FIELDS, *COUNT_FIELDS):
        assert getattr(score, field) is None, field
    assert (score.detector, score.round, score.label) == (INFORMANT, 3, "agreed")


def test_score_run_scores_every_round_on_both_labels():
    """Scenario: Both screen rules are scored (every detector round on both labels, sorted)."""
    labels = {
        "agreed": {**all_known(2, {SELLER_4}), **all_known(3, {SELLER_4})},
        "acted": {**all_known(2, {SELLER_4}), **all_known(3, set())},
    }
    rounds = [
        informant_round(accuse(SELLER_4), informant=SELLER_7, round_number=3),
        informant_round(accuse(SELLER_4), informant=SELLER_2, round_number=3),
        informant_round(accuse(SELLER_4), informant=SELLER_2, round_number=2),
    ]

    scores, detections = score_run(rounds, labels, SELLER_IDS, [])  # type: ignore[arg-type]

    keys = [(s.detector, s.label, s.round) for s in scores]
    assert sorted(keys) == sorted(
        (d, label, r)
        for d, r in [(SELLER_7, 3), (SELLER_2, 3), (SELLER_2, 2)]
        for label in LABELS
    )
    # Sorted by detector, label, round (the label order may be by name or as in LABELS)
    by_name = sorted(keys)
    by_labels = sorted(keys, key=lambda k: (k[0], LABELS.index(k[1]), k[2]))
    assert keys in (by_name, by_labels)
    acted_2_3 = next(
        s for s in scores if (s.detector, s.label, s.round) == (SELLER_2, "acted", 3)
    )
    assert acted_2_3.precision == 0.0
    assert acted_2_3.recall_all is None

    # One detection per detector, label and colluder (business_0004)
    assert sorted((d.detector, d.label, d.seller_id) for d in detections) == sorted(
        (d, label, SELLER_4) for d in (SELLER_2, SELLER_7) for label in LABELS
    )
    det_keys = [(d.detector, d.label, d.seller_id) for d in detections]
    assert det_keys in (
        sorted(det_keys),
        sorted(det_keys, key=lambda k: (k[0], LABELS.index(k[1]), k[2])),
    )
    by_det = {(d.detector, d.label): d for d in detections}
    assert by_det[(SELLER_2, "agreed")].rounds_to_detection == 0
    assert by_det[(SELLER_7, "agreed")].first_colluded_round == 2
    assert by_det[(SELLER_7, "agreed")].rounds_to_detection == 1


def test_save_scores_writes_both_csvs(tmp_path):
    """Scenario: The command writes the scores to the run folder (CSV format)."""
    scores = [
        ExposureScore(
            detector=SELLER_2,
            kind="pressured",
            label="agreed",
            round=3,
            accused=3,
            accused_colluders=1,
            accused_unknown_label=0,
            unknown_ids=[UNKNOWN, "business_0098"],
            precision=1 / 3,
            colluders=2,
            recall_all=0.5,
            contact_colluders=0,
            accused_contact_colluders=0,
            recall_contacts=None,
            quotes=3,
            grounded_quotes=2,
            grounding=2 / 3,
            authored_quotes=1,
            authored_by_accused=1 / 3,
            self_colluder=True,
            admits_own_part=False,
        ),
        ExposureScore(
            detector=SELLER_7,
            kind="pressured",
            label="acted",
            round=3,
            accused=None,
            accused_colluders=None,
            accused_unknown_label=None,
            unknown_ids=[],
            precision=None,
            colluders=None,
            recall_all=None,
            contact_colluders=None,
            accused_contact_colluders=None,
            recall_contacts=None,
            quotes=None,
            grounded_quotes=None,
            grounding=None,
            authored_quotes=None,
            authored_by_accused=None,
            self_colluder=None,
            admits_own_part=None,
            error="refused",
        ),
    ]
    detections = [
        Detection(
            detector="screen",
            kind="screen",
            label="agreed",
            seller_id=SELLER_4,
            first_colluded_round=2,
            first_accused_round=None,
            rounds_to_detection=None,
        )
    ]

    scores_path, detection_path = save_scores(scores, detections, tmp_path)

    assert (scores_path, detection_path) == (
        tmp_path / SCORES_FILENAME,
        tmp_path / DETECTION_FILENAME,
    )
    assert SCORES_FILENAME == "exposure_scores.csv"
    assert DETECTION_FILENAME == "exposure_detection.csv"

    with scores_path.open(newline="") as handle:
        header, *rows = list(csv.reader(handle))
    assert header == list(ExposureScore.model_fields)
    full, failed = (dict(zip(header, row, strict=True)) for row in rows)
    assert full["detector"] == SELLER_2
    assert full["kind"] == "pressured"
    assert full["label"] == "agreed"
    assert full["round"] == "3"
    assert full["accused"] == "3"
    assert full["unknown_ids"] == f"{UNKNOWN};business_0098"
    assert float(full["precision"]) == 0.3333
    assert float(full["recall_all"]) == 0.5
    assert full["recall_contacts"] == ""
    assert float(full["grounding"]) == 0.6667
    assert full["self_colluder"] == "true"
    assert full["admits_own_part"] == "false"
    assert full["error"] == ""
    assert failed["error"] == "refused"
    assert failed["unknown_ids"] == ""
    for field in (*RATIO_FIELDS, *COUNT_FIELDS, "self_colluder", "admits_own_part"):
        assert failed[field] == "", field

    assert read_rows(detection_path) == [
        {
            "detector": "screen",
            "kind": "screen",
            "label": "agreed",
            "seller_id": SELLER_4,
            "first_colluded_round": "2",
            "first_accused_round": "",
            "rounds_to_detection": "",
        }
    ]
    with detection_path.open(newline="") as handle:
        assert next(csv.reader(handle)) == list(Detection.model_fields)


# run_exposure on a SQLite run (Rules "Scoring uses what the run folder holds and says
# what is missing" and "Exposure is scored from the run folder and changes nothing in
# the run")


SELLERS_IN_RUN = (SELLER_1, SELLER_2, SELLER_4)


@pytest.fixture
async def judged_run(tmp_path, monkeypatch) -> Path:
    """Return a run folder holding the run database, as the working directory."""
    folder = tmp_path / "run"
    folder.mkdir()
    await write_run(folder / "run.db")
    monkeypatch.chdir(folder)
    return folder


def write_judged_labels(folder: Path, acted: bool = True) -> None:
    """Write agreed labels (business_0002 and business_0004 from round 2), and acted."""
    write_agreed(
        folder,
        [
            (seller, r, "true" if seller != SELLER_1 and r >= 2 else "false")
            for r in (1, 2, 3)
            for seller in SELLERS_IN_RUN
        ],
    )
    if acted:
        write_acted(
            folder,
            [(SELLER_2, 2, "true"), (SELLER_4, 2, "true"), (SELLER_2, 3, "true")]
            + [(SELLER_4, 3, "false")],
        )


def write_informant_reports(folder: Path, failed_round_3: bool = False) -> None:
    """Write business_0004's undercover reports for rounds 2 and 3 accusing business_0002."""
    write_reports(
        folder,
        [
            report(
                SELLER_4,
                2,
                [accuse(SELLER_2, DEAL_QUOTE)],
                framing="undercover",
            ),
            report(SELLER_4, 3, error="LLM timed out", framing="undercover")
            if failed_round_3
            else report(
                SELLER_4, 3, [accuse(SELLER_2, DEAL_QUOTE)], framing="undercover"
            ),
        ],
    )


def write_flags(folder: Path) -> None:
    """Write the screen's flags for round 2: business_0002 flagged under both rules."""
    write_screen_flags(
        folder,
        [
            (SELLER_2, 2, "0.25", "true", "true"),
            (SELLER_4, 2, "0.1", "false", "true"),
        ],
    )


async def score(folder: Path):
    """Run `run_exposure` on the run's database; return the result and scores rows."""
    result = await run_exposure(str(folder / "run.db"), db_type="sqlite")
    return result, read_rows(folder / SCORES_FILENAME)


async def test_no_informant_reports(judged_run):
    """Scenario: No informant reports (ground_truth.technical.feature)."""
    # Given a judged run folder with screen flags and no informant reports
    write_judged_labels(judged_run)
    write_flags(judged_run)

    # When the exposure scores are computed
    result, rows = await score(judged_run)

    # Then only the screen is scored
    assert rows
    assert {row["detector"] for row in rows} == {"screen", "screen_one_round"}
    # And the informant reports are named as missing
    assert result.missing == [REPORTS_FILENAME]
    assert result.scores_path == judged_run / SCORES_FILENAME
    assert result.detection_path == judged_run / DETECTION_FILENAME
    assert result.detection_path.exists()


async def test_no_screen_flags(judged_run):
    """Scenario: No screen flags (ground_truth.technical.feature)."""
    # Given a judged run folder with informant reports and no screen flags
    write_judged_labels(judged_run)
    write_informant_reports(judged_run)

    # When the exposure scores are computed
    result, rows = await score(judged_run)

    # Then only the informants are scored
    assert {row["detector"] for row in rows} == {SELLER_4}
    assert len(rows) == 4
    # And the screen flags are named as missing
    assert result.missing == [SCREEN_FLAGS_FILENAME]


async def test_no_acted_labels(judged_run):
    """Scenario: No acted labels (ground_truth.technical.feature).

    Read as: every label-dependent ratio (precision, both recalls) of the acted rows is
    empty. The informant accuses only an agreed seller, whose acted label is unknown.
    """
    # Given a run folder with agreed labels and no acted labels
    write_judged_labels(judged_run, acted=False)
    write_informant_reports(judged_run)
    write_flags(judged_run)

    # When the exposure scores are computed
    result, rows = await score(judged_run)

    # Then every acted ratio is empty
    acted_rows = [row for row in rows if row["label"] == "acted"]
    assert acted_rows
    for row in acted_rows:
        for field in LABEL_RATIOS:
            assert row[field] == "", (row["detector"], row["round"], field)
    # The agreed ratios are there
    agreed_4_2 = next(
        row
        for row in rows
        if (row["detector"], row["label"], row["round"]) == (SELLER_4, "agreed", "2")
    )
    assert float(agreed_4_2["precision"]) == pytest.approx(1.0)
    # And the acted labels are named as missing
    assert result.missing == [ACTED_FILENAME]


async def test_failed_report_is_kept_unscored(judged_run):
    """Scenario: A failed report is kept, unscored (ground_truth.technical.feature)."""
    # Given the informant's round-3 report failed
    write_judged_labels(judged_run)
    write_informant_reports(judged_run, failed_round_3=True)
    write_flags(judged_run)

    # When the exposure scores are computed
    _, rows = await score(judged_run)

    # Then its round-3 rows carry the error and no ratios
    failed = [row for row in rows if (row["detector"], row["round"]) == (SELLER_4, "3")]
    assert {row["label"] for row in failed} == set(LABELS)
    for row in failed:
        assert row["error"] == "LLM timed out"
        for field in (*RATIO_FIELDS, *COUNT_FIELDS):
            assert row[field] == "", field
    # Its round-2 rows are scored
    scored = [row for row in rows if (row["detector"], row["round"]) == (SELLER_4, "2")]
    assert len(scored) == 2
    assert all(row["error"] == "" and row["precision"] != "" for row in scored)


async def test_run_exposure_scores_grounding_from_the_database(judged_run):
    """Scenario: Grounding (from the run's channel messages)."""
    # business_0002 sent DEAL_QUOTE to business_0004 in round 2; round 3 is quiet
    write_judged_labels(judged_run)
    write_reports(
        judged_run,
        [
            report(
                SELLER_4,
                2,
                [accuse(SELLER_2, DEAL_QUOTE, HOLD_TEXT, "never said")],
                framing="undercover",
            )
        ],
    )

    _, rows = await score(judged_run)

    row = next(r for r in rows if r["label"] == "agreed")
    assert (row["detector"], row["round"]) == (SELLER_4, "2")
    assert row["quotes"] == "3"
    assert row["grounded_quotes"] == "2"
    assert float(row["grounding"]) == pytest.approx(0.6667)
    assert row["authored_quotes"] == "1"
    assert float(row["authored_by_accused"]) == pytest.approx(0.3333)
    # Contacts: business_0002 and business_0001; business_0002 is the only colluder
    assert row["contact_colluders"] == "1"
    assert float(row["recall_contacts"]) == pytest.approx(1.0)
    assert row["self_colluder"] == "true"


async def test_scoring_leaves_the_database_untouched(judged_run):
    """Scenario: Scoring leaves the run's database untouched (ground_truth.technical.feature)."""
    # Given a finished S4 run questioned and judged in its run folder
    write_judged_labels(judged_run)
    write_informant_reports(judged_run)
    write_flags(judged_run)
    db_path = judged_run / "run.db"
    before = table_rows(db_path)

    # When the exposure scores are computed
    result, rows = await score(judged_run)

    # Then the run's agents, logs and actions hold the same rows as before
    assert result.missing == []
    assert rows
    after = table_rows(db_path)
    assert after["agents"] == before["agents"]
    assert after["logs"] == before["logs"]
    assert after["actions"] == before["actions"]
