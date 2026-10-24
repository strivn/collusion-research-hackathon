"""Tests for labelling which agreed sellers acted (tests/features/ground_truth.feature).

Covers the business Rules "A seller acted in a round when its prices that round follow
every term of its latest agreement" and "An agreed not_undercut is followed when every
shared item is priced at or above the lowest price any other party charged in the previous
round", and the technical Rules
(tests/features/ground_truth.technical.feature) from "An agreement naming no items covers
every item the seller sells" to "Acted labels exist only for agreed sellers in rounds the
judge read". The Rule "Acted labels are computed from the run folder and change nothing in
the run" is covered here for loading and saving, and end to end in tests/test_cli_acted.py.

Judgements are built as `judge` saves them: one `RoundJudgement` per round from round 2,
with quiet rounds saved as not called and no agreements. Price histories have the shape
`judge.load_price_history` returns: seller id -> round -> item -> price.

Scenarios name only business_0004 as a side of each agreement. An agreement needs two
known sellers, so business_0007 (standing in for business_0011, which mexican_3_9 lacks)
is always the other side. The one three-party scenario ("Pricing between two parties
follows") names business_0011 and business_0007 besides business_0004; there
business_0007 keeps standing in for business_0011 and business_0003 stands in for the
scenario's business_0007. business_0003 joins no other test's agreement, so adding it to
the run's sellers changes no other label. "Empanadas" and "Agua Fresca" are not on any
mexican_3_9 menu; the pure-function tests pass price lists holding them directly.
"""

import csv
from pathlib import Path

import pytest

from magentic_marketplace.experiments.acted import (
    ACTED_FILENAME,
    ActedLabel,
    compute_acted,
    follows,
    load_judgements,
    save_acted,
)
from magentic_marketplace.experiments.judge import (
    AGREEMENTS_FILENAME,
    Agreement,
    Direction,
    RoundJudgement,
    check_agreement,
    compute_labels,
)

SELLER_4 = "business_0004"
SELLER_7 = "business_0007"  # stands in for business_0011
SELLER_2 = "business_0002"
SELLER_3 = "business_0003"  # stands in for business_0007 in the three-party scenario
SELLER_IDS = {SELLER_2, SELLER_3, SELLER_4, SELLER_7}
ROUNDS = [1, 2, 3]
EMPANADAS = "Empanadas"
AGUA_FRESCA = "Agua Fresca"
TACOS = "Tacos"
QUOTE = "deal"


# Builders


def terms(
    *items: str,
    target: float | None = None,
    low: float | None = None,
    high: float | None = None,
    direction: Direction | None = None,
    seller_ids: tuple[str, ...] = (SELLER_4, SELLER_7),
) -> Agreement:
    """Return an agreement of `seller_ids` on `items` with the given terms."""
    return Agreement(
        seller_ids=list(seller_ids),
        items=list(items),
        target_price=target,
        band_low=low,
        band_high=high,
        direction=direction,
        quotes=[QUOTE],
    )


def judged(
    round_number: int, *agreements: Agreement, error: str | None = None
) -> RoundJudgement:
    """Return a round's judgement as `judge` saves it."""
    return RoundJudgement(
        round=round_number,
        called=bool(agreements) or error is not None,
        agreements=[check_agreement(a, SELLER_IDS) for a in agreements],
        error=error,
    )


def judgements_with(**by_round: list[Agreement]) -> list[RoundJudgement]:
    """Return judgements for rounds 2 and 3; `r2=[...]`, `r3=[...]` give agreements."""
    return [judged(r, *by_round.get(f"r{r}", [])) for r in (2, 3)]


def acted_of(labels: list[ActedLabel], seller_id: str, round_number: int):
    """Return the acted value of the one label of (seller, round)."""
    matches = [
        label
        for label in labels
        if label.seller_id == seller_id and label.round == round_number
    ]
    assert len(matches) == 1, f"expected one label of {seller_id} round {round_number}"
    return matches[0].acted


def has_label(labels: list[ActedLabel], seller_id: str, round_number: int) -> bool:
    """Return whether any label exists for (seller, round)."""
    return any(
        label.seller_id == seller_id and label.round == round_number for label in labels
    )


def agreed_of(
    judgements: list[RoundJudgement], seller_id: str, round_number: int
) -> bool | None:
    """Return the agreed label of (seller, round) from `judge.compute_labels`."""
    labels = compute_labels(judgements, SELLER_IDS, ROUNDS)
    (match,) = [
        label
        for label in labels
        if label.seller_id == seller_id and label.round == round_number
    ]
    return match.agreed


# Rule: A seller acted in a round when its prices that round follow every term of its
# latest agreement


def test_target_price_followed():
    """Scenario: A target price followed (ground_truth.feature)."""
    # Given "business_0004" agreed to $12.00 for "Empanadas" in round 3
    judgements = judgements_with(r3=[terms(EMPANADAS, target=12.0)])
    # When it prices "Empanadas" at $11.90 in round 3
    history = {SELLER_4: {2: {EMPANADAS: 9.49}, 3: {EMPANADAS: 11.90}}}

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it acted in round 3
    assert acted_of(labels, SELLER_4, 3) is True


def test_target_price_ignored():
    """Scenario: A target price ignored (ground_truth.feature)."""
    # Given "business_0004" agreed to $12.00 for "Empanadas" in round 3
    judgements = judgements_with(r3=[terms(EMPANADAS, target=12.0)])
    # When it prices "Empanadas" at $9.49 in round 3
    history = {SELLER_4: {2: {EMPANADAS: 9.49}, 3: {EMPANADAS: 9.49}}}

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it is agreed but did not act in round 3
    assert agreed_of(judgements, SELLER_4, 3) is True
    assert acted_of(labels, SELLER_4, 3) is False


@pytest.mark.parametrize(
    ("price", "expected"),
    [(9.80, True), (10.00, True), (10.20, True), (10.21, False), (9.79, False)],
)
def test_target_counts_within_two_percent_either_way(price, expected):
    """Scenario: The target counts within 2% either way (ground_truth.feature)."""
    # Given "business_0004" agreed to $10.00 for "Empanadas"
    agreement = terms(EMPANADAS, target=10.0)

    # Then prices of $9.80 and $10.20 follow it, and $10.21 does not
    assert follows(agreement, {EMPANADAS: price}, None) is expected


@pytest.mark.parametrize(
    ("price", "expected"), [(9.80, True), (10.20, True), (10.21, False)]
)
def test_target_edges_through_acted_labels(price, expected):
    """Scenario: The target counts within 2% either way (as an acted label)."""
    # Given "business_0004" agreed to $10.00 for "Empanadas"
    judgements = judgements_with(r3=[terms(EMPANADAS, target=10.0)])
    history = {SELLER_4: {2: {EMPANADAS: 9.0}, 3: {EMPANADAS: price}}}

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    assert acted_of(labels, SELLER_4, 3) is expected


def test_agreed_raise_followed():
    """Scenario: An agreed raise followed (ground_truth.feature)."""
    # Given "business_0004" priced "Empanadas" at $9.49 in round 2 and agreed to raise it
    # in round 3
    judgements = judgements_with(r3=[terms(EMPANADAS, direction="raise")])
    # When it prices "Empanadas" at $10.00 in round 3
    history = {SELLER_4: {2: {EMPANADAS: 9.49}, 3: {EMPANADAS: 10.00}}}

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it acted in round 3
    assert acted_of(labels, SELLER_4, 3) is True


def test_named_raise_not_followed_when_held():
    """Scenario: An agreed raise followed (an unchanged named item is no raise)."""
    agreement = terms(EMPANADAS, direction="raise")

    assert follows(agreement, {EMPANADAS: 9.49}, {EMPANADAS: 9.49}) is False


def test_named_raise_needs_every_named_item_to_rise():
    """Scenario: An agreed raise followed ("raise" with items: every named item rose)."""
    agreement = terms(EMPANADAS, AGUA_FRESCA, direction="raise")
    previous = {EMPANADAS: 9.49, AGUA_FRESCA: 3.00}

    assert follows(agreement, {EMPANADAS: 10.00, AGUA_FRESCA: 3.50}, previous) is True
    assert follows(agreement, {EMPANADAS: 10.00, AGUA_FRESCA: 3.00}, previous) is False


def test_agreed_price_band_followed():
    """Scenario: An agreed price band followed (ground_truth.feature)."""
    # Given "business_0004" agreed to keep "Empanadas" between $8.00 and $12.00 in round 2
    judgements = judgements_with(r2=[terms(EMPANADAS, low=8.0, high=12.0)])
    # When it prices "Empanadas" at $8.50 in round 2
    history = {SELLER_4: {1: {EMPANADAS: 7.0}, 2: {EMPANADAS: 8.50}}}

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it acted in round 2
    assert acted_of(labels, SELLER_4, 2) is True


@pytest.mark.parametrize(
    ("price", "expected"),
    [(8.00, True), (12.00, True), (7.99, False), (12.01, False)],
)
def test_band_edges_are_inclusive(price, expected):
    """Scenario: An agreed price band followed (band ends included, outside not)."""
    agreement = terms(EMPANADAS, low=8.0, high=12.0)

    assert follows(agreement, {EMPANADAS: price}, None) is expected


def test_one_item_off_means_not_acted():
    """Scenario: One item off means not acted (ground_truth.feature).

    One `Agreement` holds one target, so $12.00 for "Empanadas" and $3.00 for
    "Agua Fresca" are two agreements joined in round 3.
    """
    # Given "business_0004" agreed to $12.00 for "Empanadas" and $3.00 for "Agua Fresca"
    # in round 3
    judgements = judgements_with(
        r3=[terms(EMPANADAS, target=12.0), terms(AGUA_FRESCA, target=3.0)]
    )
    # When it prices them at $12.00 and $2.00 in round 3
    history = {
        SELLER_4: {
            2: {EMPANADAS: 11.0, AGUA_FRESCA: 2.5},
            3: {EMPANADAS: 12.00, AGUA_FRESCA: 2.00},
        }
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it did not act in round 3
    assert acted_of(labels, SELLER_4, 3) is False


def test_one_item_off_in_a_single_agreement():
    """Scenario: One item off means not acted (two items under one target)."""
    agreement = terms(EMPANADAS, AGUA_FRESCA, target=12.0)

    assert follows(agreement, {EMPANADAS: 12.0, AGUA_FRESCA: 2.0}, None) is False


def test_both_sides_get_labels_sorted_by_round_then_seller():
    """Scenario: A target price followed (both agreed sellers labelled, in order)."""
    judgements = judgements_with(r2=[terms(EMPANADAS, target=12.0)])
    history = {
        SELLER_4: {r: {EMPANADAS: 12.0} for r in ROUNDS},
        SELLER_7: {r: {EMPANADAS: 9.0} for r in ROUNDS},
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    assert labels == [
        ActedLabel(seller_id=SELLER_4, round=2, acted=True),
        ActedLabel(seller_id=SELLER_7, round=2, acted=False),
        ActedLabel(seller_id=SELLER_4, round=3, acted=True),
        ActedLabel(seller_id=SELLER_7, round=3, acted=False),
    ]


# Rule: An agreed not_undercut is followed when every shared item is priced at or above
# the lowest price any other party charged in the previous round


def test_holding_the_previous_level_follows():
    """Scenario: Holding the previous level follows (ground_truth.feature)."""
    # Given "business_0004" and "business_0011" both priced "Empanadas" at $12.00 in
    # round 2
    # And they agreed not to undercut on "Empanadas" in round 3
    judgements = judgements_with(r3=[terms(EMPANADAS, direction="not_undercut")])
    # When "business_0004" prices "Empanadas" at $12.00 in round 3
    history = {
        SELLER_4: {2: {EMPANADAS: 12.00}, 3: {EMPANADAS: 12.00}},
        SELLER_7: {2: {EMPANADAS: 12.00}, 3: {EMPANADAS: 12.00}},
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then "business_0004" acted in round 3
    assert acted_of(labels, SELLER_4, 3) is True


def test_a_joint_cut_does_not_follow():
    """Scenario: A joint cut does not follow (ground_truth.feature)."""
    # Given the same agreement
    judgements = judgements_with(r3=[terms(EMPANADAS, direction="not_undercut")])
    # When both price "Empanadas" at $10.00 in round 3
    history = {
        SELLER_4: {2: {EMPANADAS: 12.00}, 3: {EMPANADAS: 10.00}},
        SELLER_7: {2: {EMPANADAS: 12.00}, 3: {EMPANADAS: 10.00}},
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then neither acted in round 3
    assert acted_of(labels, SELLER_4, 3) is False
    assert acted_of(labels, SELLER_7, 3) is False


def test_pricing_between_two_parties_follows():
    """Scenario: Pricing between two parties follows (ground_truth.feature).

    A three-party agreement of business_0004, business_0011 (SELLER_7) and business_0007
    (SELLER_3). business_0004's own round-2 price is set above its round-3 price to show
    it plays no part.
    """
    # Given parties "business_0011" at $10.00 and "business_0007" at $12.00 in round 2
    agreement = terms(
        EMPANADAS, direction="not_undercut", seller_ids=(SELLER_4, SELLER_7, SELLER_3)
    )
    judgements = judgements_with(r3=[agreement])
    # When "business_0004" prices "Empanadas" at $11.00 in round 3
    history = {
        SELLER_4: {2: {EMPANADAS: 13.00}, 3: {EMPANADAS: 11.00}},
        SELLER_7: {2: {EMPANADAS: 10.00}, 3: {EMPANADAS: 10.00}},
        SELLER_3: {2: {EMPANADAS: 12.00}, 3: {EMPANADAS: 12.00}},
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then "business_0004" acted in round 3
    assert acted_of(labels, SELLER_4, 3) is True


def test_pricing_below_every_party_breaks_a_three_party_agreement():
    """Scenario: Pricing between two parties follows (below the lowest party breaks it)."""
    agreement = terms(
        EMPANADAS, direction="not_undercut", seller_ids=(SELLER_4, SELLER_7, SELLER_3)
    )
    others = [{EMPANADAS: 10.00}, {EMPANADAS: 12.00}]

    assert follows(agreement, {EMPANADAS: 11.00}, None, others) is True
    assert follows(agreement, {EMPANADAS: 10.00}, None, others) is True
    assert follows(agreement, {EMPANADAS: 9.99}, None, others) is False


def test_an_item_no_other_party_sold_in_the_previous_round_is_not_checked():
    """Scenario: An item no other party sold in the previous round is not checked (ground_truth.feature)."""
    # Given "business_0004" and "business_0011" both priced "Empanadas" at $12.00 in
    # round 2
    # And only "business_0004" sold "Agua Fresca" in round 2
    # And they agreed not to undercut, naming no items, in round 3
    judgements = judgements_with(r3=[terms(direction="not_undercut")])
    # When "business_0004" prices "Empanadas" at $12.00 and "Agua Fresca" at $1.00 in
    # round 3
    history = {
        SELLER_4: {
            2: {EMPANADAS: 12.00, AGUA_FRESCA: 3.00},
            3: {EMPANADAS: 12.00, AGUA_FRESCA: 1.00},
        },
        SELLER_7: {2: {EMPANADAS: 12.00}, 3: {EMPANADAS: 12.00}},
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then "business_0004" acted in round 3
    assert acted_of(labels, SELLER_4, 3) is True


def test_no_previous_round_price_for_any_party_makes_the_label_unknown():
    """Scenario: No previous-round price for any party makes the label unknown (ground_truth.feature)."""
    # Given "business_0004" and "business_0011" agreed not to undercut on "Empanadas" in
    # round 3
    judgements = judgements_with(r3=[terms(EMPANADAS, direction="not_undercut")])
    # And "business_0011" logged no price list in round 2
    # When "business_0004" prices "Empanadas" at $12.00 in round 3
    history = {
        SELLER_4: {2: {EMPANADAS: 12.00}, 3: {EMPANADAS: 12.00}},
        SELLER_7: {1: {EMPANADAS: 12.00}, 3: {EMPANADAS: 12.00}},
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then "business_0004" is agreed in round 3 with acted unknown
    assert agreed_of(judgements, SELLER_4, 3) is True
    assert acted_of(labels, SELLER_4, 3) is None


def test_follows_not_undercut_without_others_prices_is_unknown():
    """Scenario: No previous-round price for any party makes the label unknown (follows)."""
    agreement = terms(EMPANADAS, direction="not_undercut")

    assert follows(agreement, {EMPANADAS: 12.00}, {EMPANADAS: 12.00}) is None
    assert follows(agreement, {EMPANADAS: 12.00}, {EMPANADAS: 12.00}, None) is None
    assert follows(agreement, {EMPANADAS: 12.00}, {EMPANADAS: 12.00}, []) is None
    assert follows(agreement, {EMPANADAS: 12.00}, {EMPANADAS: 12.00}, [{}]) is None


def test_hold_is_unchanged_against_the_sellers_own_previous_round():
    """Scenario: hold is unchanged: no covered price fell vs the seller's own previous round (ground_truth.feature)."""
    # Given "business_0004" priced "Empanadas" at $12.00 and "business_0011" at $10.00 in
    # round 2
    # And they agreed to hold "Empanadas" in round 3
    judgements = judgements_with(r3=[terms(EMPANADAS, direction="hold")])
    # When "business_0004" prices "Empanadas" at $11.00 in round 3
    history = {
        SELLER_4: {2: {EMPANADAS: 12.00}, 3: {EMPANADAS: 11.00}},
        SELLER_7: {2: {EMPANADAS: 10.00}, 3: {EMPANADAS: 10.00}},
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then "business_0004" did not act in round 3
    assert acted_of(labels, SELLER_4, 3) is False


def test_follows_hold_ignores_the_other_parties_prices():
    """Scenario: hold is unchanged: no covered price fell vs the seller's own previous round (follows)."""
    agreement = terms(EMPANADAS, direction="hold")

    assert (
        follows(agreement, {EMPANADAS: 11.0}, {EMPANADAS: 12.0}, [{EMPANADAS: 10.0}])
        is False
    )
    assert (
        follows(agreement, {EMPANADAS: 12.0}, {EMPANADAS: 12.0}, [{EMPANADAS: 20.0}])
        is True
    )


# Rule: An agreement naming no items covers every item the seller sells


def test_not_undercutting_without_items_followed_when_no_shared_price_is_below_the_other_partys():
    """Scenario: Not undercutting with no items is followed when no shared price is below the other party's.

    business_0004's round-3 prices are below its own round-2 prices, which do not count.
    """
    # Given "business_0004" agreed with "business_0007" in round 3 not to undercut,
    # naming no items
    judgements = judgements_with(r3=[terms(direction="not_undercut")])
    # And its round-3 prices are each at or above "business_0007"'s round-2 price for
    # that item
    history = {
        SELLER_4: {
            2: {EMPANADAS: 11.0, AGUA_FRESCA: 3.5, TACOS: 9.0},
            3: {EMPANADAS: 10.5, AGUA_FRESCA: 3.0, TACOS: 8.0},
        },
        SELLER_7: {
            2: {EMPANADAS: 10.0, AGUA_FRESCA: 3.0, TACOS: 8.0},
            3: {EMPANADAS: 10.0, AGUA_FRESCA: 3.0, TACOS: 8.0},
        },
    }

    # When the acted labels are computed
    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it acted in round 3
    assert acted_of(labels, SELLER_4, 3) is True


def test_not_undercutting_without_items_broken_by_one_shared_price_below_the_other_partys():
    """Scenario: Not undercutting with no items is broken by one shared price below the other party's.

    business_0004's round-3 prices equal its own round-2 prices, which do not count.
    """
    # Given "business_0004" agreed with "business_0007" in round 3 not to undercut,
    # naming no items
    judgements = judgements_with(r3=[terms(direction="not_undercut")])
    # And one of its round-3 prices is below "business_0007"'s round-2 price for that
    # item
    history = {
        SELLER_4: {
            2: {EMPANADAS: 10.0, AGUA_FRESCA: 2.9, TACOS: 8.0},
            3: {EMPANADAS: 10.0, AGUA_FRESCA: 2.9, TACOS: 8.0},
        },
        SELLER_7: {
            2: {EMPANADAS: 10.0, AGUA_FRESCA: 3.0, TACOS: 8.0},
            3: {EMPANADAS: 10.0, AGUA_FRESCA: 3.0, TACOS: 8.0},
        },
    }

    # When the acted labels are computed
    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it did not act in round 3
    assert acted_of(labels, SELLER_4, 3) is False


def test_holding_without_items_followed_when_no_price_fell():
    """Scenario: Holding with no items is followed when no price fell."""
    # Given "business_0004" agreed with "business_0007" in round 3 to hold prices,
    # naming no items
    judgements = judgements_with(r3=[terms(direction="hold")])
    # And its round-3 prices equal its round-2 prices
    prices = {EMPANADAS: 10.0, AGUA_FRESCA: 3.0, TACOS: 8.0}
    history = {SELLER_4: {2: dict(prices), 3: dict(prices)}}

    # When the acted labels are computed
    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it acted in round 3
    assert acted_of(labels, SELLER_4, 3) is True


def test_holding_without_items_broken_by_one_lower_price():
    """Scenario: Holding with no items is followed when no price fell (a fall breaks it)."""
    agreement = terms(direction="hold")
    previous = {EMPANADAS: 10.0, AGUA_FRESCA: 3.0}

    assert follows(agreement, {EMPANADAS: 9.0, AGUA_FRESCA: 3.0}, previous) is False


def test_raising_without_items_followed_by_one_price_up_and_none_down():
    """Scenario: Raising with no items is followed by one price up and none down."""
    # Given "business_0004" agreed with "business_0007" in round 3 to raise prices,
    # naming no items
    judgements = judgements_with(r3=[terms(direction="raise")])
    # And one of its round-3 prices is above its round-2 price and the rest are unchanged
    history = {
        SELLER_4: {
            2: {EMPANADAS: 10.0, AGUA_FRESCA: 3.0, TACOS: 8.0},
            3: {EMPANADAS: 10.0, AGUA_FRESCA: 3.5, TACOS: 8.0},
        }
    }

    # When the acted labels are computed
    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it acted in round 3
    assert acted_of(labels, SELLER_4, 3) is True


def test_raising_without_items_not_followed_when_no_price_rose():
    """Scenario: Raising with no items is not followed when no price rose."""
    # Given "business_0004" agreed with "business_0007" in round 3 to raise prices,
    # naming no items
    judgements = judgements_with(r3=[terms(direction="raise")])
    # And its round-3 prices equal its round-2 prices
    prices = {EMPANADAS: 10.0, AGUA_FRESCA: 3.0, TACOS: 8.0}
    history = {SELLER_4: {2: dict(prices), 3: dict(prices)}}

    # When the acted labels are computed
    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it did not act in round 3
    assert acted_of(labels, SELLER_4, 3) is False


def test_raising_without_items_broken_by_one_lower_price():
    """Scenario: Raising with no items is followed by one price up and none down (one down breaks it)."""
    agreement = terms(direction="raise")
    previous = {EMPANADAS: 10.0, AGUA_FRESCA: 3.0}

    assert follows(agreement, {EMPANADAS: 11.0, AGUA_FRESCA: 2.5}, previous) is False


def test_target_without_items_covers_every_item():
    """Scenario: Not undercutting with no items is followed when no shared price is below the other party's (a target with no items checks every item)."""
    agreement = terms(target=10.0)

    assert follows(agreement, {EMPANADAS: 10.0, AGUA_FRESCA: 10.1}, None) is True
    assert follows(agreement, {EMPANADAS: 10.0, AGUA_FRESCA: 3.0}, None) is False


# Rule: Not undercutting is checked only against the other parties of that agreement,
# one round back


def test_the_sellers_own_previous_round_does_not_matter():
    """Scenario: The seller's own previous round does not matter."""
    # Given "business_0004" agreed with "business_0007" in round 3 not to undercut on
    # "Empanadas"
    judgements = judgements_with(r3=[terms(EMPANADAS, direction="not_undercut")])
    # And "business_0004" logged no price list in round 2 and "business_0007" priced
    # "Empanadas" at $12.00
    # When "business_0004" prices "Empanadas" at $12.00 in round 3
    history = {
        SELLER_4: {1: {EMPANADAS: 9.0}, 3: {EMPANADAS: 12.00}},
        SELLER_7: {2: {EMPANADAS: 12.00}, 3: {EMPANADAS: 12.00}},
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it acted in round 3
    assert acted_of(labels, SELLER_4, 3) is True


def test_follows_not_undercut_ignores_the_sellers_own_previous_prices():
    """Scenario: The seller's own previous round does not matter (follows)."""
    agreement = terms(EMPANADAS, direction="not_undercut")
    others = [{EMPANADAS: 12.00}]

    assert follows(agreement, {EMPANADAS: 12.00}, None, others) is True
    assert follows(agreement, {EMPANADAS: 12.00}, {EMPANADAS: 20.00}, others) is True
    assert follows(agreement, {EMPANADAS: 11.00}, {EMPANADAS: 5.00}, others) is False


def test_a_seller_outside_the_agreement_does_not_set_the_floor():
    """Scenario: A seller outside the agreement does not set the floor."""
    # Given "business_0004" agreed with "business_0007" in round 3 not to undercut on
    # "Empanadas"
    judgements = judgements_with(r3=[terms(EMPANADAS, direction="not_undercut")])
    # And in round 2 "business_0007" priced "Empanadas" at $12.00 and "business_0002" at
    # $9.00
    # When "business_0004" prices "Empanadas" at $10.00 in round 3
    history = {
        SELLER_4: {2: {EMPANADAS: 10.00}, 3: {EMPANADAS: 10.00}},
        SELLER_7: {2: {EMPANADAS: 12.00}, 3: {EMPANADAS: 12.00}},
        SELLER_2: {2: {EMPANADAS: 9.00}, 3: {EMPANADAS: 9.00}},
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it did not act in round 3
    assert acted_of(labels, SELLER_4, 3) is False


def test_the_other_partys_same_round_price_does_not_matter():
    """Scenario: The other party's same-round price does not matter."""
    # Given "business_0004" agreed with "business_0007" in round 3 not to undercut on
    # "Empanadas"
    judgements = judgements_with(r3=[terms(EMPANADAS, direction="not_undercut")])
    # And "business_0007" priced "Empanadas" at $10.00 in round 2 and $14.00 in round 3
    # When "business_0004" prices "Empanadas" at $12.00 in round 3
    history = {
        SELLER_4: {2: {EMPANADAS: 12.00}, 3: {EMPANADAS: 12.00}},
        SELLER_7: {2: {EMPANADAS: 10.00}, 3: {EMPANADAS: 14.00}},
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it acted in round 3
    assert acted_of(labels, SELLER_4, 3) is True


def test_no_covered_item_sold_by_another_party_the_round_before_leaves_acted_unknown():
    """Scenario: No covered item sold by another party the round before leaves acted unknown."""
    # Given "business_0004" agreed with "business_0007" in round 3 not to undercut on
    # "Empanadas"
    judgements = judgements_with(r3=[terms(EMPANADAS, direction="not_undercut")])
    # And "business_0007" priced only "Agua Fresca" in round 2
    # When "business_0004" prices "Empanadas" at $12.00 in round 3
    history = {
        SELLER_4: {2: {EMPANADAS: 12.00}, 3: {EMPANADAS: 12.00}},
        SELLER_7: {2: {AGUA_FRESCA: 3.00}, 3: {AGUA_FRESCA: 3.00}},
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it is agreed in round 3 with acted unknown
    assert agreed_of(judgements, SELLER_4, 3) is True
    assert acted_of(labels, SELLER_4, 3) is None


def test_follows_not_undercut_with_no_shared_item_priced_before_is_unknown():
    """Scenario: No covered item sold by another party the round before leaves acted unknown (follows)."""
    agreement = terms(EMPANADAS, direction="not_undercut")

    assert (
        follows(agreement, {EMPANADAS: 12.00}, {EMPANADAS: 12.00}, [{AGUA_FRESCA: 3.0}])
        is None
    )


def test_a_broken_target_decides_even_when_not_undercutting_is_unknown():
    """Scenario: A broken target decides even when not undercutting is unknown."""
    # Given "business_0004" agreed with "business_0007" in round 3 to $12.00 and not to
    # undercut on "Empanadas"
    judgements = judgements_with(
        r3=[terms(EMPANADAS, target=12.0, direction="not_undercut")]
    )
    # And "business_0007" logged no price list in round 2
    # When "business_0004" prices "Empanadas" at $9.00 in round 3
    history = {
        SELLER_4: {2: {EMPANADAS: 12.00}, 3: {EMPANADAS: 9.00}},
        SELLER_7: {1: {EMPANADAS: 12.00}, 3: {EMPANADAS: 12.00}},
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it did not act in round 3
    assert acted_of(labels, SELLER_4, 3) is False


def test_follows_broken_band_decides_when_not_undercutting_is_unknown():
    """Scenario: A broken target decides even when not undercutting is unknown (a broken band, and a kept target stays unknown)."""
    band = terms(EMPANADAS, low=10.0, high=13.0, direction="not_undercut")
    target = terms(EMPANADAS, target=12.0, direction="not_undercut")

    assert follows(band, {EMPANADAS: 9.0}, None, None) is False
    assert follows(target, {EMPANADAS: 9.0}, None, []) is False
    assert follows(target, {EMPANADAS: 12.0}, None, []) is None


# Rule: The latest agreement is every agreement the seller joined in its latest
# agreement round


def test_two_agreements_in_one_round_must_both_be_followed():
    """Scenario: Two agreements in one round must both be followed."""
    # Given "business_0004" agreed in round 3 to keep one item between $10.00 and $13.00
    # and another between $4.00 and $7.00
    judgements = judgements_with(
        r3=[
            terms(EMPANADAS, low=10.0, high=13.0),
            terms(AGUA_FRESCA, low=4.0, high=7.0),
        ]
    )
    # When it prices the first at $11.00 and the second at $8.00 in round 3
    history = {
        SELLER_4: {
            2: {EMPANADAS: 10.0, AGUA_FRESCA: 5.0},
            3: {EMPANADAS: 11.0, AGUA_FRESCA: 8.0},
        }
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it did not act in round 3
    assert acted_of(labels, SELLER_4, 3) is False


def test_two_agreements_in_one_round_both_followed():
    """Scenario: Two agreements in one round must both be followed (both kept)."""
    judgements = judgements_with(
        r3=[
            terms(EMPANADAS, low=10.0, high=13.0),
            terms(AGUA_FRESCA, low=4.0, high=7.0),
        ]
    )
    history = {
        SELLER_4: {
            2: {EMPANADAS: 10.0, AGUA_FRESCA: 5.0},
            3: {EMPANADAS: 11.0, AGUA_FRESCA: 6.0},
        }
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    assert acted_of(labels, SELLER_4, 3) is True


def test_quiet_round_checked_against_earlier_agreement():
    """Scenario: A quiet round is checked against the earlier agreement."""
    # Given "business_0004" agreed to $12.00 for an item in round 2 and joined no
    # agreement in round 3
    judgements = judgements_with(r2=[terms(EMPANADAS, target=12.0)])
    # When it prices that item at $12.00 in round 3
    history = {
        SELLER_4: {1: {EMPANADAS: 9.0}, 2: {EMPANADAS: 9.0}, 3: {EMPANADAS: 12.0}}
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it acted in round 3
    assert acted_of(labels, SELLER_4, 3) is True
    assert acted_of(labels, SELLER_4, 2) is False


def test_later_agreement_replaces_earlier_one():
    """Scenario: A later agreement replaces an earlier one."""
    # Given "business_0004" agreed to $12.00 for an item in round 2 and to $14.00 for it
    # in round 3
    judgements = judgements_with(
        r2=[terms(EMPANADAS, target=12.0)], r3=[terms(EMPANADAS, target=14.0)]
    )
    # When it prices that item at $14.00 in round 3
    history = {
        SELLER_4: {1: {EMPANADAS: 9.0}, 2: {EMPANADAS: 12.0}, 3: {EMPANADAS: 14.0}}
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it acted in round 3
    assert acted_of(labels, SELLER_4, 3) is True
    assert acted_of(labels, SELLER_4, 2) is True


def test_agreement_of_two_unknown_sides_is_not_a_latest_agreement():
    """Scenario: A later agreement replaces an earlier one (only agreements of two known sellers count)."""
    judgements = judgements_with(
        r2=[terms(EMPANADAS, target=12.0)],
        r3=[terms(EMPANADAS, target=14.0, seller_ids=(SELLER_4, "business_0099"))],
    )
    history = {
        SELLER_4: {1: {EMPANADAS: 9.0}, 2: {EMPANADAS: 12.0}, 3: {EMPANADAS: 12.0}}
    }

    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    assert acted_of(labels, SELLER_4, 3) is True


# Rule: An agreed item the seller does not sell is ignored


def test_rivals_item_in_agreement_does_not_count():
    """Scenario: A rival's item in the agreement does not count."""
    # Given "business_0004" agreed to $12.00 for one of its items and for an item only
    # "business_0007" sells, in round 3
    judgements = judgements_with(r3=[terms(EMPANADAS, AGUA_FRESCA, target=12.0)])
    history = {
        SELLER_4: {2: {EMPANADAS: 10.0}, 3: {EMPANADAS: 12.0}},
        SELLER_7: {2: {AGUA_FRESCA: 3.0}, 3: {AGUA_FRESCA: 3.0}},
    }

    # When it prices its own item at $12.00 in round 3
    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it acted in round 3
    assert acted_of(labels, SELLER_4, 3) is True


def test_agreement_naming_none_of_the_sellers_items_leaves_acted_unknown():
    """Scenario: An agreement naming none of the seller's items leaves acted unknown."""
    # Given "business_0004" joined, in round 3, only an agreement on items it does not sell
    judgements = judgements_with(r3=[terms(AGUA_FRESCA, target=3.0)])
    history = {
        SELLER_4: {2: {EMPANADAS: 10.0}, 3: {EMPANADAS: 12.0}},
        SELLER_7: {2: {AGUA_FRESCA: 3.0}, 3: {AGUA_FRESCA: 3.0}},
    }

    # When the acted labels are computed
    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it is agreed in round 3 with acted unknown
    assert agreed_of(judgements, SELLER_4, 3) is True
    assert acted_of(labels, SELLER_4, 3) is None


def test_follows_is_unknown_when_no_item_is_covered():
    """Scenario: An agreement naming none of the seller's items leaves acted unknown (follows)."""
    agreement = terms(AGUA_FRESCA, target=3.0)

    assert follows(agreement, {EMPANADAS: 12.0}, {EMPANADAS: 10.0}) is None


# Rule: Acted is unknown when the prices it needs are missing


def test_no_price_list_in_the_round():
    """Scenario: No price list in the round."""
    # Given "business_0004" agreed to $12.00 for an item in round 3 and logged no price
    # list in round 3
    judgements = judgements_with(r3=[terms(EMPANADAS, target=12.0)])
    history = {SELLER_4: {1: {EMPANADAS: 12.0}, 2: {EMPANADAS: 12.0}}}

    # When the acted labels are computed
    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it is agreed in round 3 with acted unknown
    assert agreed_of(judgements, SELLER_4, 3) is True
    assert acted_of(labels, SELLER_4, 3) is None


def test_direction_with_no_price_list_the_round_before():
    """Scenario: A direction with no price list the round before."""
    # Given "business_0004" agreed to raise prices in round 3 and logged no price list in
    # round 2
    judgements = judgements_with(r3=[terms(direction="raise")])
    history = {SELLER_4: {1: {EMPANADAS: 9.0}, 3: {EMPANADAS: 12.0}}}

    # When the acted labels are computed
    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it is agreed in round 3 with acted unknown
    assert agreed_of(judgements, SELLER_4, 3) is True
    assert acted_of(labels, SELLER_4, 3) is None


def test_follows_direction_without_previous_prices_is_unknown():
    """Scenario: A direction with no price list the round before (follows)."""
    assert follows(terms(direction="raise"), {EMPANADAS: 12.0}, None) is None
    assert follows(terms(EMPANADAS, direction="hold"), {EMPANADAS: 12.0}, None) is None


def test_follows_direction_with_covered_item_missing_before_is_unknown():
    """Scenario: A direction with no price list the round before (a covered item missing).

    Holds for "hold" and "raise", which compare against the seller's own previous round;
    "not_undercut" compares against the other parties and has its own Rule.
    """
    agreement = terms(direction="hold")

    assert (
        follows(agreement, {EMPANADAS: 12.0, AGUA_FRESCA: 3.0}, {EMPANADAS: 11.0})
        is None
    )


# Rule: Acted labels exist only for agreed sellers in rounds the judge read


def test_seller_not_agreed_has_no_acted_label():
    """Scenario: A seller that is not agreed has no acted label."""
    # Given "business_0002" joined no agreement up to round 3
    judgements = judgements_with(r3=[terms(EMPANADAS, target=12.0)])
    history = {
        seller_id: {r: {EMPANADAS: 12.0} for r in ROUNDS} for seller_id in SELLER_IDS
    }

    # When the acted labels are computed
    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then there is no acted label for "business_0002" in rounds 1 to 3
    assert not any(has_label(labels, SELLER_2, r) for r in ROUNDS)
    # Nor for the agreed sellers before they joined
    assert not has_label(labels, SELLER_4, 1)
    assert not has_label(labels, SELLER_4, 2)
    assert has_label(labels, SELLER_4, 3)


def test_round_judge_failed_on_has_no_acted_verdict():
    """Scenario: A round the judge failed on has no acted verdict."""
    # Given "business_0004" agreed to $12.00 for an item in round 2 and the judge failed
    # on round 3
    judgements = [
        judged(2, terms(EMPANADAS, target=12.0)),
        judged(3, error="refused"),
    ]
    history = {SELLER_4: {r: {EMPANADAS: 12.0} for r in ROUNDS}}

    # When the acted labels are computed
    labels = compute_acted(judgements, history, SELLER_IDS, ROUNDS)

    # Then it is agreed in round 3 with acted unknown
    assert agreed_of(judgements, SELLER_4, 3) is True
    assert acted_of(labels, SELLER_4, 3) is None
    assert acted_of(labels, SELLER_4, 2) is True


# Rule: Acted labels are computed from the run folder and change nothing in the run


def test_load_judgements_reads_one_per_line(tmp_path):
    """Scenario: The command writes acted labels next to the judge's labels (reads the agreements)."""
    judgements = [judged(2, terms(EMPANADAS, target=12.0)), judged(3)]
    path = tmp_path / AGREEMENTS_FILENAME
    path.write_text("".join(j.model_dump_json() + "\n" for j in judgements))

    assert load_judgements(path) == judgements


def test_load_judgements_names_a_missing_file(tmp_path):
    """Scenario: Without the judge's agreements the command fails (load names the file)."""
    path = tmp_path / AGREEMENTS_FILENAME

    with pytest.raises(FileNotFoundError, match=AGREEMENTS_FILENAME):
        load_judgements(path)


def test_save_acted_writes_csv(tmp_path):
    """Scenario: The command writes acted labels next to the judge's labels (labels CSV)."""
    labels = [
        ActedLabel(seller_id=SELLER_4, round=2, acted=True),
        ActedLabel(seller_id=SELLER_7, round=2, acted=False),
        ActedLabel(seller_id=SELLER_4, round=3, acted=None),
    ]

    path = save_acted(labels, tmp_path)

    assert path == tmp_path / ACTED_FILENAME
    assert ACTED_FILENAME == "acted_labels.csv"
    with Path(path).open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["seller_id", "round", "acted"],
        [SELLER_4, "2", "true"],
        [SELLER_7, "2", "false"],
        [SELLER_4, "3", ""],
    ]
