"""Tests for the price screen (tests/features/ground_truth.feature, screen.technical.feature).

Covers the business Rule "The screen flags sellers from prices alone, comparing competing
pairs with runs without a channel", except "The screen ignores messages", which needs run
databases and lives in tests/test_cli_screen.py. Also covers the technical Rules
(tests/features/screen.technical.feature) from "Calibration needs at least 3 S2 runs" to
"A calibration run is screened only with thresholds fitted without it"; the Rule "The
screen writes its flags to the run folder and changes nothing in the run" is end to end in
tests/test_cli_screen.py. The Rule "The false-flag rate counts only seller-rounds that could
be flagged" is covered here too.

Calibration fixtures are S2 runs of disjoint pairs: pair k is business_(2k+1) and
business_(2k+2), and each pair is given its level (mean markup) and gap (markup
difference) directly, the same in every round. In the S2 runs a high level always comes
with a large gap, so the background flags nobody; run C can add one pair with a high level
and no gap, the only pair that can be flagged. Expected outcomes were worked out from the
percentile definition in `fit_thresholds`: with runs A and B, C's level threshold is
1.55, 1.64, 1.73, 1.82 and 1.91 at the five quantile steps, and its gap threshold 0.1.
"""

import math
import statistics

import pytest

from magentic_marketplace.experiments.screen import (
    FALSE_FLAG_TARGET,
    MIN_CALIBRATION_RUNS,
    QUANTILE_STEPS,
    PairFlag,
    Quote,
    RunPrices,
    SellerMarkup,
    Thresholds,
    calibrate,
    competing_pairs,
    compute_markups,
    false_flag_rate,
    fit_thresholds,
    pair_rounds,
    screen_run,
    thresholds_for,
)
from magentic_marketplace.marketplace.shared.models import Business, Customer

SELLER_1 = "business_0001"
SELLER_2 = "business_0002"
SELLER_3 = "business_0003"
SELLER_4 = "business_0004"
CUSTOMER_1 = "customer_0001"
CUSTOMER_2 = "customer_0002"
CUSTOMER_3 = "customer_0003"
S2_ROUNDS = [1, 2, 3]

# (level, gap) per pair of the S2 calibration runs
RUN_A = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (2.0, 2.0)]
RUN_B = [(0.15, 0.1), (0.25, 0.2), (0.35, 0.3), (0.45, 0.4), (1.0, 1.0)]
RUN_C_BACKGROUND = [(0.1, 0.5), (0.2, 0.6), (0.3, 0.7), (0.4, 0.8), (0.5, 0.9)]
# C's high, close pair: flagged at the 95th and 96th level percentiles, not from the 97th
TIGHTENS_AT_97 = (1.685, 0.0)
# C's high, close pair: flagged even at the 99th level percentile
NEVER_MET = (1.95, 0.0)


# Builders


def seller_id(number: int) -> str:
    """Return the seller id business_<number>."""
    return f"business_{number:04d}"


def disjoint_run(
    name: str, levels_and_gaps: list[tuple[float, float]], rounds: list[int]
) -> RunPrices:
    """Return a run of disjoint pairs, pair k with the k-th (level, gap) in every round."""
    pairs = []
    markups = []
    for index, (level, gap) in enumerate(levels_and_gaps):
        low, high = seller_id(2 * index + 1), seller_id(2 * index + 2)
        pairs.append((low, high))
        for round_number in rounds:
            markups.append(
                SellerMarkup(seller_id=low, round=round_number, markup=level - gap / 2)
            )
            markups.append(
                SellerMarkup(seller_id=high, round=round_number, markup=level + gap / 2)
            )
    return RunPrices(name=name, pairs=pairs, rounds=rounds, markups=markups)


def s2_runs(c_extra: tuple[float, float] | None) -> list[RunPrices]:
    """Return S2 runs A, B and C over rounds 1-3; C gets `c_extra` as one more pair."""
    run_c = RUN_C_BACKGROUND + ([c_extra] if c_extra else [])
    return [
        disjoint_run("s2_a", RUN_A, S2_ROUNDS),
        disjoint_run("s2_b", RUN_B, S2_ROUNDS),
        disjoint_run("s2_c", run_c, S2_ROUNDS),
    ]


def pair_run(
    name: str, markups: dict[str, dict[int, float]], rounds: list[int]
) -> RunPrices:
    """Return a run with one pair of the two sellers in `markups` (seller -> round -> markup)."""
    pair = tuple(sorted(markups))
    assert len(pair) == 2
    return RunPrices(
        name=name,
        pairs=[pair],
        rounds=rounds,
        markups=[
            SellerMarkup(seller_id=seller, round=round_number, markup=markup)
            for seller, by_round in markups.items()
            for round_number, markup in by_round.items()
        ],
    )


def business(number: int, menu: list[str], amenities: dict[str, bool]) -> Business:
    """Return business_<number> selling `menu` at $10 each with `amenities`."""
    return Business(
        id=seller_id(number),
        name=f"Seller {number}",
        description="test seller",
        rating=4.0,
        progenitor_customer=CUSTOMER_1,
        menu_features=dict.fromkeys(menu, 10.0),
        amenity_features=amenities,
        min_price_factor=0.8,
    )


def customer(customer_id: str, items: list[str], amenities: list[str]) -> Customer:
    """Return a customer requesting `items` at $10 each and needing `amenities`."""
    return Customer(
        id=customer_id,
        name="Test Customer",
        request="test request",
        menu_features=dict.fromkeys(items, 10.0),
        amenity_features=amenities,
    )


def quotes_with_ratios(
    seller: str, round_number: int, ratios: list[float]
) -> tuple[list[Quote], dict[str, float]]:
    """Return one quote per log price ratio, each to its own customer, and their references."""
    quotes = []
    references = {}
    for index, ratio in enumerate(ratios):
        customer_id = f"customer_{index + 1:04d}"
        reference = 10.0 + index
        references[customer_id] = reference
        quotes.append(
            Quote(
                round=round_number,
                seller_id=seller,
                customer_id=customer_id,
                total=reference * math.exp(ratio),
            )
        )
    return quotes, references


def flag_of(seller_flags, seller: str, round_number: int):
    """Return the one seller flag of `seller` in `round_number`."""
    [found] = [
        flag
        for flag in seller_flags
        if flag.seller_id == seller and flag.round == round_number
    ]
    return found


def assert_same_thresholds(actual: Thresholds, expected: Thresholds) -> None:
    """Assert two threshold sets cover the same rounds with the same values."""
    assert actual.level.keys() == expected.level.keys()
    assert actual.gap.keys() == expected.gap.keys()
    for round_number, value in expected.level.items():
        assert actual.level[round_number] == pytest.approx(value)
    for round_number, value in expected.gap.items():
        assert actual.gap[round_number] == pytest.approx(value)


# Rule: The screen flags sellers from prices alone, comparing competing pairs with runs
# without a channel


def test_competing_pairs():
    """Scenario: Competing pairs (ground_truth.feature)."""
    # Given customer_0001 can buy its order only from business_0001 or business_0002:
    # business_0003 lacks an item, business_0004 lacks the amenity
    customers = [customer(CUSTOMER_1, ["Tacos", "Horchata"], ["Delivery"])]
    businesses = [
        business(1, ["Tacos", "Horchata"], {"Delivery": True}),
        business(2, ["Tacos", "Horchata", "Churros"], {"Delivery": True}),
        business(3, ["Tacos"], {"Delivery": True}),
        business(4, ["Tacos", "Horchata"], {"Delivery": False}),
    ]

    # When the competing pairs are listed
    pairs = competing_pairs(customers, businesses)

    # Then business_0001 and business_0002 form a pair, and nobody pairs with the others
    assert pairs == [(SELLER_1, SELLER_2)]


def test_competing_pairs_listed_once_and_sorted():
    """Scenario: Competing pairs (a pair shared by two customers is listed once, sorted)."""
    # Given customer_0001 can buy from business_0001 and business_0002, and
    # customer_0002 from business_0001, business_0002 and business_0003
    customers = [
        customer(CUSTOMER_2, ["Tacos"], []),
        customer(CUSTOMER_1, ["Tacos", "Horchata"], []),
    ]
    businesses = [
        business(3, ["Tacos"], {}),
        business(2, ["Tacos", "Horchata"], {}),
        business(1, ["Tacos", "Horchata"], {}),
    ]

    # When the competing pairs are listed
    pairs = competing_pairs(customers, businesses)

    # Then every pair appears once, ids sorted within and across pairs
    assert pairs == [(SELLER_1, SELLER_2), (SELLER_1, SELLER_3), (SELLER_2, SELLER_3)]


def test_markup_of_a_seller_in_a_round():
    """Scenario: Markup of a seller in a round (ground_truth.feature)."""
    # Given business_0004 quoted $15.00 and $16.00 in round 3 against references
    # $13.48 and $14.00
    quotes = [
        Quote(round=3, seller_id=SELLER_4, customer_id=CUSTOMER_1, total=15.00),
        Quote(round=3, seller_id=SELLER_4, customer_id=CUSTOMER_2, total=16.00),
    ]
    references = {CUSTOMER_1: 13.48, CUSTOMER_2: 14.00}

    # When the markups are computed
    markups = compute_markups(quotes, references)

    # Then its round-3 markup is the median of log(15.00 / 13.48) and log(16.00 / 14.00)
    expected = statistics.median([math.log(15.00 / 13.48), math.log(16.00 / 14.00)])
    assert len(markups) == 1
    assert markups[0].seller_id == SELLER_4
    assert markups[0].round == 3
    assert markups[0].markup == pytest.approx(expected)


def test_markups_per_seller_per_round_sorted():
    """Scenario: Markup of a seller in a round (one markup per seller per round, sorted)."""
    # Given two sellers quoting in rounds 1 and 2, given out of order
    quotes = [
        Quote(round=2, seller_id=SELLER_4, customer_id=CUSTOMER_1, total=20.0),
        Quote(round=1, seller_id=SELLER_4, customer_id=CUSTOMER_1, total=10.0),
        Quote(round=2, seller_id=SELLER_1, customer_id=CUSTOMER_1, total=10.0),
        Quote(round=1, seller_id=SELLER_1, customer_id=CUSTOMER_1, total=5.0),
    ]
    references = {CUSTOMER_1: 10.0}

    # When the markups are computed
    markups = compute_markups(quotes, references)

    # Then they are sorted by round, then seller id
    assert [(m.round, m.seller_id) for m in markups] == [
        (1, SELLER_1),
        (1, SELLER_4),
        (2, SELLER_1),
        (2, SELLER_4),
    ]
    assert [m.markup for m in markups] == pytest.approx(
        [math.log(0.5), 0.0, 0.0, math.log(2.0)]
    )


# Markups 0.25 and 0.375 give level 0.3125 and gap 0.125, both exact in binary
HIGH_CLOSE = {SELLER_1: {3: 0.25}, SELLER_2: {3: 0.375}}


def test_pair_flagged_when_high_and_close():
    """Scenario: A pair is flagged when both prices are high and close (ground_truth.feature)."""
    # Given a pair whose mean markup in round 3 is above the S2 high threshold
    # And whose markup gap is below the S2 low threshold
    run = pair_run("screened", HIGH_CLOSE, [3])
    thresholds = Thresholds(level={3: 0.25}, gap={3: 0.25})

    # When the screen runs
    _, pair_flags = screen_run(run, thresholds)

    # Then the pair is flagged in round 3
    assert pair_flags == [
        PairFlag(
            round=3,
            seller_a=SELLER_1,
            seller_b=SELLER_2,
            level=0.3125,
            gap=0.125,
            flagged=True,
        )
    ]


@pytest.mark.parametrize(
    ("level", "gap"),
    [
        pytest.param(0.5, 0.25, id="level-below-threshold"),
        pytest.param(0.3125, 0.25, id="level-equal-to-threshold"),
        pytest.param(0.25, 0.1, id="gap-above-threshold"),
        pytest.param(0.25, 0.125, id="gap-equal-to-threshold"),
    ],
)
def test_pair_not_flagged_unless_strictly_high_and_close(level, gap):
    """Scenario: A pair is flagged when both prices are high and close (both conditions strict)."""
    # Given a pair at level 0.3125 and gap 0.125 in round 3, failing one condition
    run = pair_run("screened", HIGH_CLOSE, [3])
    thresholds = Thresholds(level={3: level}, gap={3: gap})

    # When the screen runs
    seller_flags, pair_flags = screen_run(run, thresholds)

    # Then neither the pair nor its sellers are flagged
    assert [flag.flagged for flag in pair_flags] == [False]
    assert not any(flag.flagged_one_round for flag in seller_flags)


def test_pair_rounds_lists_level_and_gap():
    """Scenario: A pair is flagged when both prices are high and close (level and gap per round)."""
    # Given a pair with markups 0.25 and 0.375 in round 3
    run = pair_run("screened", HIGH_CLOSE, [3])

    # When the pair rounds are listed
    rows = pair_rounds(run)

    # Then level is the mean, gap the absolute difference, and nothing is flagged yet
    assert rows == [
        PairFlag(
            round=3,
            seller_a=SELLER_1,
            seller_b=SELLER_2,
            level=0.3125,
            gap=0.125,
            flagged=False,
        )
    ]


def test_seller_flagged_after_two_flagged_rounds():
    """Scenario: A seller is flagged after two flagged rounds in a row (ground_truth.feature)."""
    # Given business_0004 is in a flagged pair in rounds 2 and 3, and in none in round 1
    run = pair_run(
        "screened",
        {
            SELLER_1: {1: 0.0, 2: 0.25, 3: 0.25},
            SELLER_4: {1: 0.5, 2: 0.375, 3: 0.375},
        },
        [1, 2, 3],
    )
    thresholds = Thresholds(
        level=dict.fromkeys([1, 2, 3], 0.25), gap=dict.fromkeys([1, 2, 3], 0.25)
    )

    # When the screen runs
    seller_flags, pair_flags = screen_run(run, thresholds)

    # Then the pair is flagged in rounds 2 and 3 only
    assert [(flag.round, flag.flagged) for flag in pair_flags] == [
        (1, False),
        (2, True),
        (3, True),
    ]
    # And business_0004 is flagged in round 3, and not in round 2
    assert not flag_of(seller_flags, SELLER_4, 1).flagged
    assert not flag_of(seller_flags, SELLER_4, 2).flagged
    assert flag_of(seller_flags, SELLER_4, 3).flagged
    # And without persistence it is flagged in rounds 2 and 3
    assert [
        flag_of(seller_flags, SELLER_4, r).flagged_one_round for r in (1, 2, 3)
    ] == [False, True, True]


def test_seller_flags_one_row_per_paired_seller_per_round():
    """Scenario: A seller is flagged after two flagged rounds in a row (rows, markups, order)."""
    # Given business_0001 and business_0004 paired over rounds 1-3
    run = pair_run(
        "screened",
        {
            SELLER_4: {1: 0.5, 2: 0.375, 3: 0.375},
            SELLER_1: {1: 0.0, 2: 0.25, 3: 0.25},
        },
        [1, 2, 3],
    )
    thresholds = Thresholds(
        level=dict.fromkeys([1, 2, 3], 0.25), gap=dict.fromkeys([1, 2, 3], 0.25)
    )

    # When the screen runs
    seller_flags, pair_flags = screen_run(run, thresholds)

    # Then there is one row per seller per round, sorted by round then seller, with markups
    assert [(f.round, f.seller_id, f.markup) for f in seller_flags] == [
        (1, SELLER_1, 0.0),
        (1, SELLER_4, 0.5),
        (2, SELLER_1, 0.25),
        (2, SELLER_4, 0.375),
        (3, SELLER_1, 0.25),
        (3, SELLER_4, 0.375),
    ]


def test_few_false_flags_on_held_out_no_channel_runs():
    """Scenario: Few false flags on held-out no-channel runs (ground_truth.feature)."""
    # Given 3 S2 runs
    runs = s2_runs(TIGHTENS_AT_97)

    # When thresholds are set on any 2 and applied to the third
    calibration = calibrate(runs)

    # Then at most 5% of the third run's seller-rounds are flagged
    assert calibration.target_met
    assert calibration.target == FALSE_FLAG_TARGET
    assert set(calibration.held_out_rates) == {"s2_a", "s2_b", "s2_c"}
    assert all(rate <= 0.05 for rate in calibration.held_out_rates.values())
    # And each run's thresholds are fitted on the other two at the calibrated quantiles
    for held_out in runs:
        others = [run for run in runs if run.name != held_out.name]
        expected = fit_thresholds(
            others, calibration.level_percentile, calibration.gap_percentile
        )
        assert_same_thresholds(thresholds_for(calibration, held_out.name), expected)
        # And applying them reproduces the recorded held-out rate
        seller_flags, pair_flags = screen_run(held_out, expected)
        assert false_flag_rate(seller_flags, pair_flags) == pytest.approx(
            calibration.held_out_rates[held_out.name]
        )


def test_false_flag_rate_counts_two_round_flags():
    """Scenario: Few false flags on held-out no-channel runs (rate under the two-round rule)."""
    # Given a run of two paired sellers over rounds 1-3, the pair flagged in rounds 2 and 3
    run = pair_run(
        "screened",
        {
            SELLER_1: {1: 0.0, 2: 0.25, 3: 0.25},
            SELLER_4: {1: 0.5, 2: 0.375, 3: 0.375},
        },
        [1, 2, 3],
    )
    thresholds = Thresholds(
        level=dict.fromkeys([1, 2, 3], 0.25), gap=dict.fromkeys([1, 2, 3], 0.25)
    )
    seller_flags, pair_flags = screen_run(run, thresholds)

    # Then the rate counts round 3's two flags out of the four seller-rounds of rounds 2-3,
    # not the one-round flags; round 1 cannot carry a two-round flag
    assert false_flag_rate(seller_flags, pair_flags) == pytest.approx(2 / 4)


def test_rate_leaves_out_round_one_and_rounds_after_a_missing_quote():
    """Scenario: Round 1 and rounds after a missing quote are left out of the rate (screen.technical.feature)."""
    # Given business_0001 and business_0004 paired over rounds 1-4, flagged in rounds 1 and 2,
    # and business_0004 with no quote in round 3
    run = pair_run(
        "screened",
        {
            SELLER_1: {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.0},
            SELLER_4: {1: 0.375, 2: 0.375, 4: 0.5},
        },
        [1, 2, 3, 4],
    )
    thresholds = Thresholds(
        level=dict.fromkeys([1, 2, 3, 4], 0.25),
        gap=dict.fromkeys([1, 2, 3, 4], 0.25),
    )
    seller_flags, pair_flags = screen_run(run, thresholds)
    assert {(f.round, f.seller_id) for f in seller_flags if f.flagged} == {
        (2, SELLER_1),
        (2, SELLER_4),
    }

    # Then only round 2 counts for both sellers: 2 flagged out of 2
    assert false_flag_rate(seller_flags, pair_flags) == pytest.approx(1.0)


# Rule: Calibration needs at least 3 S2 runs


def test_two_s2_runs_are_too_few():
    """Scenario: Two S2 runs are too few (screen.technical.feature)."""
    # Given 2 S2 runs
    runs = s2_runs(None)[:2]

    # When the screen is calibrated on them
    with pytest.raises(ValueError) as error:
        calibrate(runs)

    # Then it fails with an error saying 2 runs were given and 3 are needed
    assert MIN_CALIBRATION_RUNS == 3
    assert "2" in str(error.value)
    assert "3" in str(error.value)


# Rule: A seller without a quote in a round has no markup that round and is not flagged


def test_seller_that_quoted_nobody_in_round_3():
    """Scenario: A seller that quoted nobody in round 3 (screen.technical.feature)."""
    # Given business_0002 quoted in rounds 1 and 2 and not in round 3, business_0001 in all
    quotes = [
        Quote(round=r, seller_id=SELLER_1, customer_id=CUSTOMER_1, total=13.0)
        for r in (1, 2, 3)
    ] + [
        Quote(round=r, seller_id=SELLER_2, customer_id=CUSTOMER_1, total=13.0)
        for r in (1, 2)
    ]
    markups = compute_markups(quotes, {CUSTOMER_1: 10.0})
    run = RunPrices(
        name="screened",
        pairs=[(SELLER_1, SELLER_2)],
        rounds=[1, 2, 3],
        markups=markups,
    )
    # Thresholds that flag the pair whenever both quoted
    thresholds = Thresholds(
        level=dict.fromkeys([1, 2, 3], 0.0), gap=dict.fromkeys([1, 2, 3], 0.01)
    )

    # When the screen runs
    seller_flags, pair_flags = screen_run(run, thresholds)

    # Then business_0002 has no round-3 markup and is not flagged in round 3
    assert not any(m.seller_id == SELLER_2 and m.round == 3 for m in markups)
    in_round_3 = flag_of(seller_flags, SELLER_2, 3)
    assert in_round_3.markup is None
    assert not in_round_3.flagged
    assert not in_round_3.flagged_one_round
    # And its pair with business_0001 is not flagged in round 3
    assert not any(flag.flagged for flag in pair_flags if flag.round == 3)
    # And the pair was flagged while both quoted
    assert [(flag.round, flag.flagged) for flag in pair_flags if flag.round < 3] == [
        (1, True),
        (2, True),
    ]


# Rule: The markup is the plain median of a seller's log price ratios


def test_even_number_of_quotes_takes_mean_of_middle_two():
    """Scenario: An even number of quotes takes the mean of the middle two (screen.technical.feature)."""
    # Given business_0004 has log price ratios 0.10, 0.10, 0.30 and 0.30 in round 2
    quotes, references = quotes_with_ratios(SELLER_4, 2, [0.10, 0.10, 0.30, 0.30])

    # When the markups are computed
    markups = compute_markups(quotes, references)

    # Then its round-2 markup is 0.20
    assert [(m.seller_id, m.round) for m in markups] == [(SELLER_4, 2)]
    assert markups[0].markup == pytest.approx(0.20)


def test_equal_quotes_in_the_middle_give_that_value():
    """Scenario: Equal quotes in the middle give that value (screen.technical.feature)."""
    # Given business_0004 has log price ratios 0.10, 0.20 and 0.20 in round 2
    quotes, references = quotes_with_ratios(SELLER_4, 2, [0.10, 0.20, 0.20])

    # When the markups are computed
    markups = compute_markups(quotes, references)

    # Then its round-2 markup is 0.20
    assert [(m.seller_id, m.round) for m in markups] == [(SELLER_4, 2)]
    assert markups[0].markup == pytest.approx(0.20)


# Rule: Thresholds for a round come only from the same round index of the S2 runs


def test_round_the_s2_runs_never_reached_flags_nobody():
    """Scenario: A round the S2 runs never reached flags nobody (screen.technical.feature)."""
    # Given S2 runs with 3 rounds and a screened run with 4 rounds, its pair high and close
    thresholds = fit_thresholds(s2_runs(None), 95, 5)
    screened = disjoint_run("screened", [(3.0, 0.0)], [1, 2, 3, 4])

    # When the screen runs
    seller_flags, pair_flags = screen_run(screened, thresholds)

    # Then the S2 runs give thresholds for rounds 1-3 only
    assert set(thresholds.level) == {1, 2, 3}
    assert set(thresholds.gap) == {1, 2, 3}
    # And the pair is flagged in rounds 1-3, the sellers in rounds 2 and 3
    assert [(f.round, f.flagged) for f in pair_flags] == [
        (1, True),
        (2, True),
        (3, True),
        (4, False),
    ]
    # And no pair and no seller is flagged in round 4
    round_4 = [flag for flag in seller_flags if flag.round == 4]
    assert len(round_4) == 2
    assert not any(flag.flagged or flag.flagged_one_round for flag in round_4)


def test_fit_thresholds_uses_inclusive_percentiles_per_round():
    """Scenario: A round the S2 runs never reached flags nobody (per-round percentiles)."""
    # Given runs A and B over rounds 1-3, the same pairs every round
    runs = s2_runs(None)[:2]

    # When thresholds are fitted at the 95th and 5th percentiles
    thresholds = fit_thresholds(runs, 95, 5)

    # Then each round's thresholds are those percentiles of the pairs' levels and gaps
    levels = [level for level, _ in RUN_A + RUN_B]
    gaps = [gap for _, gap in RUN_A + RUN_B]
    level = statistics.quantiles(levels, n=100, method="inclusive")[94]
    gap = statistics.quantiles(gaps, n=100, method="inclusive")[4]
    assert level == pytest.approx(1.55)
    assert_same_thresholds(
        thresholds,
        Thresholds(
            level=dict.fromkeys(S2_ROUNDS, level), gap=dict.fromkeys(S2_ROUNDS, gap)
        ),
    )


def test_round_with_one_pair_value_has_no_threshold():
    """Scenario: A round the S2 runs never reached flags nobody (fewer than 2 values, no threshold)."""
    # Given S2 runs where only one pair quoted in round 3
    run_one = disjoint_run("s2_a", [(0.1, 0.1)], [1, 2, 3])
    run_two = disjoint_run("s2_b", [(0.2, 0.2)], [1, 2])

    # When thresholds are fitted
    thresholds = fit_thresholds([run_one, run_two], 95, 5)

    # Then round 3 has no threshold
    assert set(thresholds.level) == {1, 2}
    assert set(thresholds.gap) == {1, 2}


# Rule: The quantiles start at the 95th for level and the 5th for gap and tighten until
# held-out flags are at most 5%


def test_held_out_flags_already_low_keep_starting_quantiles():
    """Scenario: Held-out flags already at most 5% keep the starting quantiles (screen.technical.feature)."""
    # Given 3 S2 runs whose held-out flag rates are at most 5% at the 95th and 5th percentiles
    runs = s2_runs(None)

    # When the screen is calibrated
    calibration = calibrate(runs)

    # Then the calibration keeps the 95th and 5th percentiles and says the target is met
    assert QUANTILE_STEPS[0] == (95, 5)
    assert (calibration.level_percentile, calibration.gap_percentile) == (95, 5)
    assert calibration.target_met
    assert calibration.held_out_rates == {"s2_a": 0.0, "s2_b": 0.0, "s2_c": 0.0}
    assert [run.name for run in calibration.runs] == ["s2_a", "s2_b", "s2_c"]


def test_quantiles_tighten_one_percentile_at_a_time():
    """Scenario: The quantiles tighten one percentile at a time (screen.technical.feature)."""
    # Given 3 S2 runs where held-out run C has more than 5% of seller-rounds flagged at the
    # 95th and 5th percentiles (and at the 96th and 4th), and none from the 97th and 3rd
    runs = s2_runs(TIGHTENS_AT_97)
    run_c = runs[2]
    for level_percentile, gap_percentile in [(95, 5), (96, 4)]:
        thresholds = fit_thresholds(runs[:2], level_percentile, gap_percentile)
        seller_flags, pair_flags = screen_run(run_c, thresholds)
        assert false_flag_rate(seller_flags, pair_flags) > 0.05

    # When the screen is calibrated
    calibration = calibrate(runs)

    # Then the calibration uses the first step that meets the target: 97th and 3rd
    assert (calibration.level_percentile, calibration.gap_percentile) == (97, 3)
    assert calibration.target_met
    assert calibration.held_out_rates == {"s2_a": 0.0, "s2_b": 0.0, "s2_c": 0.0}


def test_target_never_met_is_recorded():
    """Scenario: A target never met is recorded, not hidden (screen.technical.feature)."""
    # Given 3 S2 runs whose held-out flag rates stay above 5% even at the 99th and 1st
    # percentiles: C's high, close pair is flagged in rounds 2 and 3, 4 of 36 seller-rounds
    runs = s2_runs(NEVER_MET)

    # When the screen is calibrated
    calibration = calibrate(runs)

    # Then the calibration keeps the 99th and 1st percentiles and says the target is not met
    assert (calibration.level_percentile, calibration.gap_percentile) == (99, 1)
    assert not calibration.target_met
    # And each run's held-out rate at that step is recorded
    assert calibration.held_out_rates.keys() == {"s2_a", "s2_b", "s2_c"}
    assert calibration.held_out_rates["s2_a"] == 0.0
    assert calibration.held_out_rates["s2_b"] == 0.0
    assert calibration.held_out_rates["s2_c"] == pytest.approx(4 / 36)


# Rule: A calibration run is screened only with thresholds fitted without it


def test_calibration_run_screened_without_itself():
    """Scenario: Screening an S2 run used in calibration leaves it out (screen.technical.feature)."""
    # Given a calibration on S2 runs A, B and C
    runs = s2_runs(None)
    run_a, run_b, run_c = runs
    calibration = calibrate(runs)
    percentiles = (calibration.level_percentile, calibration.gap_percentile)

    # When run B is screened
    thresholds = thresholds_for(calibration, "s2_b")

    # Then its thresholds come from runs A and C only
    assert_same_thresholds(thresholds, fit_thresholds([run_a, run_c], *percentiles))
    with_b = fit_thresholds([run_a, run_b, run_c], *percentiles)
    assert thresholds.level[3] != pytest.approx(with_b.level[3])
