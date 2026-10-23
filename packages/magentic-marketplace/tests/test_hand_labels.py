"""Tests for checking the judge against hand labels (tests/features/ground_truth.feature).

Covers the business Rule "The judge is checked against Ivan's hand labels", and the
technical Rules (tests/features/ground_truth.technical.feature) from "A thread is one round
of messages between one pair of sellers" to "A label that cannot be read is reported, never
guessed". The Rule "Hand labels are drawn and checked from run folders and change nothing
in a run" is covered end to end in tests/test_cli_hand_labels.py.

Threads, posts and judgements are built directly. Judge answers have the shape
`draw_sample` and `check_labels` take: thread id -> the sellers the judge names (empty for
none). A filled sample file is the text `format_sample` writes with Ivan's labels written
onto its "Agreed:" lines. Judge quotes, agreement items and target prices are marker
strings no message holds, so finding one in a sample file means the judge leaked into it.
"""

import re
from pathlib import Path

import pytest

from magentic_marketplace.experiments.hand_labels import (
    NO_SELLER,
    SampleFile,
    Thread,
    build_threads,
    check_labels,
    draw_sample,
    format_sample,
    judge_answer,
    make_thread_id,
    parse_sample,
    run_label_sample,
)
from magentic_marketplace.experiments.judge import (
    AGREEMENTS_FILENAME,
    Agreement,
    ChannelPost,
    RoundJudgement,
    check_agreement,
)
from magentic_marketplace.platform.database.sqlite import connect_to_sqlite_database

SELLER_2 = "business_0002"
SELLER_4 = "business_0004"
SELLER_7 = "business_0007"
SELLER_IDS = {SELLER_2, SELLER_4, SELLER_7}
RUN_A = "run_a"
RUN_B = "run_b"
HOLD_TEXT = "hold empanadas at $12"
JUDGE_QUOTE = "JUDGE-QUOTE we lock it in"
JUDGE_ITEM = "JUDGE-ITEM Tamales Oaxaquenos"
JUDGE_TARGET = 43.21


# Builders


def post(
    sender: str, recipients: list[str], round_number: int, turn: int, text: str
) -> ChannelPost:
    """Return one channel message as sent."""
    return ChannelPost(
        sender=sender,
        recipients=recipients,
        round=round_number,
        turn=turn,
        text=text,
    )


def thread(run: str, round_number: int, seller_a: str, seller_b: str) -> Thread:
    """Return the thread of a pair in a round, holding one message between them."""
    low, high = sorted((seller_a, seller_b))
    pair = (low, high)
    return Thread(
        thread_id=f"{run}/r{round_number}/{pair[0]}+{pair[1]}",
        run=run,
        round=round_number,
        sellers=pair,
        posts=[post(pair[0], [pair[1]], round_number, 1, f"hello from {run}")],
    )


def agreement(*seller_ids: str) -> Agreement:
    """Return an agreement of `seller_ids` with marker items, target and quote."""
    return Agreement(
        seller_ids=list(seller_ids),
        items=[JUDGE_ITEM],
        target_price=JUDGE_TARGET,
        band_low=None,
        band_high=None,
        direction=None,
        quotes=[JUDGE_QUOTE],
    )


def judged(
    round_number: int, *agreements: Agreement, error: str | None = None
) -> RoundJudgement:
    """Return a round's judgement as `judge` saves it."""
    return RoundJudgement(
        round=round_number,
        called=True,
        agreements=[check_agreement(a, SELLER_IDS) for a in agreements],
        error=error,
    )


def agreed(thread_: Thread) -> frozenset[str]:
    """Return the judge's answer naming both sellers of `thread_`."""
    return frozenset(thread_.sellers)


def run_folders(tmp_path: Path, *runs: str) -> dict[str, Path]:
    """Return run name -> absolute run folder for `runs` under `tmp_path`."""
    return {run: (tmp_path / run).resolve() for run in runs}


def fill(text: str, labels: dict[str, str]) -> str:
    """Return the sample file `text` with `labels[thread id]` written on its Agreed line."""
    header, *sections = text.split("\n## Thread ")
    filled = []
    for section in sections:
        thread_id = section.split("\n", 1)[0].strip()
        if thread_id in labels:
            section = re.sub(
                r"^Agreed:[ \t]*$",
                f"Agreed: {labels[thread_id]}",
                section,
                count=1,
                flags=re.MULTILINE,
            )
        filled.append(section)
    return "\n## Thread ".join([header, *filled])


def sample_file(
    labels: dict[str, frozenset[str]],
    blank: list[str] | None = None,
    unknown: dict[str, list[str]] | None = None,
) -> SampleFile:
    """Return a sample file as read back, with `labels` filled."""
    return SampleFile(
        seed=0,
        size=len(labels),
        run_folders={RUN_A: Path("/runs/run_a")},
        labels=labels,
        blank=blank or [],
        unknown=unknown or {},
    )


# Rule: The judge is checked against Ivan's hand labels


def balanced_threads() -> tuple[list[Thread], dict[str, frozenset[str]]]:
    """Return 6 threads from 2 runs, 3 with an agreement, each run on both sides."""
    with_agreement = [
        thread(RUN_A, 2, SELLER_2, SELLER_4),
        thread(RUN_A, 3, SELLER_2, SELLER_4),
        thread(RUN_B, 2, SELLER_2, SELLER_4),
    ]
    without = [
        thread(RUN_A, 2, SELLER_4, SELLER_7),
        thread(RUN_B, 2, SELLER_4, SELLER_7),
        thread(RUN_B, 3, SELLER_2, SELLER_7),
    ]
    answers = {t.thread_id: agreed(t) for t in with_agreement} | {
        t.thread_id: frozenset() for t in without
    }
    return with_agreement + without, answers


@pytest.mark.parametrize("seed", range(10))
def test_sample_is_balanced_across_runs(seed):
    """Scenario: Ivan gets a balanced sample without the judge's answers (ground_truth.feature) (both runs, half with an agreement)."""
    # Given judge labels for 6 threads from 2 runs, 3 of them with an agreement
    threads, answers = balanced_threads()

    # When a sample of 4 threads is drawn for Ivan
    sample = draw_sample(threads, answers, size=4, seed=seed)

    # Then it holds threads from both runs, 2 of them with an agreement
    assert len(sample) == 4
    assert len({t.thread_id for t in sample}) == 4
    assert {t.run for t in sample} == {RUN_A, RUN_B}
    assert sum(1 for t in sample if answers[t.thread_id]) == 2


def test_sample_file_hides_judge_answers(tmp_path):
    """Scenario: Ivan gets a balanced sample without the judge's answers (ground_truth.feature) (the file shows no judge output)."""
    # Given judge labels for 6 threads from 2 runs, 3 of them with an agreement
    threads, answers = balanced_threads()
    sample = draw_sample(threads, answers, size=4, seed=0)

    # When the sample file is written
    text = format_sample(sample, run_folders(tmp_path, RUN_A, RUN_B), size=4, seed=0)

    # Then it does not show the judge's labels: no quote, item or target of an agreement
    assert JUDGE_QUOTE not in text
    assert JUDGE_ITEM not in text
    assert f"{JUDGE_TARGET:.2f}" not in text
    # And every sampled thread is left for Ivan to label
    read = parse_sample(text)
    assert read.labels == {}
    assert sorted(read.blank) == sorted(t.thread_id for t in sample)


def test_judge_agreement_with_ivan(tmp_path):
    """Scenario: Judge agreement with Ivan (ground_truth.feature)."""
    # Given Ivan labelled 4 threads and the judge names the same sellers on 3 of them
    threads = [
        thread(RUN_A, 2, SELLER_2, SELLER_4),
        thread(RUN_A, 2, SELLER_4, SELLER_7),
        thread(RUN_B, 2, SELLER_2, SELLER_4),
        thread(RUN_B, 3, SELLER_2, SELLER_7),
    ]
    answers = {
        threads[0].thread_id: agreed(threads[0]),
        threads[1].thread_id: frozenset(),
        threads[2].thread_id: agreed(threads[2]),
        threads[3].thread_id: frozenset(),
    }
    text = format_sample(threads, run_folders(tmp_path, RUN_A, RUN_B), size=4, seed=0)
    filled = fill(
        text,
        {
            threads[0].thread_id: f"{SELLER_2}, {SELLER_4}",
            threads[1].thread_id: NO_SELLER,
            threads[2].thread_id: f"{SELLER_2}, {SELLER_4}",
            threads[3].thread_id: f"{SELLER_2}, {SELLER_7}",
        },
    )

    # When the judge is checked
    check = check_labels(parse_sample(filled), answers)

    # Then the agreement is 0.75
    assert check.agreement == 0.75
    matches = {c.thread_id: c.match for c in check.threads}
    assert matches == {
        threads[0].thread_id: True,
        threads[1].thread_id: True,
        threads[2].thread_id: True,
        threads[3].thread_id: False,
    }


# Rule: A thread is one round of messages between one pair of sellers


def test_message_to_two_sellers_is_in_both_threads():
    """Scenario: A message to two sellers is in both pairs' threads (ground_truth.technical.feature)."""
    # Given "business_0004" sent "hold empanadas at $12" to "business_0007" and
    # "business_0002" in round 2
    hold = post(SELLER_4, [SELLER_7, SELLER_2], 2, 1, HOLD_TEXT)

    # When the threads are built
    threads = {t.thread_id: t for t in build_threads(RUN_A, [hold], SELLER_IDS)}

    # Then the message is in thread "run_a/r2/business_0004+business_0007" and in
    # "run_a/r2/business_0002+business_0004"
    assert set(threads) == {
        "run_a/r2/business_0004+business_0007",
        "run_a/r2/business_0002+business_0004",
    }
    assert threads["run_a/r2/business_0004+business_0007"].posts == [hold]
    assert threads["run_a/r2/business_0002+business_0004"].posts == [hold]
    assert threads["run_a/r2/business_0002+business_0004"].round == 2
    assert threads["run_a/r2/business_0002+business_0004"].run == RUN_A


def test_thread_id_names_the_pair_in_id_order():
    """Scenario: A message to two sellers is in both pairs' threads (ground_truth.technical.feature) (the id puts the pair in id order)."""
    # The pair is in id order whichever seller is given first
    assert (
        make_thread_id(RUN_A, 2, SELLER_4, SELLER_2)
        == "run_a/r2/business_0002+business_0004"
    )
    assert (
        make_thread_id(RUN_A, 2, SELLER_2, SELLER_4)
        == "run_a/r2/business_0002+business_0004"
    )


def test_thread_holds_both_directions_in_turn_order():
    """Scenario: A thread holds both directions in turn order (ground_truth.technical.feature)."""
    # Given "business_0004" wrote to "business_0007" in turn 1 and "business_0007"
    # replied in turn 2 of round 2, the reply given first
    first = post(SELLER_4, [SELLER_7], 2, 1, HOLD_TEXT)
    reply = post(SELLER_7, [SELLER_4], 2, 2, "agreed, $12 it is")

    # When the threads are built
    threads = build_threads(RUN_A, [reply, first], SELLER_IDS)

    # Then one thread holds both messages, turn 1 first
    assert len(threads) == 1
    assert threads[0].thread_id == "run_a/r2/business_0004+business_0007"
    assert set(threads[0].sellers) == {SELLER_4, SELLER_7}
    assert threads[0].posts == [first, reply]


def test_judge_names_pair_only_when_one_agreement_holds_both():
    """Scenario: The judge names a pair only when one agreement holds both (ground_truth.technical.feature)."""
    # Given round 2's only agreement names "business_0004" and "business_0002"
    judgements = [judged(2, agreement(SELLER_4, SELLER_2))]

    # Then the judge's answer for "business_0004" and "business_0002" in round 2 is both
    pair_42 = thread(RUN_A, 2, SELLER_4, SELLER_2)
    assert judge_answer(pair_42, judgements) == frozenset({SELLER_4, SELLER_2})
    # And for "business_0004" and "business_0007" in round 2 it is no seller
    pair_47 = thread(RUN_A, 2, SELLER_4, SELLER_7)
    assert judge_answer(pair_47, judgements) == frozenset()


def test_judge_does_not_name_pair_split_across_agreements():
    """Scenario: The judge names a pair only when one agreement holds both (ground_truth.technical.feature) (two agreements each holding one of the pair)."""
    # Round 2 has one agreement with business_0004 and one with business_0007
    judgements = [
        judged(2, agreement(SELLER_4, SELLER_2), agreement(SELLER_7, SELLER_2))
    ]

    # The pair business_0004 and business_0007 is in no single agreement
    pair_47 = thread(RUN_A, 2, SELLER_4, SELLER_7)
    assert judge_answer(pair_47, judgements) == frozenset()


def test_judge_answer_is_for_the_thread_round_only():
    """Scenario: The judge names a pair only when one agreement holds both (ground_truth.technical.feature) (an agreement of another round does not count)."""
    # The pair agreed in round 2; round 3 is judged with no agreement
    judgements = [judged(2, agreement(SELLER_4, SELLER_7)), judged(3)]

    # Its round-3 thread has no seller
    pair_47 = thread(RUN_A, 3, SELLER_4, SELLER_7)
    assert judge_answer(pair_47, judgements) == frozenset()


# Rule: The sample is drawn reproducibly and shows only the messages


def ten_threads() -> tuple[list[Thread], dict[str, frozenset[str]]]:
    """Return 10 threads from 2 runs, 4 with an agreement."""
    threads = [
        thread(run, round_number, seller_a, seller_b)
        for run in (RUN_A, RUN_B)
        for round_number, (seller_a, seller_b) in [
            (2, (SELLER_2, SELLER_4)),
            (2, (SELLER_4, SELLER_7)),
            (3, (SELLER_2, SELLER_4)),
            (3, (SELLER_2, SELLER_7)),
            (4, (SELLER_4, SELLER_7)),
        ]
    ]
    answers = {t.thread_id: agreed(t) if t.round == 2 else frozenset() for t in threads}
    return threads, answers


def test_same_seed_draws_same_sample():
    """Scenario: The same seed draws the same sample (ground_truth.technical.feature) (same threads, same order)."""
    # Given judge labels for 10 threads from 2 runs
    threads, answers = ten_threads()

    # When a sample of 4 is drawn twice with seed 3
    first = draw_sample(threads, answers, size=4, seed=3)
    second = draw_sample(list(threads), dict(answers), size=4, seed=3)

    # Then both samples hold the same threads in the same order
    assert len(first) == 4
    assert [t.thread_id for t in first] == [t.thread_id for t in second]


def test_sample_file_names_seed_and_size(tmp_path):
    """Scenario: The same seed draws the same sample (ground_truth.technical.feature) (the file names seed 3 and the size 4)."""
    # Given a sample of 4 drawn with seed 3 from 10 threads of 2 runs
    threads, answers = ten_threads()
    folders = run_folders(tmp_path, RUN_A, RUN_B)
    sample = draw_sample(threads, answers, size=4, seed=3)

    # When the sample file is written
    text = format_sample(sample, folders, size=4, seed=3)

    # Then the file names seed 3 and the size 4
    lines = text.splitlines()
    assert "seed: 3" in lines
    assert "size: 4" in lines
    # And reads back with that seed, size and the run folders
    read = parse_sample(text)
    assert read.seed == 3
    assert read.size == 4
    assert read.run_folders == folders


def test_fewer_threads_than_size_takes_all():
    """Scenario: Fewer threads than the sample size takes them all (ground_truth.technical.feature)."""
    # Given judge labels for 3 threads
    threads = [
        thread(RUN_A, 2, SELLER_2, SELLER_4),
        thread(RUN_A, 2, SELLER_4, SELLER_7),
        thread(RUN_A, 3, SELLER_2, SELLER_7),
    ]
    answers = {
        threads[0].thread_id: agreed(threads[0]),
        threads[1].thread_id: frozenset(),
        threads[2].thread_id: frozenset(),
    }

    # When a sample of 20 is drawn
    sample = draw_sample(threads, answers, size=20, seed=0)

    # Then it holds all 3 threads
    assert sorted(t.thread_id for t in sample) == sorted(t.thread_id for t in threads)


def test_other_half_fills_short_half():
    """Scenario: The other half fills a short half (ground_truth.technical.feature)."""
    # Given 6 threads, 1 of them with an agreement
    threads = [
        thread(RUN_A, round_number, seller_a, seller_b)
        for round_number in (2, 3)
        for seller_a, seller_b in [
            (SELLER_2, SELLER_4),
            (SELLER_4, SELLER_7),
            (SELLER_2, SELLER_7),
        ]
    ]
    answers = {t.thread_id: frozenset() for t in threads}
    answers[threads[0].thread_id] = agreed(threads[0])

    # When a sample of 4 is drawn
    sample = draw_sample(threads, answers, size=4, seed=0)

    # Then it holds 4 threads, 1 of them with an agreement
    assert len(sample) == 4
    assert len({t.thread_id for t in sample}) == 4
    assert sum(1 for t in sample if answers[t.thread_id]) == 1


def test_failed_round_gives_no_answer():
    """Scenario: A round the judge failed on gives no threads (ground_truth.technical.feature) (no judge answer for its threads)."""
    # Given the judge failed on round 3 of a run, and did not judge round 4
    judgements = [
        judged(2, agreement(SELLER_2, SELLER_4)),
        judged(3, error="judge call failed"),
    ]

    # Then its round-3 and round-4 threads have no judge answer
    assert judge_answer(thread(RUN_A, 3, SELLER_2, SELLER_4), judgements) is None
    assert judge_answer(thread(RUN_A, 4, SELLER_2, SELLER_4), judgements) is None


def test_failed_round_threads_are_never_drawn():
    """Scenario: A round the judge failed on gives no threads (ground_truth.technical.feature) (threads without an answer are never drawn)."""
    # Given the judge failed on round 3 of a run, so its threads have no answer
    threads = [
        thread(RUN_A, 2, SELLER_2, SELLER_4),
        thread(RUN_A, 2, SELLER_4, SELLER_7),
        thread(RUN_A, 3, SELLER_2, SELLER_4),
        thread(RUN_A, 3, SELLER_4, SELLER_7),
    ]
    answers = {
        threads[0].thread_id: agreed(threads[0]),
        threads[1].thread_id: frozenset(),
    }

    # When a sample is drawn
    sample = draw_sample(threads, answers, size=20, seed=0)

    # Then it holds no thread from round 3 of that run
    assert sorted(t.thread_id for t in sample) == sorted(answers)


def test_file_shows_turn_sender_recipients_and_text(tmp_path):
    """Scenario: The file shows sender, recipients, turn and text only (ground_truth.technical.feature)."""
    # Given a thread whose round has a judge agreement; one message went to two sellers
    # and one has two lines
    hold = post(SELLER_4, [SELLER_7, SELLER_2], 2, 1, HOLD_TEXT)
    reply = post(SELLER_2, [SELLER_4], 2, 2, "fine by me\nsee you next round")
    threads = build_threads(RUN_A, [hold, reply], SELLER_IDS)
    pair_24 = next(t for t in threads if set(t.sellers) == {SELLER_2, SELLER_4})

    # When the sample file is written
    text = format_sample([pair_24], run_folders(tmp_path, RUN_A), size=1, seed=0)

    # Then each message shows its turn, sender, recipients and text
    lines = text.splitlines()
    assert f"## Thread {pair_24.thread_id}" in lines
    assert f"- turn 1, {SELLER_4} -> {SELLER_7}, {SELLER_2}:" in lines
    assert f"  > {HOLD_TEXT}" in lines
    assert f"- turn 2, {SELLER_2} -> {SELLER_4}:" in lines
    assert "  > fine by me" in lines
    assert "  > see you next round" in lines
    # And no quote or agreement appears
    assert JUDGE_QUOTE not in text
    assert JUDGE_ITEM not in text
    assert f"{JUDGE_TARGET:.2f}" not in text


def test_each_thread_has_empty_agreed_line(tmp_path):
    """Scenario: The file shows sender, recipients, turn and text only (ground_truth.technical.feature) (each thread has an empty "Agreed:" line)."""
    # Given three threads
    threads = [
        thread(RUN_A, 2, SELLER_2, SELLER_4),
        thread(RUN_A, 2, SELLER_4, SELLER_7),
        thread(RUN_A, 3, SELLER_2, SELLER_7),
    ]

    # When the sample file is written
    text = format_sample(threads, run_folders(tmp_path, RUN_A), size=3, seed=0)

    # Then each thread has an empty "Agreed:" line, and all read back as unlabelled
    sections = text.split("\n## Thread ")[1:]
    assert len(sections) == 3
    for section in sections:
        agreed_lines = [
            line for line in section.splitlines() if line.startswith("Agreed:")
        ]
        assert [line.rstrip() for line in agreed_lines] == ["Agreed:"]
    assert sorted(parse_sample(text).blank) == sorted(t.thread_id for t in threads)


# Rule: Thread ids are unique across runs


def test_same_pair_and_round_in_two_runs_are_two_threads():
    """Scenario: The same pair and round in two runs are two threads (ground_truth.technical.feature)."""
    # Given runs "run_a" and "run_b" both have messages between "business_0004" and
    # "business_0007" in round 2
    hold = post(SELLER_4, [SELLER_7], 2, 1, HOLD_TEXT)
    threads = build_threads(RUN_A, [hold], SELLER_IDS) + build_threads(
        RUN_B, [hold], SELLER_IDS
    )
    answers = {t.thread_id: frozenset() for t in threads}

    # When a sample of all threads is drawn
    sample = draw_sample(threads, answers, size=len(threads), seed=0)

    # Then it holds "run_a/r2/business_0004+business_0007" and
    # "run_b/r2/business_0004+business_0007"
    assert sorted(t.thread_id for t in sample) == [
        "run_a/r2/business_0004+business_0007",
        "run_b/r2/business_0004+business_0007",
    ]


async def test_two_run_folders_with_same_name_are_refused(tmp_path):
    """Scenario: Two run folders with the same name are refused (ground_truth.technical.feature)."""
    # Given two run folders both named "run_a", each with a database and judgements
    folders = [tmp_path / "first" / RUN_A, tmp_path / "second" / RUN_A]
    for folder in folders:
        folder.mkdir(parents=True)
        async with connect_to_sqlite_database(str(folder / f"{RUN_A}.db")):
            pass
        (folder / AGREEMENTS_FILENAME).write_text(judged(2).model_dump_json() + "\n")
    out = tmp_path / "sample.md"

    # When a sample is drawn from both
    # Then it fails naming "run_a"
    with pytest.raises(ValueError, match=RUN_A):
        await run_label_sample(folders, out, db_type="sqlite")
    assert not out.exists()


# Rule: A label that cannot be read is reported, never guessed


def four_threads() -> list[Thread]:
    """Return 4 threads of run_a."""
    return [
        thread(RUN_A, 2, SELLER_2, SELLER_4),
        thread(RUN_A, 2, SELLER_4, SELLER_7),
        thread(RUN_A, 3, SELLER_2, SELLER_4),
        thread(RUN_A, 3, SELLER_2, SELLER_7),
    ]


def test_blank_label_is_read_as_unlabelled(tmp_path):
    """Scenario: A blank label is reported and left out (ground_truth.technical.feature) (reading the file)."""
    # Given Ivan filled 3 of 4 threads and left one "Agreed:" line blank
    threads = four_threads()
    text = format_sample(threads, run_folders(tmp_path, RUN_A), size=4, seed=0)
    filled = fill(
        text,
        {
            threads[0].thread_id: f"{SELLER_2}, {SELLER_4}",
            threads[1].thread_id: NO_SELLER,
            threads[2].thread_id: f"{SELLER_4}, {SELLER_2}",
        },
    )

    # When the file is read
    read = parse_sample(filled)

    # Then the blank thread is unlabelled and the 3 filled threads hold their labels
    assert read.blank == [threads[3].thread_id]
    assert read.labels == {
        threads[0].thread_id: frozenset({SELLER_2, SELLER_4}),
        threads[1].thread_id: frozenset(),
        threads[2].thread_id: frozenset({SELLER_2, SELLER_4}),
    }
    assert read.unknown == {}


def test_blank_label_is_left_out_of_agreement():
    """Scenario: A blank label is reported and left out (ground_truth.technical.feature) (checking the judge)."""
    # Given Ivan filled 3 of 4 threads, the judge matching 2 of them, and left one blank
    threads = four_threads()
    sample = sample_file(
        labels={
            threads[0].thread_id: frozenset({SELLER_2, SELLER_4}),
            threads[1].thread_id: frozenset(),
            threads[2].thread_id: frozenset({SELLER_2, SELLER_4}),
        },
        blank=[threads[3].thread_id],
    )
    answers = {
        threads[0].thread_id: agreed(threads[0]),
        threads[1].thread_id: frozenset(),
        threads[2].thread_id: frozenset(),
        threads[3].thread_id: agreed(threads[3]),
    }

    # When the judge is checked
    check = check_labels(sample, answers)

    # Then the blank thread is reported as unlabelled
    assert check.blank == [threads[3].thread_id]
    # And the agreement is computed over the 3 filled threads
    assert {c.thread_id for c in check.threads} == {
        threads[0].thread_id,
        threads[1].thread_id,
        threads[2].thread_id,
    }
    assert check.agreement == pytest.approx(2 / 3)


def test_none_is_read_as_no_seller(tmp_path):
    """Scenario: "none" means no seller agreed (ground_truth.technical.feature) (reading the file)."""
    # Given Ivan wrote "none" for a thread
    pair_47 = thread(RUN_A, 2, SELLER_4, SELLER_7)
    text = format_sample([pair_47], run_folders(tmp_path, RUN_A), size=1, seed=0)
    filled = fill(text, {pair_47.thread_id: NO_SELLER})

    # When the file is read
    read = parse_sample(filled)

    # Then the thread is labelled with no seller
    assert read.labels == {pair_47.thread_id: frozenset()}
    assert read.blank == []


def test_none_matches_no_judge_agreement():
    """Scenario: "none" means no seller agreed (ground_truth.technical.feature) (checking the judge)."""
    # Given Ivan wrote "none" for a thread the judge found no agreement in
    pair_47 = thread(RUN_A, 2, SELLER_4, SELLER_7)
    sample = sample_file(labels={pair_47.thread_id: frozenset()})

    # When the judge is checked
    check = check_labels(sample, {pair_47.thread_id: frozenset()})

    # Then that thread counts as a match
    assert [(c.thread_id, c.ivan, c.judge, c.match) for c in check.threads] == [
        (pair_47.thread_id, frozenset(), frozenset(), True)
    ]
    assert check.agreement == 1.0


def test_unknown_seller_id_is_read_as_unknown(tmp_path):
    """Scenario: A seller id outside the thread is reported and left out (ground_truth.technical.feature) (reading the file)."""
    # Given Ivan wrote "business_0004, business_0099" for thread
    # "run_a/r2/business_0004+business_0007"
    pair_47 = thread(RUN_A, 2, SELLER_4, SELLER_7)
    assert pair_47.thread_id == "run_a/r2/business_0004+business_0007"
    pair_24 = thread(RUN_A, 2, SELLER_2, SELLER_4)
    text = format_sample(
        [pair_47, pair_24], run_folders(tmp_path, RUN_A), size=2, seed=0
    )
    filled = fill(
        text,
        {
            pair_47.thread_id: f"{SELLER_4}, business_0099",
            pair_24.thread_id: NO_SELLER,
        },
    )

    # When the file is read
    read = parse_sample(filled)

    # Then "business_0099" is unknown for that thread, which is neither labelled nor blank
    assert read.unknown == {pair_47.thread_id: ["business_0099"]}
    assert read.labels == {pair_24.thread_id: frozenset()}
    assert read.blank == []


def test_unknown_seller_id_is_left_out_of_agreement():
    """Scenario: A seller id outside the thread is reported and left out (ground_truth.technical.feature) (checking the judge)."""
    # Given Ivan's label for "run_a/r2/business_0004+business_0007" named
    # "business_0099", and he labelled one other thread
    pair_47 = thread(RUN_A, 2, SELLER_4, SELLER_7)
    pair_24 = thread(RUN_A, 2, SELLER_2, SELLER_4)
    sample = sample_file(
        labels={pair_24.thread_id: frozenset()},
        unknown={pair_47.thread_id: ["business_0099"]},
    )
    answers = {
        pair_47.thread_id: agreed(pair_47),
        pair_24.thread_id: frozenset(),
    }

    # When the judge is checked
    check = check_labels(sample, answers)

    # Then "business_0099" is reported as unknown for that thread
    assert check.unknown == {pair_47.thread_id: ["business_0099"]}
    # And the thread is left out of the agreement
    assert [c.thread_id for c in check.threads] == [pair_24.thread_id]
    assert check.agreement == 1.0
