"""Checking the judge against Ivan's hand labels.

Two steps, two commands. `label-sample` draws a balanced sample of channel threads from one
or more judged run folders into one markdown file, without the judge's answers.
`label-check` reads the filled file back and reports how often the judge names the same
sellers as Ivan.

A thread is every channel message of one round between one pair of sellers, in turn order.
A message to several sellers belongs to each pair's thread. Its id is
"<run>/r<round>/<seller>+<seller>", the pair in id order, where <run> is the run folder's
name. The judge's answer for a thread is both sellers of the pair when one agreement of that
round names both, and no seller otherwise. A round the judge failed on, or did not judge,
gives no answer, so its threads are never drawn.

Sample file layout (`format_sample` writes it, `parse_sample` reads it):

    # Hand-label sample

    seed: <seed>
    size: <requested size>
    runs:
    - <run name>: <absolute run folder path>

    <one paragraph telling Ivan how to fill the Agreed lines>

    ## Thread <thread id>

    - turn <t>, <sender> -> <recipient>, <recipient>:
      > <text line>
      > <text line>

    Agreed:

Threads are separated by blank lines. On an "Agreed:" line Ivan writes the ids of the
thread's sellers who agreed on prices, comma-separated, or "none". A blank line means not
labelled.
"""

import csv
import random
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from magentic_marketplace.experiments.acted import load_judgements
from magentic_marketplace.experiments.judge import (
    AGREEMENTS_FILENAME,
    ChannelPost,
    RoundJudgement,
    load_channel_posts,
)
from magentic_marketplace.experiments.questioning import load_seller_ids
from magentic_marketplace.platform.database import connect_to_postgresql_database
from magentic_marketplace.platform.database.base import BaseDatabaseController
from magentic_marketplace.platform.database.sqlite.sqlite import (
    SQLiteDatabaseController,
)

SAMPLE_SIZE = 20
SAMPLE_SEED = 0
NO_SELLER = "none"
CHECK_SUFFIX = ".check.csv"


class Thread(BaseModel):
    """One round of channel messages between one pair of sellers."""

    thread_id: str
    run: str
    round: int
    sellers: tuple[str, str]
    posts: list[ChannelPost]


class SampleFile(BaseModel):
    """A sample file as read back.

    `labels` holds each thread Ivan filled, thread id -> sellers he named (empty for
    "none"). `blank` lists threads with an empty "Agreed:" line. `unknown` lists, per
    thread, the ids Ivan wrote that are not a seller of that thread; such a thread is in
    neither `labels` nor `blank`.
    """

    seed: int
    size: int
    run_folders: dict[str, Path]
    labels: dict[str, frozenset[str]]
    blank: list[str]
    unknown: dict[str, list[str]]


class ThreadCheck(BaseModel):
    """Ivan's and the judge's answer for one labelled thread."""

    thread_id: str
    ivan: frozenset[str]
    judge: frozenset[str]
    match: bool


class LabelCheck(BaseModel):
    """The judge checked against Ivan's labels.

    `agreement` is the share of labelled threads where both name the same sellers, or None
    when no thread is labelled. `blank` and `unknown` are carried over from the sample file.
    """

    agreement: float | None
    threads: list[ThreadCheck]
    blank: list[str]
    unknown: dict[str, list[str]]


def make_thread_id(run: str, round_number: int, seller_a: str, seller_b: str) -> str:
    """Return "<run>/r<round>/<seller>+<seller>" with the two sellers in id order."""
    first, second = sorted((seller_a, seller_b))
    return f"{run}/r{round_number}/{first}+{second}"


def _split_thread_id(thread_id: str) -> tuple[str, int, tuple[str, str]]:
    """Return the run, round and seller pair written in a thread id."""
    run, round_part, pair = thread_id.rsplit("/", 2)
    seller_a, seller_b = pair.split("+")
    return run, int(round_part.removeprefix("r")), (seller_a, seller_b)


def build_threads(
    run: str, posts: list[ChannelPost], seller_ids: set[str]
) -> list[Thread]:
    """Return every thread of one run, sorted by round, then pair.

    A post is in the thread of its sender and each recipient that is a seller in
    `seller_ids`. Posts within a thread are in turn order, ties in the order given.
    """
    # Group posts by thread id, only between two sellers
    grouped: dict[str, Thread] = {}
    for post in posts:
        if post.sender not in seller_ids:
            continue
        recipients = {
            r for r in post.recipients if r in seller_ids and r != post.sender
        }
        for recipient in sorted(recipients):
            thread_id = make_thread_id(run, post.round, post.sender, recipient)
            if thread_id not in grouped:
                grouped[thread_id] = Thread(
                    thread_id=thread_id,
                    run=run,
                    round=post.round,
                    sellers=_split_thread_id(thread_id)[2],
                    posts=[],
                )
            grouped[thread_id].posts.append(post)

    # Turn order inside a thread (sorted is stable), threads by round then pair
    threads = sorted(grouped.values(), key=lambda t: (t.round, t.sellers))
    for thread in threads:
        thread.posts.sort(key=lambda post: post.turn)
    return threads


def judge_answer(
    thread: Thread, judgements: list[RoundJudgement]
) -> frozenset[str] | None:
    """Return the judge's answer for `thread`: both sellers, or an empty set.

    Both when one agreement of the thread's round names both sellers of the pair. None when
    the round has no judgement or its judgement has an error.
    """
    # The round's judgement decides; a failed or missing round has no answer
    judgement = next((j for j in judgements if j.round == thread.round), None)
    if judgement is None or judgement.error:
        return None
    pair = set(thread.sellers)
    if any(pair <= set(a.seller_ids) for a in judgement.agreements):
        return frozenset(pair)
    return frozenset()


def _interleave_by_run(threads: list[Thread], rng: random.Random) -> list[Thread]:
    """Return `threads` shuffled within each run, then taken one run at a time in turn."""
    by_run: dict[str, list[Thread]] = defaultdict(list)
    for thread in threads:
        by_run[thread.run].append(thread)
    lists = []
    for run in sorted(by_run):
        rng.shuffle(by_run[run])
        lists.append(by_run[run])
    return [t for row in zip_longest(*lists) for t in row if t is not None]


def draw_sample(
    threads: list[Thread],
    answers: dict[str, frozenset[str]],
    size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> list[Thread]:
    """Return a balanced sample of up to `size` threads, the same for the same seed.

    Threads without an entry in `answers` (thread id -> judge's answer) are never drawn.
    size // 2 threads come from those with an agreement (non-empty answer), the rest from
    those without; when one side is short, the other fills up. Within each side, threads
    are spread across runs in turn. With fewer threads than `size`, all are drawn. The
    sample is shuffled, so its order says nothing about the answers.
    """
    # Split by the judge's answer, spread each side across runs
    rng = random.Random(seed)
    answered = [t for t in threads if t.thread_id in answers]
    with_agreement = _interleave_by_run(
        [t for t in answered if answers[t.thread_id]], rng
    )
    without = _interleave_by_run([t for t in answered if not answers[t.thread_id]], rng)

    # Half from each side, the other side tops up a short one
    want = size // 2
    picked_with = with_agreement[:want]
    picked_without = without[: size - want]
    top_up = size - len(picked_with) - len(picked_without)
    if top_up > 0:
        picked_with += with_agreement[len(picked_with) :][:top_up]
        top_up = size - len(picked_with) - len(picked_without)
        picked_without += without[len(picked_without) :][:top_up]

    sample = picked_with + picked_without
    rng.shuffle(sample)
    return sample


INSTRUCTIONS = (
    'For each thread, write after "Agreed:" the ids of the thread\'s sellers who agreed '
    'on prices with the other, comma-separated, or "none". Leave it blank to skip the '
    "thread."
)


def format_sample(
    sample: list[Thread], run_folders: dict[str, Path], size: int, seed: int
) -> str:
    """Return the sample file's text (layout in the module docstring).

    Shows each message's turn, sender, recipients and text, and nothing of the judge.
    """
    # Header: seed, size and where each run lives
    lines = ["# Hand-label sample", "", f"seed: {seed}", f"size: {size}", "runs:"]
    lines += [f"- {name}: {folder.resolve()}" for name, folder in run_folders.items()]
    lines += ["", INSTRUCTIONS]

    # One block per thread; every text line is quoted so it cannot pass for structure
    for thread in sample:
        lines += ["", f"## Thread {thread.thread_id}"]
        for post in thread.posts:
            lines += [
                "",
                f"- turn {post.turn}, {post.sender} -> {', '.join(post.recipients)}:",
            ]
            lines += [
                f"  > {text_line}".rstrip() for text_line in post.text.split("\n")
            ]
        lines += ["", "Agreed:"]
    return "\n".join(lines) + "\n"


def parse_sample(text: str) -> SampleFile:
    """Read a sample file's header and Ivan's "Agreed:" lines."""
    seed = 0
    size = 0
    run_folders: dict[str, Path] = {}
    labels: dict[str, frozenset[str]] = {}
    blank: list[str] = []
    unknown: dict[str, list[str]] = {}
    thread_id: str | None = None
    in_runs = False

    for line in text.splitlines():
        if line.startswith("## Thread "):
            thread_id = line.removeprefix("## Thread ").strip()
            in_runs = False
        elif thread_id is None:
            # Header, before the first thread
            if line.startswith("seed:"):
                seed = int(line.removeprefix("seed:").strip())
            elif line.startswith("size:"):
                size = int(line.removeprefix("size:").strip())
            elif line.startswith("runs:"):
                in_runs = True
            elif in_runs and line.startswith("- "):
                name, _, folder = line.removeprefix("- ").partition(": ")
                run_folders[name.strip()] = Path(folder.strip())
        elif line.startswith("Agreed:"):
            value = line.removeprefix("Agreed:").strip()
            if not value:
                blank.append(thread_id)
            elif value.lower() == NO_SELLER:
                labels[thread_id] = frozenset()
            else:
                named = [s.strip() for s in value.split(",") if s.strip()]
                pair = _split_thread_id(thread_id)[2]
                outside = [s for s in named if s not in pair]
                if outside:
                    unknown[thread_id] = outside
                else:
                    labels[thread_id] = frozenset(named)
            thread_id = None

    return SampleFile(
        seed=seed,
        size=size,
        run_folders=run_folders,
        labels=labels,
        blank=blank,
        unknown=unknown,
    )


def check_labels(sample: SampleFile, answers: dict[str, frozenset[str]]) -> LabelCheck:
    """Compare Ivan's labels with the judge's answers (thread id -> answer).

    Raises ValueError naming any labelled thread that has no judge answer.
    """
    # Each labelled thread needs a judge answer to compare with
    checks = []
    for thread_id, ivan in sample.labels.items():
        if thread_id not in answers:
            raise ValueError(f"No judge answer for labelled thread {thread_id}")
        judge = answers[thread_id]
        checks.append(
            ThreadCheck(
                thread_id=thread_id, ivan=ivan, judge=judge, match=ivan == judge
            )
        )
    agreement = sum(c.match for c in checks) / len(checks) if checks else None
    return LabelCheck(
        agreement=agreement,
        threads=checks,
        blank=sample.blank,
        unknown=sample.unknown,
    )


def save_check(check: LabelCheck, path: Path) -> Path:
    """Write one row per labelled thread to `path`; return it.

    CSV columns thread_id, ivan, judge, match. Sellers are space-separated in id order,
    empty for none; match is "true" or "false".
    """
    # One row per labelled thread
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["thread_id", "ivan", "judge", "match"])
        for row in check.threads:
            writer.writerow(
                [
                    row.thread_id,
                    " ".join(sorted(row.ivan)),
                    " ".join(sorted(row.judge)),
                    "true" if row.match else "false",
                ]
            )
    return path


async def _threads_of_run(database: BaseDatabaseController, run: str) -> list[Thread]:
    """Return the threads of one run's database."""
    seller_ids = await load_seller_ids(database)
    return build_threads(run, await load_channel_posts(database), seller_ids)


async def run_label_sample(
    run_folders: list[Path],
    out: Path,
    size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
    db_type: Literal["postgres", "sqlite"] = "postgres",
    postgres_host: str = "localhost",
    postgres_port: int = 5432,
    postgres_password: str = "postgres",
) -> Path:
    """Draw a sample from judged run folders and write it to `out`; return `out`.

    Each folder's name is its run name and Postgres schema; with sqlite its database is
    "<folder>/<folder name>.db". The judge's answers come from each folder's
    judge_agreements.jsonl. Raises ValueError naming a run name used by two folders.
    """
    # Refuse a repeated run name before touching any database
    folders: dict[str, Path] = {}
    for folder in run_folders:
        if folder.name in folders:
            raise ValueError(f"Run name {folder.name} is used by two run folders")
        folders[folder.name] = folder

    # Threads and the judge's answers per run
    threads: list[Thread] = []
    answers: dict[str, frozenset[str]] = {}
    for run, folder in folders.items():
        judgements = load_judgements(folder / AGREEMENTS_FILENAME)
        if db_type == "sqlite":
            db_path = folder / f"{run}.db"
            if not db_path.exists():
                raise FileNotFoundError(f"SQLite database file {db_path} not found")
            sqlite_database = SQLiteDatabaseController(str(db_path))
            await sqlite_database.initialize()
            run_threads = await _threads_of_run(sqlite_database, run)
        else:
            async with connect_to_postgresql_database(
                schema=run,
                host=postgres_host,
                port=postgres_port,
                password=postgres_password,
                mode="existing",
            ) as database:
                run_threads = await _threads_of_run(database, run)
        threads += run_threads
        for thread in run_threads:
            answer = judge_answer(thread, judgements)
            if answer is not None:
                answers[thread.thread_id] = answer

    # Draw and write the file
    sample = draw_sample(threads, answers, size=size, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_sample(sample, folders, size, seed))
    return out


def run_label_check(sample_path: Path) -> tuple[LabelCheck, Path]:
    """Check a filled sample file against the judge answers in the run folders it names.

    Writes "<sample stem>" + CHECK_SUFFIX next to it. Returns the check and that path.
    """
    sample = parse_sample(sample_path.read_text())

    # The judge's answer for each labelled thread needs only the run's judgement file
    judgements: dict[str, list[RoundJudgement]] = {}
    answers: dict[str, frozenset[str]] = {}
    for thread_id in sample.labels:
        run, round_number, pair = _split_thread_id(thread_id)
        if run not in sample.run_folders:
            raise ValueError(
                f"Thread {thread_id} names run {run}, not in the file header"
            )
        if run not in judgements:
            judgements[run] = load_judgements(
                sample.run_folders[run] / AGREEMENTS_FILENAME
            )
        thread = Thread(
            thread_id=thread_id, run=run, round=round_number, sellers=pair, posts=[]
        )
        answer = judge_answer(thread, judgements[run])
        if answer is not None:
            answers[thread_id] = answer

    check = check_labels(sample, answers)
    check_path = sample_path.with_name(sample_path.stem + CHECK_SUFFIX)
    return check, save_check(check, check_path)
