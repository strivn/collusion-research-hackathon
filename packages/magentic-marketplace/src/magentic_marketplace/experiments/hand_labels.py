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

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from magentic_marketplace.experiments.judge import ChannelPost, RoundJudgement

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
    raise NotImplementedError


def build_threads(
    run: str, posts: list[ChannelPost], seller_ids: set[str]
) -> list[Thread]:
    """Return every thread of one run, sorted by round, then pair.

    A post is in the thread of its sender and each recipient that is a seller in
    `seller_ids`. Posts within a thread are in turn order, ties in the order given.
    """
    raise NotImplementedError


def judge_answer(
    thread: Thread, judgements: list[RoundJudgement]
) -> frozenset[str] | None:
    """Return the judge's answer for `thread`: both sellers, or an empty set.

    Both when one agreement of the thread's round names both sellers of the pair. None when
    the round has no judgement or its judgement has an error.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def format_sample(
    sample: list[Thread], run_folders: dict[str, Path], size: int, seed: int
) -> str:
    """Return the sample file's text (layout in the module docstring).

    Shows each message's turn, sender, recipients and text, and nothing of the judge.
    """
    raise NotImplementedError


def parse_sample(text: str) -> SampleFile:
    """Read a sample file's header and Ivan's "Agreed:" lines."""
    raise NotImplementedError


def check_labels(sample: SampleFile, answers: dict[str, frozenset[str]]) -> LabelCheck:
    """Compare Ivan's labels with the judge's answers (thread id -> answer).

    Raises ValueError naming any labelled thread that has no judge answer.
    """
    raise NotImplementedError


def save_check(check: LabelCheck, path: Path) -> Path:
    """Write one row per labelled thread to `path`; return it.

    CSV columns thread_id, ivan, judge, match. Sellers are space-separated in id order,
    empty for none; match is "true" or "false".
    """
    raise NotImplementedError


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
    raise NotImplementedError


def run_label_check(sample_path: Path) -> tuple[LabelCheck, Path]:
    """Check a filled sample file against the judge answers in the run folders it names.

    Writes "<sample stem>" + CHECK_SUFFIX next to it. Returns the check and that path.
    """
    raise NotImplementedError
