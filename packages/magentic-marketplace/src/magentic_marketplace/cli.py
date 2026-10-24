"""Command-line interface for magentic-marketplace."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from magentic_marketplace.experiments.acted import run_acted
from magentic_marketplace.experiments.conduct import run_conduct
from magentic_marketplace.experiments.experiment_runner import (
    load_experiment_config,
    run_experiment,
)
from magentic_marketplace.experiments.export_experiment import export_experiment
from magentic_marketplace.experiments.exposure import run_exposure
from magentic_marketplace.experiments.extract_agent_llm_traces import (
    run_extract_traces,
)
from magentic_marketplace.experiments.hand_labels import (
    SAMPLE_SEED,
    SAMPLE_SIZE,
    run_label_check,
    run_label_sample,
)
from magentic_marketplace.experiments.judge import JUDGE_MODEL, run_judge
from magentic_marketplace.experiments.list_experiments import list_experiments
from magentic_marketplace.experiments.questioning import run_question
from magentic_marketplace.experiments.round_scores import run_score
from magentic_marketplace.experiments.run_analytics import run_analytics
from magentic_marketplace.experiments.run_audit import run_audit
from magentic_marketplace.experiments.run_config import (
    RunConfig,
    load_run_config,
    save_run_config,
)
from magentic_marketplace.experiments.run_experiment import run_marketplace_experiment
from magentic_marketplace.experiments.screen import run_screen, run_screen_calibrate
from magentic_marketplace.experiments.utils import setup_logging
from magentic_marketplace.ui import run_ui_server

DEFAULT_POSTGRES_PORT = 5432
DEFAULT_UI_PORT = 5000


def run_experiment_command(args):
    """Handle the experiment subcommand."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Set logging level if provided
    if hasattr(args, "log_level"):
        numeric_level = getattr(logging, args.log_level.upper())
        logging.getLogger().setLevel(numeric_level)

    # Take the run settings from the config file, or from the flags with stock defaults
    if args.config is not None:
        conflicting_flags = {
            "data_dir": args.data_dir,
            "--experiment-name": args.experiment_name,
            "--search-algorithm": args.search_algorithm,
            "--search-bandwidth": args.search_bandwidth,
            "--customer-max-steps": args.customer_max_steps,
        }
        for flag, value in conflicting_flags.items():
            if value is not None:
                raise ValueError(f"--config cannot be combined with {flag}")
        config = load_run_config(Path(args.config))
        experiment_name = config.experiment_name
        search_algorithm = config.search_algorithm
        search_bandwidth = config.search_bandwidth
        customer_max_steps = config.customer_max_steps
        data_dir = Path(config.data)
    else:
        if args.data_dir is None:
            raise ValueError("data_dir is required unless --config is given")
        config = None
        # A flag left unset takes the stock default declared on RunConfig
        stock = {
            name: field.default
            for name, field in RunConfig.model_fields.items()
            if not field.is_required()
        }
        experiment_name = args.experiment_name
        search_algorithm = (
            stock["search_algorithm"]
            if args.search_algorithm is None
            else args.search_algorithm
        )
        search_bandwidth = (
            stock["search_bandwidth"]
            if args.search_bandwidth is None
            else args.search_bandwidth
        )
        customer_max_steps = (
            stock["customer_max_steps"]
            if args.customer_max_steps is None
            else args.customer_max_steps
        )
        data_dir = Path(args.data_dir)

    # Validate data directory structure
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)

    businesses_dir = data_dir / "businesses"
    customers_dir = data_dir / "customers"

    if not businesses_dir.exists():
        logger.error(f"Businesses directory does not exist: {businesses_dir}")
        sys.exit(1)

    if not customers_dir.exists():
        logger.error(f"Customers directory does not exist: {customers_dir}")
        sys.exit(1)

    # Try load .env
    env_file = getattr(args, "env_file", ".env")
    did_load_env = load_dotenv(env_file)
    if did_load_env:
        logger.info(f"Loaded environment variables from env file at path: {env_file}")
    else:
        logger.warning(
            f"No environment variables loaded from env file at path: {env_file}"
        )

    # Record the settings this run used in the run folder
    if config is not None:
        save_run_config(config, Path.cwd())

    logger.info(
        "Marketplace Experiment Runner\n"
        "This experiment will:\n"
        f"1. Load businesses from: {businesses_dir}\n"
        f"2. Load customers from: {customers_dir}\n"
        f"3. Create a Postgres database schema: {experiment_name}\n"
        "4. Start a marketplace server with simple marketplace protocol\n"
        "5. Register all business and customer agents\n"
        "6. Run the marketplace simulation\n"
    )

    # Run the experiment
    asyncio.run(
        run_marketplace_experiment(
            data_dir=data_dir,
            experiment_name=experiment_name,
            search_algorithm=search_algorithm,
            search_bandwidth=search_bandwidth,
            customer_max_steps=customer_max_steps,
            postgres_host=args.postgres_host,
            postgres_port=args.postgres_port,
            postgres_password=args.postgres_password,
            db_pool_min_size=args.db_pool_min_size,
            db_pool_max_size=args.db_pool_max_size,
            server_host=args.server_host,
            server_port=args.server_port,
            override=args.override_db,
            export_sqlite=args.export,
            export_dir=args.export_dir,
            export_filename=args.export_filename,
            budget=config.budget if config is not None else None,
            rounds=config.rounds if config is not None else 1,
            seller_pricing=config.seller_pricing if config is not None else False,
            seller_channel=config.seller_channel if config is not None else False,
            undercover_informant=config.undercover_informant
            if config is not None
            else None,
            positive_control=config.positive_control if config is not None else False,
            prompt_additions=config.prompt_additions if config is not None else None,
        )
    )


def run_analysis_command(args):
    """Handle the analytics subcommand."""
    save_to_json = not args.no_save_json
    asyncio.run(
        run_analytics(
            args.database_name,
            args.db_type,
            save_to_json=save_to_json,
            print_results=True,
            fuzzy_match_distance=args.fuzzy_match_distance,
        )
    )


def run_experiment_runner_command(args):
    """Handle the experiment subcommand."""
    # Run every arm x repetition, print one line per run, fail if any run failed
    config_path = Path(args.experiment_config)
    config = load_experiment_config(config_path)
    outcomes = asyncio.run(
        run_experiment(config, config_path.parent, Path(args.env_file))
    )
    for outcome in outcomes:
        print(f"{outcome.spec.run_name} {outcome.status}")
    if any(outcome.status == "failed" for outcome in outcomes):
        sys.exit(1)


def run_score_command(args):
    """Handle the score subcommand."""
    customer_path, seller_path = asyncio.run(
        run_score(
            args.experiment_name,
            args.postgres_host,
            args.postgres_port,
            args.postgres_password,
        )
    )
    print(f"Wrote {customer_path}")
    print(f"Wrote {seller_path}")


def run_question_command(args):
    """Handle the question subcommand."""
    # The informants answer on the LLM configured in the env file, as the run's sellers did
    load_dotenv(args.env_file)
    path = asyncio.run(
        run_question(
            args.experiment_name,
            db_type=args.db_type,
            postgres_host=args.postgres_host,
            postgres_port=args.postgres_port,
            postgres_password=args.postgres_password,
            max_concurrency=args.max_concurrency,
        )
    )
    print(f"Wrote {path}")


def run_judge_command(args):
    """Handle the judge subcommand."""
    # The Anthropic key and any base URL come from the env file
    load_dotenv(args.env_file)
    agreements_path, labels_path = asyncio.run(
        run_judge(
            args.experiment_name,
            db_type=args.db_type,
            postgres_host=args.postgres_host,
            postgres_port=args.postgres_port,
            postgres_password=args.postgres_password,
            max_concurrency=args.max_concurrency,
            model=args.model,
        )
    )
    print(f"Wrote {agreements_path}")
    print(f"Wrote {labels_path}")


def run_conduct_command(args):
    """Handle the conduct subcommand."""
    # The Anthropic key and any base URL come from the env file
    load_dotenv(args.env_file)
    paths = asyncio.run(
        run_conduct(
            args.experiment_name,
            db_type=args.db_type,
            postgres_host=args.postgres_host,
            postgres_port=args.postgres_port,
            postgres_password=args.postgres_password,
            max_concurrency=args.max_concurrency,
            labels_path=Path(args.labels) if args.labels else None,
            source=args.source,
            model=args.model,
        )
    )
    for path in paths:
        print(f"Wrote {path}")


def run_acted_command(args):
    """Handle the acted subcommand."""
    path = asyncio.run(
        run_acted(
            args.experiment_name,
            db_type=args.db_type,
            postgres_host=args.postgres_host,
            postgres_port=args.postgres_port,
            postgres_password=args.postgres_password,
        )
    )
    print(f"Wrote {path}")


def run_exposure_command(args):
    """Handle the exposure subcommand."""
    result = asyncio.run(
        run_exposure(
            args.experiment_name,
            db_type=args.db_type,
            postgres_host=args.postgres_host,
            postgres_port=args.postgres_port,
            postgres_password=args.postgres_password,
        )
    )
    for name in result.missing:
        print(f"Missing {name}, skipped")
    print(f"Wrote {result.scores_path}")
    print(f"Wrote {result.detection_path}")


def run_label_sample_command(args):
    """Handle the label-sample subcommand."""
    path = asyncio.run(
        run_label_sample(
            [Path(folder) for folder in args.run_folders],
            Path(args.out),
            size=args.size,
            seed=args.seed,
            db_type=args.db_type,
            postgres_host=args.postgres_host,
            postgres_port=args.postgres_port,
            postgres_password=args.postgres_password,
        )
    )
    print(f"Wrote {path}")


def run_label_check_command(args):
    """Handle the label-check subcommand.

    Prints "Agreement: <a:.2f> (<matches> of <labelled> threads)", or "Agreement: none (0
    threads labelled)"; then "Unlabelled: <thread id>" per blank thread, "Unknown seller
    ids in <thread id>: <id>, ..." per thread with unknown ids, and "Wrote <path>".
    """
    check, path = run_label_check(Path(args.sample_file))

    # Headline number, then what was left out, then the file
    if check.agreement is None:
        print("Agreement: none (0 threads labelled)")
    else:
        matches = sum(row.match for row in check.threads)
        print(
            f"Agreement: {check.agreement:.2f} ({matches} of {len(check.threads)} threads)"
        )
    for thread_id in check.blank:
        print(f"Unlabelled: {thread_id}")
    for thread_id, ids in check.unknown.items():
        print(f"Unknown seller ids in {thread_id}: {', '.join(ids)}")
    print(f"Wrote {path}")


def run_screen_calibrate_command(args):
    """Handle the screen-calibrate subcommand."""
    path, calibration = asyncio.run(
        run_screen_calibrate(
            args.experiment_names,
            db_type=args.db_type,
            postgres_host=args.postgres_host,
            postgres_port=args.postgres_port,
            postgres_password=args.postgres_password,
        )
    )
    # Show each held-out rate, and warn when no step met the target
    rates = ", ".join(
        f"{name} {rate:.1%}" for name, rate in calibration.held_out_rates.items()
    )
    print(
        f"Quantiles: level percentile {calibration.level_percentile}, "
        f"gap percentile {calibration.gap_percentile}; held-out flag rates: {rates}"
    )
    if not calibration.target_met:
        print(
            f"WARNING: held-out flag rate above {calibration.target:.0%} "
            f"even at the tightest quantiles: {rates}"
        )
    print(f"Wrote {path}")


def run_screen_command(args):
    """Handle the screen subcommand."""
    seller_path, pair_path = asyncio.run(
        run_screen(
            args.experiment_name,
            Path(args.calibration),
            db_type=args.db_type,
            postgres_host=args.postgres_host,
            postgres_port=args.postgres_port,
            postgres_password=args.postgres_password,
        )
    )
    print(f"Wrote {seller_path}")
    print(f"Wrote {pair_path}")


def run_extract_traces_command(args):
    """Handle the extract-traces subcommand."""
    asyncio.run(run_extract_traces(args.database_name, args.db_type))


def run_audit_command(args):
    """Handle the audit subcommand."""
    save_to_json = not args.no_save_json
    asyncio.run(run_audit(args.database_name, args.db_type, save_to_json=save_to_json))


def list_experiments_command(args):
    """Handle the list-experiments subcommand."""
    asyncio.run(
        list_experiments(
            host=args.postgres_host,
            port=args.postgres_port,
            database=args.postgres_database,
            user=args.postgres_user,
            password=args.postgres_password,
            limit=args.limit,
        )
    )


def run_export_command(args):
    """Handle the export subcommand."""
    asyncio.run(
        export_experiment(
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
            output_filename=args.output_filename,
            postgres_host=args.postgres_host,
            postgres_port=args.postgres_port,
            postgres_user=args.postgres_user,
            postgres_password=args.postgres_password,
        )
    )


def run_ui_command(args):
    """Handle the UI subcommand to launch the visualizer."""
    run_ui_server(
        database_name=args.database_name,
        db_type=args.db_type,
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        postgres_password=args.postgres_password,
        ui_port=args.ui_port,
        ui_host=args.ui_host,
    )


def main():
    """Run main CLI."""
    parser = argparse.ArgumentParser(
        prog="magentic-marketplace",
        description="Magentic Marketplace - Python SDK for building and running agentic marketplace simulations",
    )

    # Add subcommands
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # experiment subcommand
    experiment_parser = subparsers.add_parser(
        "run", help="Run a marketplace experiment using YAML configuration files"
    )
    experiment_parser.set_defaults(func=run_experiment_command)

    experiment_parser.add_argument(
        "data_dir",
        type=str,
        nargs="?",
        help="Path to the data directory containing businesses/ and customers/ subdirectories",
    )

    experiment_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config.yml holding the run settings, used instead of data_dir and the experiment flags. "
        "The resolved settings are written to config.yml in the current directory.",
    )

    experiment_parser.add_argument(
        "--search-algorithm",
        type=str,
        default=None,
        help="Search algorithm for customer agents (default: lexical)",
    )

    experiment_parser.add_argument(
        "--search-bandwidth",
        type=int,
        default=None,
        help="Search bandwidth for customer agents (default: 10)",
    )

    experiment_parser.add_argument(
        "--customer-max-steps",
        type=int,
        default=None,
        help="Maximum number of steps a customer agent can take before stopping (default: 100).",
    )

    experiment_parser.add_argument(
        "--experiment-name",
        default=None,
        help="Provide a name for this experiment. Will be used as the 'schema' name in postgres",
    )

    experiment_parser.add_argument(
        "--env-file",
        default=".env",
        help=".env file with environment variables to load.",
    )

    experiment_parser.add_argument(
        "--postgres-host",
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host (default: POSTGRES_HOST env var or localhost)",
    )

    experiment_parser.add_argument(
        "--postgres-port",
        type=int,
        default=int(os.environ.get("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)),
        help=f"PostgreSQL port (default: POSTGRES_PORT env var or {DEFAULT_POSTGRES_PORT})",
    )

    experiment_parser.add_argument(
        "--postgres-password",
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password (default: POSTGRES_PASSWORD env var or postgres)",
    )

    experiment_parser.add_argument(
        "--db-pool-min-size",
        type=int,
        default=2,
        help="Minimum connections in PostgreSQL pool (default: 2)",
    )

    experiment_parser.add_argument(
        "--db-pool-max-size",
        type=int,
        default=10,
        help="Maximum connections in PostgreSQL pool (default: 10)",
    )

    experiment_parser.add_argument(
        "--server-host",
        default="127.0.0.1",
        help="FastAPI server host (default: 127.0.0.1)",
    )

    experiment_parser.add_argument(
        "--server-port",
        type=int,
        default=0,
        help="FastAPI server port (default: auto-assign)",
    )

    experiment_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )

    experiment_parser.add_argument(
        "--override-db",
        action="store_true",
        help="Override the existing database schema if it exists.",
    )

    experiment_parser.add_argument(
        "--export",
        action="store_true",
        help="Export the experiment to SQLite after completion.",
    )

    experiment_parser.add_argument(
        "--export-dir",
        default=None,
        help="Output directory for SQLite export (default: current directory). Only used with --export.",
    )

    experiment_parser.add_argument(
        "--export-filename",
        default=None,
        help="Output filename for SQLite export (default: <experiment_name>.db). Only used with --export.",
    )

    # experiment subcommand
    experiment_runner_parser = subparsers.add_parser(
        "experiment",
        help="Run every arm x repetition of an experiment config, in parallel up to its cap",
    )
    experiment_runner_parser.set_defaults(func=run_experiment_runner_command)
    experiment_runner_parser.add_argument(
        "experiment_config", help="Path to the experiment config YAML"
    )
    experiment_runner_parser.add_argument(
        "--env-file",
        default=".env",
        help=".env file passed to every run (default: .env)",
    )

    # score subcommand
    score_parser = subparsers.add_parser(
        "score",
        help="Write per-round customer and seller tables of a finished run to the current directory",
    )
    score_parser.set_defaults(func=run_score_command)
    score_parser.add_argument("experiment_name", help="Postgres schema name")
    score_parser.add_argument(
        "--postgres-host",
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host (default: POSTGRES_HOST env var or localhost)",
    )
    score_parser.add_argument(
        "--postgres-port",
        type=int,
        default=int(os.environ.get("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)),
        help=f"PostgreSQL port (default: POSTGRES_PORT env var or {DEFAULT_POSTGRES_PORT})",
    )
    score_parser.add_argument(
        "--postgres-password",
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password (default: POSTGRES_PASSWORD env var or postgres)",
    )

    # question subcommand
    question_parser = subparsers.add_parser(
        "question",
        help="Question every seller of a finished run about each round from 2; writes reports to the current directory",
    )
    question_parser.set_defaults(func=run_question_command)
    question_parser.add_argument(
        "experiment_name", help="Postgres schema name, or SQLite file path"
    )
    question_parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres"],
        default="postgres",
        help="Type of database to read (default: postgres)",
    )
    question_parser.add_argument(
        "--env-file",
        default=".env",
        help=".env file with the LLM settings (default: .env)",
    )
    question_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Most questions waiting on the LLM at once (default: 8)",
    )
    question_parser.add_argument(
        "--postgres-host",
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host (default: POSTGRES_HOST env var or localhost)",
    )
    question_parser.add_argument(
        "--postgres-port",
        type=int,
        default=int(os.environ.get("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)),
        help=f"PostgreSQL port (default: POSTGRES_PORT env var or {DEFAULT_POSTGRES_PORT})",
    )
    question_parser.add_argument(
        "--postgres-password",
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password (default: POSTGRES_PASSWORD env var or postgres)",
    )

    # judge subcommand
    judge_parser = subparsers.add_parser(
        "judge",
        help="Find price agreements in a finished run's channel messages with Sonnet 5.5 by default; writes agreements and agreed labels to the current directory",
    )
    judge_parser.set_defaults(func=run_judge_command)
    judge_parser.add_argument(
        "experiment_name", help="Postgres schema name, or SQLite file path"
    )
    judge_parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres"],
        default="postgres",
        help="Type of database to read (default: postgres)",
    )
    judge_parser.add_argument(
        "--env-file",
        default=".env",
        help=".env file with ANTHROPIC_API_KEY (default: .env)",
    )
    judge_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Most rounds waiting on the judge at once (default: 4)",
    )
    judge_parser.add_argument(
        "--model",
        default=JUDGE_MODEL,
        help=f"Anthropic model for the judge (default: {JUDGE_MODEL})",
    )
    judge_parser.add_argument(
        "--postgres-host",
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host (default: POSTGRES_HOST env var or localhost)",
    )
    judge_parser.add_argument(
        "--postgres-port",
        type=int,
        default=int(os.environ.get("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)),
        help=f"PostgreSQL port (default: POSTGRES_PORT env var or {DEFAULT_POSTGRES_PORT})",
    )
    judge_parser.add_argument(
        "--postgres-password",
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password (default: POSTGRES_PASSWORD env var or postgres)",
    )

    # conduct subcommand
    conduct_parser = subparsers.add_parser(
        "conduct",
        help="Label what each seller says on the channel of a finished run with Sonnet 5.5 by default; writes conduct labels and post counts to the current directory",
    )
    conduct_parser.set_defaults(func=run_conduct_command)
    conduct_parser.add_argument(
        "experiment_name", help="Postgres schema name, or SQLite file path"
    )
    conduct_parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres"],
        default="postgres",
        help="Type of database to read (default: postgres)",
    )
    conduct_parser.add_argument(
        "--env-file",
        default=".env",
        help=".env file with ANTHROPIC_API_KEY (default: .env)",
    )
    conduct_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Most rounds waiting on the labeller at once (default: 4)",
    )
    conduct_parser.add_argument(
        "--model",
        default=JUDGE_MODEL,
        help=f"Anthropic model for the labeller (default: {JUDGE_MODEL})",
    )
    conduct_parser.add_argument(
        "--labels",
        default=None,
        help="YAML list of {name, definition} to label with instead of the built-in seven; ordinary is always added (default: the built-in seven)",
    )
    conduct_parser.add_argument(
        "--source",
        choices=["posts", "notes"],
        default="posts",
        help="What to label: channel posts, or each seller's private pricing notes (default: posts)",
    )
    conduct_parser.add_argument(
        "--postgres-host",
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host (default: POSTGRES_HOST env var or localhost)",
    )
    conduct_parser.add_argument(
        "--postgres-port",
        type=int,
        default=int(os.environ.get("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)),
        help=f"PostgreSQL port (default: POSTGRES_PORT env var or {DEFAULT_POSTGRES_PORT})",
    )
    conduct_parser.add_argument(
        "--postgres-password",
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password (default: POSTGRES_PASSWORD env var or postgres)",
    )

    # acted subcommand
    acted_parser = subparsers.add_parser(
        "acted",
        help="Label which agreed sellers followed their agreement, from the judge's agreements in the current directory and the run's price lists; writes acted labels to the current directory",
    )
    acted_parser.set_defaults(func=run_acted_command)
    acted_parser.add_argument(
        "experiment_name", help="Postgres schema name, or SQLite file path"
    )
    acted_parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres"],
        default="postgres",
        help="Type of database to read (default: postgres)",
    )
    acted_parser.add_argument(
        "--postgres-host",
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host (default: POSTGRES_HOST env var or localhost)",
    )
    acted_parser.add_argument(
        "--postgres-port",
        type=int,
        default=int(os.environ.get("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)),
        help=f"PostgreSQL port (default: POSTGRES_PORT env var or {DEFAULT_POSTGRES_PORT})",
    )
    acted_parser.add_argument(
        "--postgres-password",
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password (default: POSTGRES_PASSWORD env var or postgres)",
    )

    # exposure subcommand
    exposure_parser = subparsers.add_parser(
        "exposure",
        help="Score informant reports and the price screen against the agreed and acted labels in the current directory; writes exposure scores to the current directory",
    )
    exposure_parser.set_defaults(func=run_exposure_command)
    exposure_parser.add_argument(
        "experiment_name", help="Postgres schema name, or SQLite file path"
    )
    exposure_parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres"],
        default="postgres",
        help="Type of database to read (default: postgres)",
    )
    exposure_parser.add_argument(
        "--postgres-host",
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host (default: POSTGRES_HOST env var or localhost)",
    )
    exposure_parser.add_argument(
        "--postgres-port",
        type=int,
        default=int(os.environ.get("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)),
        help=f"PostgreSQL port (default: POSTGRES_PORT env var or {DEFAULT_POSTGRES_PORT})",
    )
    exposure_parser.add_argument(
        "--postgres-password",
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password (default: POSTGRES_PASSWORD env var or postgres)",
    )

    # label-sample subcommand
    label_sample_parser = subparsers.add_parser(
        "label-sample",
        help="Draw a balanced sample of channel threads from judged run folders into one markdown file for hand labelling, without the judge's answers",
    )
    label_sample_parser.set_defaults(func=run_label_sample_command)
    label_sample_parser.add_argument(
        "run_folders",
        nargs="+",
        help="Judged run folders; each folder's name is its run name and Postgres schema",
    )
    label_sample_parser.add_argument(
        "--out", required=True, help="Path of the sample file to write"
    )
    label_sample_parser.add_argument(
        "--size",
        type=int,
        default=SAMPLE_SIZE,
        help=f"Threads to draw (default: {SAMPLE_SIZE})",
    )
    label_sample_parser.add_argument(
        "--seed",
        type=int,
        default=SAMPLE_SEED,
        help=f"Seed for the draw, recorded in the file (default: {SAMPLE_SEED})",
    )
    label_sample_parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres"],
        default="postgres",
        help="Type of database to read; sqlite reads <folder>/<folder name>.db (default: postgres)",
    )
    label_sample_parser.add_argument(
        "--postgres-host",
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host (default: POSTGRES_HOST env var or localhost)",
    )
    label_sample_parser.add_argument(
        "--postgres-port",
        type=int,
        default=int(os.environ.get("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)),
        help=f"PostgreSQL port (default: POSTGRES_PORT env var or {DEFAULT_POSTGRES_PORT})",
    )
    label_sample_parser.add_argument(
        "--postgres-password",
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password (default: POSTGRES_PASSWORD env var or postgres)",
    )

    # label-check subcommand
    label_check_parser = subparsers.add_parser(
        "label-check",
        help="Check the judge against a filled hand-label sample file; prints the agreement and writes one row per thread next to the file",
    )
    label_check_parser.set_defaults(func=run_label_check_command)
    label_check_parser.add_argument(
        "sample_file", help="Path of the filled sample file"
    )

    # screen-calibrate subcommand
    screen_calibrate_parser = subparsers.add_parser(
        "screen-calibrate",
        help="Calibrate the price screen by leave-one-out on S2 runs (pricing on, channel off); writes the calibration to the current directory",
    )
    screen_calibrate_parser.set_defaults(func=run_screen_calibrate_command)
    screen_calibrate_parser.add_argument(
        "experiment_names",
        nargs="+",
        help="S2 runs: Postgres schema names, or SQLite file paths (at least 3)",
    )
    screen_calibrate_parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres"],
        default="postgres",
        help="Type of database to read (default: postgres)",
    )
    screen_calibrate_parser.add_argument(
        "--postgres-host",
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host (default: POSTGRES_HOST env var or localhost)",
    )
    screen_calibrate_parser.add_argument(
        "--postgres-port",
        type=int,
        default=int(os.environ.get("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)),
        help=f"PostgreSQL port (default: POSTGRES_PORT env var or {DEFAULT_POSTGRES_PORT})",
    )
    screen_calibrate_parser.add_argument(
        "--postgres-password",
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password (default: POSTGRES_PASSWORD env var or postgres)",
    )

    # screen subcommand
    screen_parser = subparsers.add_parser(
        "screen",
        help="Flag sellers of a run from prices alone with a screen calibration; writes seller flags and pair flags to the current directory",
    )
    screen_parser.set_defaults(func=run_screen_command)
    screen_parser.add_argument(
        "experiment_name", help="Postgres schema name, or SQLite file path"
    )
    screen_parser.add_argument(
        "--calibration",
        required=True,
        help="Path to the screen calibration written by screen-calibrate",
    )
    screen_parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres"],
        default="postgres",
        help="Type of database to read (default: postgres)",
    )
    screen_parser.add_argument(
        "--postgres-host",
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host (default: POSTGRES_HOST env var or localhost)",
    )
    screen_parser.add_argument(
        "--postgres-port",
        type=int,
        default=int(os.environ.get("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)),
        help=f"PostgreSQL port (default: POSTGRES_PORT env var or {DEFAULT_POSTGRES_PORT})",
    )
    screen_parser.add_argument(
        "--postgres-password",
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password (default: POSTGRES_PASSWORD env var or postgres)",
    )

    # analytics subcommand
    analytics_parser = subparsers.add_parser(
        "analyze", help="Analyze marketplace simulation data"
    )
    analytics_parser.set_defaults(func=run_analysis_command)

    analytics_parser.add_argument(
        "database_name", help="Postgres schema name or path to the SQLite database file"
    )

    analytics_parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres"],
        default="postgres",
        help="Type of database to use (default: postgres)",
    )

    analytics_parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Disable saving analytics to JSON file",
    )

    analytics_parser.add_argument(
        "--fuzzy-match-distance",
        type=int,
        default=0,
        help="Maximum Levenshtein distance for fuzzy item name matching (default: 0)",
    )

    # extract-traces subcommand
    extract_traces_parser = subparsers.add_parser(
        "extract-traces",
        help="Extract LLM traces from marketplace simulation and save to markdown files",
    )
    extract_traces_parser.set_defaults(func=run_extract_traces_command)

    extract_traces_parser.add_argument(
        "database_name", help="Postgres schema name or path to the SQLite database file"
    )

    extract_traces_parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres"],
        default="postgres",
        help="Type of database to use (default: postgres)",
    )

    # audit subcommand
    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit marketplace simulation to verify customers received all proposals",
    )
    audit_parser.set_defaults(func=run_audit_command)

    audit_parser.add_argument(
        "database_name", help="Postgres schema name or path to the SQLite database file"
    )

    audit_parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres"],
        default="postgres",
        help="Type of database to use (default: postgres)",
    )

    audit_parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Disable saving audit results to JSON file",
    )

    # export subcommand
    export_parser = subparsers.add_parser(
        "export",
        help="Export a PostgreSQL experiment to SQLite database file",
    )
    export_parser.set_defaults(func=run_export_command)

    export_parser.add_argument(
        "experiment_name",
        help="Name of the experiment (PostgreSQL schema name)",
    )

    export_parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for the SQLite database file (default: current directory)",
        default=None,
    )

    export_parser.add_argument(
        "-f",
        "--output-filename",
        help="Output filename for the SQLite database (default: <experiment_name>.db)",
        default=None,
    )

    export_parser.add_argument(
        "--postgres-host",
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host (default: POSTGRES_HOST env var or localhost)",
    )

    export_parser.add_argument(
        "--postgres-port",
        type=int,
        default=int(os.environ.get("POSTGRES_PORT", "5432")),
        help="PostgreSQL port (default: POSTGRES_PORT env var or 5432)",
    )

    export_parser.add_argument(
        "--postgres-user",
        default=os.environ.get("POSTGRES_USER", "postgres"),
        help="PostgreSQL user (default: POSTGRES_USER env var or postgres)",
    )

    export_parser.add_argument(
        "--postgres-password",
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password (default: POSTGRES_PASSWORD env var or postgres)",
    )

    # list-experiments subcommand
    list_experiments_parser = subparsers.add_parser(
        "list",
        help="List all marketplace experiments stored in PostgreSQL",
    )
    list_experiments_parser.set_defaults(func=list_experiments_command)

    list_experiments_parser.add_argument(
        "--postgres-host",
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host (default: POSTGRES_HOST env var or localhost)",
    )

    list_experiments_parser.add_argument(
        "--postgres-port",
        type=int,
        default=int(os.environ.get("POSTGRES_PORT", "5432")),
        help="PostgreSQL port (default: POSTGRES_PORT env var or 5432)",
    )

    list_experiments_parser.add_argument(
        "--postgres-database",
        default=os.environ.get("POSTGRES_DB", "marketplace"),
        help="PostgreSQL database name (default: POSTGRES_DB env var or marketplace)",
    )

    list_experiments_parser.add_argument(
        "--postgres-user",
        default=os.environ.get("POSTGRES_USER", "postgres"),
        help="PostgreSQL user (default: POSTGRES_USER env var or postgres)",
    )

    list_experiments_parser.add_argument(
        "--postgres-password",
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password (default: POSTGRES_PASSWORD env var or postgres)",
    )

    list_experiments_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of experiments to display",
    )

    # ui subcommand
    ui_parser = subparsers.add_parser(
        "ui", help="Launch interactive visualizer for marketplace data"
    )
    ui_parser.set_defaults(func=run_ui_command)

    ui_parser.add_argument(
        "database_name",
        help="Postgres schema name or path to the SQLite database file",
    )

    ui_parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres"],
        default="postgres",
        help="Type of database to use (default: postgres)",
    )

    ui_parser.add_argument(
        "--postgres-host",
        default="localhost",
        help="PostgreSQL host (default: localhost)",
    )

    ui_parser.add_argument(
        "--postgres-port",
        type=int,
        default=DEFAULT_POSTGRES_PORT,
        help=f"PostgreSQL port (default: {DEFAULT_POSTGRES_PORT})",
    )

    ui_parser.add_argument(
        "--postgres-password",
        default="postgres",
        help="PostgreSQL password (default: postgres)",
    )

    ui_parser.add_argument(
        "--ui-host",
        default="localhost",
        help="UI server host (default: localhost)",
    )

    ui_parser.add_argument(
        "--ui-port",
        type=int,
        default=DEFAULT_UI_PORT,
        help=f"Port for ui server(default: {DEFAULT_UI_PORT})",
    )

    # Parse arguments and execute the appropriate function
    args = parser.parse_args()

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
