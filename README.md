# LLM Seller Collusion: Undercover Agents and Monitoring

Research extensions to [Microsoft’s Magentic Marketplace](https://github.com/microsoft/multi-agent-marketplace),
studying whether LLM sellers coordinate prices, whether undercover sellers expose
that coordination, and how sellers respond to monitoring.

## Research setup

Four restaurant sellers compete for eight customers over repeated rounds. Sellers
can update prices, remember previous sales, and message one another. Optional
instructions introduce undercover sellers or tell sellers their messages are monitored.

The project adds customer budgets, seller pricing and memory, seller messaging,
configurable models by role, an experiment runner, and tools for scoring purchases,
labelling price talk, and checking accusations against quoted messages.
Price talk and realized price changes are measured separately.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker, and model API credentials.
Simulation and evaluator calls can incur API charges.

```bash
# Clone this repository and install its dependencies.
git clone https://github.com/strivn/collusion-research-hackathon.git
cd collusion-research-hackathon
uv sync --frozen

# Configure credentials locally, then start Postgres.
cp sample.env .env
# Edit .env to select your provider and model and supply API keys.
docker compose up -d
```

```bash
# Run a stock baseline in its own output folder.
mkdir -p research/results/example
cd research/results/example
uv run --project ../../.. magentic-marketplace run \
  ../../../data/mexican_3_9 --experiment-name portfolio_example \
  --env-file ../../../.env
```

Use a new experiment name for each run; names identify database schemas.
Run commands from their output folder because metrics and traces are written there.
Research settings are available through `run --config`; multi-arm experiments use
`experiment`. See [configuration fields](packages/magentic-marketplace/src/magentic_marketplace/experiments/run_config.py),
[experiment runner](packages/magentic-marketplace/src/magentic_marketplace/experiments/experiment_runner.py),
and `magentic-marketplace --help`.

## Development

```bash
# Run quality checks and the CI test selection.
uv run poe check-all
uv run pytest -m "not skip_ci and not rnr and not postgres"
```

Full type checking requires the optional search dependencies (`uv sync --all-extras`).
Development uses `develop`. Frozen `main` preserves the unmodified upstream baseline
at commit `9a651c7`. New simulation features default to stock behavior.

## Upstream credit

The base framework, original datasets, visualizer, and upstream documentation come
from Microsoft’s Magentic Marketplace. This repository retains its [MIT license](LICENSE).
The [upstream documentation](https://microsoft.github.io/multi-agent-marketplace/)
describes the base framework. Cite its [paper](https://arxiv.org/abs/2510.25779) when using it:

```
@misc{bansal-arxiv-2025,
      title={Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets}, 
      author={Gagan Bansal and Wenyue Hua and Zezhou Huang and Adam Fourney and Amanda Swearngin and Will Epperson and Tyler Payne and Jake M. Hofman and Brendan Lucier and Chinmay Singh and Markus Mobius and Akshay Nambi and Archana Yadav and Kevin Gao and David M. Rothschild and Aleksandrs Slivkins and Daniel G. Goldstein and Hussein Mozannar and Nicole Immorlica and Maya Murad and Matthew Vogel and Subbarao Kambhampati and Eric Horvitz and Saleema Amershi},
      year={2025},
      eprint={2510.25779},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2510.25779}, 
}
```
