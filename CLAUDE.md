# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A private copy of Microsoft's Magentic Marketplace (LLM buyer and seller agents in a
simulated market), with our own patches.

## Repo boundary

- Remote is `strivn/collusion-research-hackathon` (private). Never fetch from, push to,
  or open PRs, issues or comments on `microsoft/multi-agent-marketplace`, and never add
  it as a remote.
- Links inside the code and `docs/` (e.g. the `pr-comments` poe task, docs URLs) point
  at Microsoft repos. Do not run them.
- Branches: `main` is frozen at `9a651c7`, the unmodified stock code. `develop` is the
  trunk and default branch.
- `docs/` is upstream's documentation. Our research docs live in `research/`, which is
  excluded via `.git/info/exclude` and never committed. Read `research/` before changing
  agent behaviour.
- CI workflows (`.github/workflows/test.yml` etc.) trigger on `main` only.

## Commands

Run from the repo root. Python >=3.11, managed by `uv` workspace (member:
`packages/magentic-marketplace`).

```bash
uv sync                           # install; add --all-extras only for rnr search (torch)
cp sample.env .env                # LLM provider/model/keys + Postgres creds
docker compose up -d              # Postgres 16 on :5432 (+ pgadmin on :8080); required for runs

magentic-marketplace run data/mexican_3_9 --experiment-name <name>   # one simulation
magentic-marketplace analyze <name>        # welfare/revenue report, writes JSON
magentic-marketplace extract-traces <name> # LLM prompts/responses to markdown
magentic-marketplace audit <name>          # did customers receive every proposal
magentic-marketplace list | export <name> | ui <name>
```

`--experiment-name` becomes the Postgres schema; reusing a name fails unless
`--override-db`. The CLI loads `.env` from the current directory (`--env-file`).

Quality checks (poe tasks in root `pyproject.toml`):

```bash
uv run poe check-all     # ruff format --check, ruff check, pyright packages, codespell
uv run poe fix-all       # ruff format + ruff check --fix
uv run pytest -m "not skip_ci and not rnr and not postgres"   # what CI runs
uv run pytest packages/magentic-marketplace/tests/protocol/test_send_message.py::test_name
```

Postgres-marked tests need the docker database and `dev.env`. Ruff enforces docstrings
(`D` rules).

Visualizer (`packages/marketplace-visualizer`, React + Vite): `npm run dev`, `npm run check`.
`npm run build` writes into `packages/magentic-marketplace/src/magentic_marketplace/ui/static`,
which `magentic-marketplace ui` serves.

## Architecture

`P = packages/magentic-marketplace/src/magentic_marketplace`

Two layers:

- `P/platform/` is domain-agnostic: a FastAPI server (`server/`), an HTTP client, a
  `BaseAgent` with a `run()` loop that calls `step()` until `shutdown()`, a database
  controller (Postgres for runs, SQLite only as export target), and
  `BaseMarketplaceProtocol`.
- `P/marketplace/` is the restaurant market on top: actions, protocol, agents, LLM clients.

Flow of one run (`P/experiments/run_experiment.py`):

1. Load `businesses/*.yaml` and `customers/*.yaml` from the data dir.
2. `MarketplaceLauncher` starts the server in-process with `SimpleMarketplaceProtocol`
   and a fresh Postgres schema.
3. `AgentLauncher.run_agents_with_dependencies` runs every agent as a concurrent asyncio
   task. Customers are primary; when all customers finish, businesses are told to shut
   down. There are no rounds and no state carried between runs.

Protocol (`P/marketplace/protocol/protocol.py`): three actions, `Search`, `SendMessage`,
`FetchMessages`, defined in `P/marketplace/actions/actions.py`. A message is a
`TextMessage`, `OrderProposal` or `Payment` (`actions/messaging.py`). Adding an action
means touching the action union, `get_actions`, `execute_action` and the agent's tool
set. `send_message` only checks that the recipient exists, so any agent can message any
agent.

Agents (`P/marketplace/agents/`), both subclass `BaseSimpleMarketplaceAgent`, which
wraps send/fetch and LLM calls (`generate`, `generate_struct`) and logs each call:

- `customer/`: each step asks the LLM for one of `search_businesses`, `send_messages`
  (text or `Payment`), `check_messages`, `end_transaction` (`customer/models.py`).
  Prompts in `customer/prompts.py`. Stops at `--customer-max-steps`.
- `business/`: polls messages every 2 s, groups by customer, answers text with an LLM
  reply or `OrderProposal` (`business/responses.py`, prompts in `business/prompts.py`),
  confirms payments against its `OrderProposalStorage`. Keeps one history per customer;
  it never sees its other conversations. Stock sellers only reply, never initiate.
  Proposal validation (`responses.py`) checks only that item names are on the menu and
  that `total_price` equals the sum of the items, retrying up to 3 times. Unit prices are
  not checked against the menu; only the prompt line "ONLY tell potential customers what
  you have on the menu with CORRECT PRICES" keeps them there.
- Sellers often reply with a proposal and no text message, putting their whole answer
  (amenities included) in the proposal's `special_instructions`. Query both message types
  when reading what a seller said.
- The customer prompt re-renders the full action trajectory every step
  (`customer/prompts.py`), so prompt size grows with steps. Each business call sees the
  full history with that one customer.

LLM (`P/marketplace/llm/`): provider and model come from env (`LLM_PROVIDER`
openai|gemini|anthropic, `LLM_MODEL`, `LLM_REASONING_EFFORT`, `LLM_MAX_CONCURRENCY`) via
`BaseLLMConfig`. Only the OpenAI client accepts a custom base URL (`OPENAI_BASE_URL`).

Analytics (`P/experiments/run_analytics.py`): customer utility is
2 x sum(`menu_features` prices) if the order matches the request, minus payment. The
customer agent never sees that number.

Data (`data/<cuisine>_<customers>_<businesses>/`): customer YAML has `request`,
`menu_features` (item -> price used for scoring) and `amenity_features`; business YAML
has `menu_features` (menu prices), `amenity_features`, `rating`, `progenitor_customer`
and `min_price_factor`. `min_price_factor` is declared in `P/marketplace/shared/models.py`
but no code reads it.

Logs: every action and LLM call is a JSONB row in the schema's `agents`, `actions` and
`logs` tables. Messages are fetched in `row_index` order.

## Gotchas

- `docker compose` needs `.env` (it is the `env_file` for both services), not just the CLI.
  Do not read `.env`; it holds API keys.
- Anthropic client (`llm/clients/anthropic.py`) with current models: `claude-haiku-5-5`
  returns 400 on a non-default `temperature`, so leave `LLM_TEMPERATURE` unset.
  `LLM_REASONING_EFFORT=minimal` sends `thinking: {type: "disabled"}`, which Haiku 5.5
  accepts at its default effort. Opus 5.5 and Sonnet 5.5 reject both disabled thinking
  and the forced `tool_choice` used for structured output, so they need a client patch.
- The `logs` table stores one combined token count per LLM call (input + output), so
  spend can only be bounded, not computed, without patching the clients.
- `analyze` and `extract-traces` write `analytics_results_<name>.json` and
  `<name>-agent-llm-traces/` into the current directory. Run them from a scratch dir.
- `ui` defaults to port 5000, which macOS AirPlay Receiver usually holds; pass `--ui-port`.
- Inspect a run directly: `docker compose exec -T postgres psql -U postgres -d marketplace`,
  then query `<schema>.actions` (message under `data->'request'->'parameters'->'message'`)
  and `<schema>.logs` (`data->'data'->>'type' = 'llm_call'`, with `prompt`, `response`,
  `token_count`, `duration_ms`).
