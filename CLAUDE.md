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
  trunk and default branch. No branch protection is possible on this private free-plan
  repo: never force-push, delete or commit directly to either.
- Every new feature sits behind a setting whose default reproduces stock behaviour, so a
  run with all new features off still matches `main`.

## Working area

| Area | Path | In git |
|---|---|---|
| Upstream code we patch | `packages/`, `data/` | yes |
| Upstream documentation | `docs/` | yes, left untouched |
| Our experiment code (run config, runner, scoring, judge, screen), next to stock modules such as `run_analytics.py` | `packages/magentic-marketplace/src/magentic_marketplace/experiments/` | yes, code only |
| Our datasets | `data/<name>/` | yes |
| Research notes, tracker, run outputs | `research/` (run outputs in `research/results/<run>/`) | no, excluded via `.git/info/exclude` |
| Local agent context | `CLAUDE.local.md`, `.claude/` | no |
| Secrets | `.env` | no; never read it |
| Throwaway files | the session scratchpad, outside the repo | n/a |

- Run every `magentic-marketplace` command from its run folder,
  `research/results/<run>/`, with `--env-file` pointing at the repo `.env`. The CLI and
  the database code write metrics, analytics and traces into the current directory.
- No new top-level files or folders without asking Ivan.
- Clean check: `git status --short --ignored` at the repo root lists only `.claude/`,
  `.env`, `.remember/`, `.venv/`, `CLAUDE.local.md`, `research/`, `__pycache__/` and
  `node_modules/` entries. Anything else is a stray file: move it to its run folder or
  delete it.
- `.gitignore` ignores every `*.json`. A JSON file that belongs in git needs `git add -f`
  and a mention in the PR.

## Commands

Run from the repo root. Python >=3.11, managed by `uv` workspace (member:
`packages/magentic-marketplace`).

```bash
uv sync                           # install; add --all-extras only for rnr search (torch)
cp sample.env .env                # LLM provider/model/keys + Postgres creds
docker compose up -d              # Postgres 16 on :5432 (+ pgadmin on :8080); required for runs

magentic-marketplace run data/mexican_3_9 --experiment-name <name>   # one stock simulation
magentic-marketplace run --config config.yml --env-file <repo>/.env   # one run with our settings
magentic-marketplace experiment <experiment>.yml --env-file .env      # arms x repetitions, from repo root
magentic-marketplace analyze <name>        # stock welfare/revenue report, writes JSON; not round-aware
magentic-marketplace extract-traces <name> # LLM prompts/responses to markdown
magentic-marketplace audit <name>          # did customers receive every proposal
magentic-marketplace list | export <name> | ui <name>
```

`--experiment-name` (or `experiment_name` in the config) becomes the Postgres schema;
reusing a name fails unless `--override-db`. The CLI loads `.env` from the current
directory unless `--env-file` says otherwise. `--config` cannot be combined with
`data_dir` or the `--experiment-name`, `--search-*` and `--customer-max-steps` flags.
Post-run commands (`score`, `judge`, ...) are in their own section below.

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

1. Load `businesses/*.yaml` and `customers/*.yaml` from the data dir. With the informant
   or budgets on, write `informant.yml` and `budgets.yml` into the current directory.
2. `MarketplaceLauncher` starts the server in-process with `SimpleMarketplaceProtocol`
   and a fresh Postgres schema.
3. For each round, `AgentLauncher.run_agents_with_dependencies` runs a fresh set of
   agents as concurrent asyncio tasks. Customers are primary; when all customers finish,
   businesses are told to shut down, and only then does the next round start. One round
   (the default) is the stock run. Rounds are described below.

Protocol (`P/marketplace/protocol/protocol.py`): five actions defined in
`P/marketplace/actions/actions.py`: stock `Search`, `SendMessage`, `FetchMessages`, and
the seller channel's `SendChannelMessage`, `FetchChannelMessages`. A message is a
`TextMessage`, `OrderProposal` or `Payment` (`actions/messaging.py`). Adding an action
means touching the action union, `get_actions`, `execute_action` and the agent's tool
set. `send_message` only checks that the recipient exists, so any agent can message any
agent. Every action an agent executes carries `{"round": n}` in its request metadata
(`agents/base.py`), and every LLM call log carries `round` in its log metadata.

Agents (`P/marketplace/agents/`), both subclass `BaseSimpleMarketplaceAgent`, which
wraps send/fetch and LLM calls (`generate`, `generate_struct`) and logs each call:

- `customer/`: each step asks the LLM for one of `search_businesses`, `send_messages`
  (text or `Payment`), `check_messages`, `end_transaction` (`customer/models.py`).
  Prompts in `customer/prompts.py`. Stops at `--customer-max-steps`. With a budget, the
  system prompt states it and tells the agent not to reveal it, and the agent refuses
  any payment whose proposal total exceeds the budget (compared in cents), logging a
  `refused_payment` row.
- `business/`: polls messages every 2 s, groups by customer, answers text with an LLM
  reply or `OrderProposal` (`business/responses.py`, prompts in `business/prompts.py`),
  confirms payments against its `OrderProposalStorage`. Keeps one history per customer;
  it never sees its other conversations with customers. Sellers never message customers
  first. Proposal validation (`responses.py`) checks that item names are on the menu and
  that `total_price` equals the sum of the items, retrying up to 3 times. With pricing
  off, unit prices are not checked; only the prompt line "ONLY tell potential customers
  what you have on the menu with CORRECT PRICES" keeps them at menu prices. With pricing
  on, every unit price must be within 0.005 of the seller's price for the round, and the
  prompt shows the round prices instead of the menu prices.
- Sellers often reply with a proposal and no text message, putting their whole answer
  (amenities included) in the proposal's `special_instructions`. Query both message types
  when reading what a seller said.
- The customer prompt re-renders the full action trajectory every step
  (`customer/prompts.py`), so prompt size grows with steps. Each business call sees the
  full history with that one customer.
- Each agent fetches only messages above `messages_after_index`, the last action row of
  the previous round, so a customer or seller never sees earlier rounds' conversations.
  From round 2, proposal ids gain `_r<round>` (`<seller>_<customer>_r2_1`) so they never
  repeat across rounds.

LLM (`P/marketplace/llm/`): provider and model come from env (`LLM_PROVIDER`
openai|gemini|anthropic, `LLM_MODEL`, `LLM_REASONING_EFFORT`, `LLM_MAX_CONCURRENCY`) via
`BaseLLMConfig`, unless the run config's `llm:` block sets them per role (Run config
below). Only the OpenAI client accepts a custom base URL (`OPENAI_BASE_URL`, or
`base_url` in the `llm:` block). Each agent merges its role's arguments over its
`llm_config` in `generate`/`generate_struct` (`agents/base.py`); a business agent sends its
talk and pricing calls with `strategy_llm_kwargs` (`strategy_generate_struct`) and its
customer replies with `llm_kwargs`. `set_requests_per_minute` (`llm/base.py`) spaces the
starts of every LLM call in the process.

Analytics (`P/experiments/run_analytics.py`): customer utility is
2 x sum(`menu_features` prices) if the order matches the request, minus payment. The
customer agent never sees that number.

Data (`data/<cuisine>_<customers>_<businesses>/`): customer YAML has `request`,
`menu_features` (item -> price used for scoring) and `amenity_features`; business YAML
has `menu_features` (menu prices), `amenity_features`, `rating`, `progenitor_customer`
and `min_price_factor`. `min_price_factor` is declared in `P/marketplace/shared/models.py`
but no code reads it.

Market shape: in the stock datasets (`mexican_3_9`, `_10_30`, `_33_99`, `_100_300`) each
customer has 3 sellers with its items, 2 of them with its amenities, and each seller can
serve at most 1 customer: disjoint trios, so 2 of 3 sellers sell nothing each round. Our
datasets: `data/mexican_3_9_demanded` and `_demanded_plus2` (menus cut to the demanded items,
`experiments/collusion/menu_variants.py`), and `data/mexican_oligopoly_4_8`
(`experiments/collusion/oligopoly_dataset.py`): 4 sellers with all 6 items and every
amenity, round-1 price = reference x {0.92, 0.97, 1.03, 1.08} rotated per item, and 8
customers with 1-3 items each, so every customer can buy from all 4 sellers.

Logs: every action and LLM call is a JSONB row in the schema's `agents`, `actions` and
`logs` tables. Messages are fetched in `row_index` order. Our features add log rows with
`data->'data'->>'type'` = `budget_assignment` (once, before round 1), `price_list` (one
per seller per round with pricing on, with the seller's notes), `channel_message` (one
per delivered recipient), `refused_payment`, `speaking_order` (one per talk turn with
`random_speaking_order`) and `first_word` (one per talk call with `random_first_word`).
The post-run commands read these rows.

## Run config

`P/experiments/run_config.py`, read by `run --config`. Unknown keys are rejected. Each
optional field defaults to stock behaviour:

| Field | Default | Effect |
|---|---|---|
| `data`, `experiment_name` | required | data dir; Postgres schema |
| `search_algorithm`, `search_bandwidth`, `customer_max_steps` | `lexical`, 10, 100 | stock CLI flags |
| `budget: {multipliers, seed}` | none | budget = m x reference price (sum of the customer's `menu_features`), m dealt from `multipliers` so counts differ by at most one (`experiments/budgets.py`) |
| `rounds` | 1 | number of rounds |
| `seller_pricing` | false | sellers set a price list each round from round 2 |
| `seller_channel` | false | sellers talk before pricing, from round 2 |
| `undercover_informant: {seed}` | none | one seller, picked by seed, is the informant. Requires `seller_channel: true`; the config is refused otherwise |
| `undercover_informant.instruction` | none | replaces `UNDERCOVER_INSTRUCTION` in the informant's talk, pricing and `question` framing |
| `prompt_additions: [{text, sellers, prompts, place}]` | `[]` | extra seller text. `sellers`: `all`, `ordinary` (all but the informant) or `undercover`; `prompts`: any of `talk`, `pricing`; `place`: `top` (after the undercover and coordination instructions, then a blank line) or `after_goal` (after the revenue goal line, after a space). Config order is kept. Customer replies never get them. An `undercover` addition needs `undercover_informant` and a `talk` addition needs `seller_channel`; the config is refused otherwise |
| `minimal_talk_prompt` | false | the talk prompt names the tool only ("You also have a message tool to other sellers."): no "Before setting your prices..." sentence, no revenue goal line, no "Sending messages is optional." Memory, allowance, messages and answer format stay; after_goal additions stand where the goal line was. The pricing prompt is unchanged |
| `random_speaking_order: {seed}` | none | in every talk turn the asked sellers speak one at a time, in `sorted(ids)` shuffled by `random.Random(f"{seed}:{round}")` (one generator per round, drawn turn after turn), so a later speaker's prompt shows the messages sent to it earlier in the turn. Turn rules unchanged. One `speaking_order` log row (round, turn, seller_ids in speaking order) per turn. Unset: a turn's sellers talk concurrently |
| `random_first_word: {seed}` | none | each talk prompt gets `Start your message with the word "<WORD>".` right before the answer-format line; WORD = `random.Random(f"{seed}:{round}:{turn}:{seller_id}").choice(FIRST_WORDS)` (24 neutral words, `agents/business/talk.py`). One `first_word` log row (seller_id, round, turn, word) per talk call. The pricing prompt and customer replies are unchanged |
| `llm: {customers, seller_replies, seller_strategy, requests_per_minute}` | all unset | model per role, each `{provider, model, reasoning_effort, base_url, api_key_env}`; an unset role uses the env LLM. `seller_strategy` makes the talk and pricing calls, `seller_replies` the replies to customers. `api_key_env` names the env variable holding the key (the config never holds a key); a `base_url` without `api_key_env` is refused. `requests_per_minute` caps LLM call starts per run process |

Setups: Pricing Only = `rounds` > 1 and `seller_pricing`; Pricing + Messaging adds
`seller_channel`; Pricing + Messaging + Undercover adds `undercover_informant`.

The validator checks only the informant and `prompt_additions` rules. Other combinations run but do nothing
useful: any feature with `rounds: 1`; `seller_channel` without `seller_pricing` (sellers
talk, but no pricing call reads the messages); an informant without `seller_pricing`
(`question` reads only pricing calls, so `informant_reports.jsonl` comes out empty).

`data` resolves against the current directory. The runner makes it absolute; a
hand-written `config.yml` run from `research/results/<run>/` needs an absolute path.

`run --config` writes the resolved settings (defaults included) to `config.yml`, plus
`budgets.yml` (customer id, m, budget) and `informant.yml` (seed, seller id) when those
features are on, all into the current directory, the run folder. `question` reads
`informant.yml` and the undercover instruction in `config.yml` back from there.

## Rounds and pricing

`P/experiments/run_experiment.py` runs rounds in sequence with fresh agent objects each
round; nothing carries over except what is listed here.

- Round 1 always sells at menu prices. With `seller_pricing`, each seller logs its menu
  prices as the round-1 `price_list` and makes no pricing call.
- From round 2, each seller makes one pricing call before serving customers
  (`agents/business/pricing.py`): last round's prices, the goal "maximize total revenue
  over all rounds", its memory, and with the channel on this round's channel messages.
  A list with a missing item, an off-menu item or a price <= 0 is sent back up to 3
  times; after that the seller keeps last round's prices and its notes are empty, so
  next round's memory has no notes. Prices are rounded to cents.
  The prompts never state the total number of rounds.
- Seller memory (`experiments/seller_memory.py`, built only with pricing on): the notes
  the seller wrote in its last pricing answer (its only free-form memory), its own price,
  units sold and revenue per item for every past round, and every rival's last-round
  price for each item it also sells. The sales numbers come from `round_scores.py`.
- Round gate (`agents/round_gate.py`): customers wait until every seller of the round has
  finished talking and pricing. A seller whose start-up fails still releases the gate.
  After 300 s customers start anyway and log a warning.

## Seller channel and talk phase

`P/marketplace/protocol/channel.py` and `P/marketplace/agents/business/talk.py`.

- Only sellers may send or fetch; recipients must be distinct other sellers.
- Allowance: `CHANNEL_ALLOWANCE` = 5 messages per seller per round; a message to k
  sellers uses k. A send that needs more than the seller has left is refused whole and
  nothing is delivered. Refused sends stay in `actions` with an error result and do not
  count against the allowance.
- Talk phase, from round 2 with `seller_channel`: turn 1 asks every seller at once.
  Each later turn asks only the sellers that received a message in the previous turn and
  still have allowance left. The phase ends when a turn delivers nothing. A seller whose
  talk call raises counts as sending nothing.
- The talk prompt states the round number and the goal "maximize total revenue over all
  rounds", lists other sellers by name and id only, and shows the messages left, the
  seller's memory (without the ask for notes) and this round's messages so far. It says
  sending is optional and never mentions collusion.
- Storage: each send is one `SendChannelMessage` row in `actions` (recipients under
  `data->'request'->'parameters'->'to_seller_ids'`, round in the request metadata). Each
  delivered recipient also gets one `channel_message` row in `logs` (sender, recipient,
  round, turn, text). `judge`, `exposure` and `label-sample` read the `logs` rows
  (`judge.load_channel_posts`).

## Undercover informant

- `P/marketplace/agents/business/informant.py` holds the texts: `UNDERCOVER_INSTRUCTION`
  (privately works for the Market Authority, flag suspicious behaviour, do not reveal
  it), `PRESSURE_QUESTION`, `REPORT_REQUEST`.
- `P/experiments/informant.py` picks the seller: `random.Random(seed).choice` over the
  sorted seller ids, written to `informant.yml`.
- The informant's talk and pricing prompts open with `UNDERCOVER_INSTRUCTION`, or with
  `undercover_informant.instruction` when the config sets one. Its replies to customers
  are unchanged.

## Experiment runner

`P/experiments/experiment_runner.py`, via `magentic-marketplace experiment <file>.yml`.
Run it from the repo root: `results_dir`, `tracker` and each arm's `data` path resolve
against the current directory; arm config paths resolve against the experiment file.

| Key | Default | Meaning |
|---|---|---|
| `name` | required | experiment name |
| `arms` | required | arm name -> run config file |
| `repetitions` | required | runs per arm |
| `base_seed` | required | repetition i uses seed `base_seed + i - 1` |
| `parallel` | 1 | runs at once |
| `results_dir` | `research/results` | parent of the run folders |
| `tracker` | `research/tracker.md` | file the runner appends one row per run to |
| `rq` | "" | research question written into the tracker row |

- Run `<arm>_rep<i>`, schema `<name>_<arm>_rep<i>` (stored lowercase), folder
  `<results_dir>/<name>/<arm>_rep<i>/`. The runner writes the arm's config with the
  run's schema and seed to `input.yml` there, then runs `run --config input.yml` and
  `score` in that folder as subprocesses.
- The repetition seed `base_seed + i - 1` replaces the budget, `random_speaking_order`
  and `random_first_word` seeds, so arms of one repetition share customer budgets,
  speaking orders and first words. The informant seed is `base_seed` in every run, and
  repetition i plants the seller at position i - 1 of its shuffle. Runs are ordered
  repetition first so paired runs start together.
- Each run is its own process with its own LLM semaphore, so up to `parallel` x
  `LLM_MAX_CONCURRENCY` (default 64) calls can be in flight, and up to `parallel` x
  `llm.requests_per_minute` calls start per minute.
- The tracker's model column names `llm.seller_strategy.model` when set, else `LLM_MODEL`.
- The tracker must contain a line exactly `## Runs` followed by a table; the row goes
  after its last line. Without the heading, the first run to finish raises `ValueError`
  and ends the command. A failed run never stops the others; the command exits 1 if any
  failed.
- The runner never passes `--override-db`, so re-running an experiment `name` whose
  schemas exist fails every run, each with a "failed" tracker row.

## Post-run commands

Run each per-run command from its run folder. The commands read databases (Postgres
schema by default, `--db-type sqlite` takes a file path; `score` is Postgres only) and
never write to them. Each writes where its row says.

| Command | Reads (besides the database) | LLM | Writes |
|---|---|---|---|
| `score <schema>` | nothing | none | `customer_rounds.csv` (value, budget, price paid, customer share, seller share, lost per customer-round), `seller_rounds.csv` (price, units sold, revenue per seller-item-round) |
| `question <schema>` | `informant.yml` and `config.yml` if present | `llm.seller_strategy` of `config.yml` when set, else env LLM (`LLM_MODEL`) | `informant_reports.jsonl`: every seller, each round from 2, shown its last pricing prompt and answer; the informant framed by `UNDERCOVER_INSTRUCTION` (or the config's `undercover_informant.instruction`), every other seller by `PRESSURE_QUESTION`. Accusations need a known seller id and a quote; bad ones are sent back up to 3 times, then dropped |
| `judge <schema>` | nothing | Sonnet 5.5, paid API | `judge_agreements.jsonl` (one line per round from 2; a round without channel messages gets `called: false` and no call), `agreed_labels.csv` (true from the round a seller joins an agreement; empty, meaning unknown, for a not-yet-agreed seller from the first round whose call failed) |
| `conduct <schema> [--labels <file>] [--source posts\|notes]` | `informant.yml` if present; the labels file | Sonnet 5.5, paid API | posts (default): `conduct_labels.csv` (round, sender, recipients, turn, label, quote, role), `conduct_counts.csv` (posts sent per seller-round); notes: `conduct_notes_labels.csv` (seller_id, round, label, quote, role) from each seller's pricing notes, no channel needed. The labeller is not told who the informant is. Labels: the built-in seven, or the file's YAML list of `{name, definition}` (example: `experiments/collusion/conduct_labels_v2.yml`); `ordinary` is always added last and a file may not list it. An answer with an unknown label is sent back up to 3 times, then the round is unlabelled |
| `acted <schema>` | `judge_agreements.jsonl` | none | `acted_labels.csv`: did each agreed seller's price list follow its latest agreement |
| `screen-calibrate <schema>...` | several databases | none | `screen_calibration.json` in the current directory, leave-one-out over at least 3 Pricing Only runs |
| `screen <schema> --calibration <path>` | the calibration | none | `screen_flags.csv`, `screen_pairs.csv`: sellers flagged from quoted prices alone, never messages |
| `exposure <schema>` | the four files above (reports, screen flags, agreed and acted labels); a missing one is skipped and named | none | `exposure_scores.csv` (precision, recall, grounding per detector, label, round), `exposure_detection.csv` (time to detection) |
| `label-sample <folders>... --out <file>` | each folder's `judge_agreements.jsonl`, and the database named after the folder | none | `--out`: a markdown sample of channel threads for hand labelling, without the judge's answers |
| `label-check <file>` | the filled sample and each named folder's `judge_agreements.jsonl`; no database | none | `<file stem>.check.csv`, next to the sample; prints judge-vs-hand agreement |

`label-sample` uses each folder's name as its schema. Runner folders are named
`<arm>_rep<i>` but their schemas `<name>_<arm>_rep<i>`, so on runner output it connects
to a schema that does not exist.

`judge` calls `anthropic.AsyncAnthropic` directly, not the repo's LLM client: model
`claude-sonnet-5-5`, effort high, JSON-schema output, server-side fallback beta. It takes
`ANTHROPIC_API_KEY` from `--env-file`, so it bills the API key, not a subscription.
`load_dotenv` does not override variables already set, so a key or
`ANTHROPIC_BASE_URL` exported in the shell wins over the file.

## Gotchas

- `docker compose` needs `.env` (it is the `env_file` for both services), not just the CLI.
  Do not read `.env`; it holds API keys.
- Anthropic client (`llm/clients/anthropic.py`) with current models: `claude-haiku-5-5`
  returns 400 on a non-default `temperature`, so leave `LLM_TEMPERATURE` unset.
  `LLM_REASONING_EFFORT=minimal` sends `thinking: {type: "disabled"}`, which Haiku 5.5
  accepts at its default effort. Every other string (`low`, `high`) also disables
  thinking; only an integer budget turns it on. Opus 5.5 and Sonnet 5.5 reject both disabled thinking
  and the forced `tool_choice` used for structured output, so they need a client patch.
  When a structured answer fails validation, the client retries with the error as a
  `tool_result` for the rejected `tool_use` (before #46 the retry was a 400 and the call
  was lost).
- OpenAI client: `reasoning_effort` `minimal` is sent as is to gpt-5 models (gpt-5-mini:
  0 reasoning tokens) and turned into `low` only for o-models. Before #46 every reasoning
  model got `low`, so earlier GPT-5 runs reasoned.
  gpt-6 models (GPT-6 Luna) take the reasoning branch too (`max_completion_tokens`, no
  `temperature`); Luna accepts `reasoning_effort: none` and rejects `minimal`, which the
  client turns into `low` for any model not starting with gpt-5. Set `none` explicitly.
- OpenRouter works through the OpenAI client (`base_url: https://openrouter.ai/api/v1`,
  `api_key_env: OPENROUTER_API_KEY`). Our key is on the free tier: 50 requests per day, and
  the free Gemma 4 endpoints returned 429 while `inkling:free` returned 403 (2026-10-24), so free
  models cannot carry a run.
- Each `llm_call` log row has `input_tokens` and `output_tokens` next to the stock
  combined `token_count`, for all three clients, so spend can be computed per call.
  `judge` and `question` write no `llm_call` rows (they pass no logger), so their spend
  is not in the database.
- `question` and `judge` default `--env-file` to `.env` in the current directory, which
  in a run folder does not exist. Pass the repo `.env`.
- The Postgres controller lowercases the schema name, so `--experiment-name Foo` is
  stored and read as `foo`.
- `analyze` and `extract-traces` write `analytics_results_<name>.json` and
  `<name>-agent-llm-traces/` into the current directory. Run them from the run folder.
- `ui` defaults to port 5000, which macOS AirPlay Receiver usually holds; pass `--ui-port`.
- Inspect a run directly: `docker compose exec -T postgres psql -U postgres -d marketplace`,
  then query `<schema>.actions` (message under `data->'request'->'parameters'->'message'`)
  and `<schema>.logs` (`data->'data'->>'type' = 'llm_call'`, with `prompt`, `response`,
  `token_count`, `input_tokens`, `output_tokens`, `duration_ms`). Filter by round with
  `data->'request'->'metadata'->>'round'` on actions, `data->'metadata'->>'round'` on
  `llm_call` logs, and `data->'data'->>'round'` on `price_list`, `channel_message` and
  `refused_payment` logs. `budget_assignment` has no round.
