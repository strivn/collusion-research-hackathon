"""FastAPI server for marketplace analytics and visualization."""

from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..experiments.budgets import reference_price
from ..experiments.judge import load_channel_posts
from ..experiments.round_scores import (
    CustomerRoundRow,
    SellerRoundRow,
    compute_round_tables,
    meets_request,
    score_customer_round,
)
from ..experiments.run_analytics import MarketplaceAnalytics
from ..marketplace.actions import ActionAdapter, Search, SendMessage
from ..marketplace.actions.messaging import OrderProposal
from ..marketplace.shared.models import (
    BusinessAgentProfile,
    CustomerAgentProfile,
    MarketplaceAgentProfileAdapter,
)
from ..platform.database import (
    connect_to_postgresql_database,
    connect_to_sqlite_database,
)
from ..platform.database.base import BaseDatabaseController

# Global database controller (set during lifespan)
_db_controller: BaseDatabaseController | None = None

# Last /api/marketplace-data response, keyed by the action and log row counts it was built from
_marketplace_data_cache: tuple[tuple[int, int], dict[str, Any]] | None = None


async def _load_customers():
    """Load all customer agents from the database."""
    if _db_controller is None:
        raise RuntimeError("Database controller not initialized")

    agent_rows = await _db_controller.agents.get_all()
    customers = []

    for agent_row in agent_rows:
        agent = MarketplaceAgentProfileAdapter.validate_python(
            agent_row.data.model_dump()
        )

        if isinstance(agent, CustomerAgentProfile):
            customer_data = agent.customer
            customer = {
                "id": agent.id,
                "name": customer_data.name,
                "user_request": customer_data.request,
                "menu_features": customer_data.menu_features,
                "amenity_features": customer_data.amenity_features,
            }
            customers.append(customer)

    return customers


async def _load_businesses():
    """Load all business agents from the database."""
    if _db_controller is None:
        raise RuntimeError("Database controller not initialized")

    agent_rows = await _db_controller.agents.get_all()
    businesses = []

    for agent_row in agent_rows:
        agent = MarketplaceAgentProfileAdapter.validate_python(
            agent_row.data.model_dump()
        )

        if isinstance(agent, BusinessAgentProfile):
            business_data = agent.business
            business = {
                "id": agent.id,
                "name": business_data.name,
                "rating": business_data.rating,
                "price_min": min(business_data.menu_features.values())
                if business_data.menu_features
                else 0,
                "price_max": max(business_data.menu_features.values())
                if business_data.menu_features
                else 0,
                "description": business_data.description,
                "menu_features": business_data.menu_features,
                "amenity_features": business_data.amenity_features,
            }
            businesses.append(business)

    return businesses


async def _load_messages():
    """Load all messages from actions in the database."""
    if _db_controller is None:
        raise RuntimeError("Database controller not initialized")

    action_rows = await _db_controller.actions.get_all()
    messages = []

    for action_row in action_rows:
        action_request = action_row.data.request
        action_result = action_row.data.result

        if action_result.is_error:
            continue

        # Actions of stock single-round runs carry no round
        round_number = action_request.metadata.get("round", 1)

        try:
            action = ActionAdapter.validate_python(action_request.parameters)

            if isinstance(action, SendMessage):
                message_content = action.message
                content_dict = message_content.model_dump(mode="json")

                if message_content.type == "text" and "content" in content_dict:
                    content_value = content_dict["content"]
                else:
                    content_value = content_dict

                message = {
                    "id": action_row.id,
                    "to_agent": action.to_agent_id,
                    "from_agent": action.from_agent_id,
                    "type": message_content.type,
                    "content": content_value,
                    "created_at": action.created_at.isoformat(),
                    "round": round_number,
                }
                messages.append(message)
            elif isinstance(action, Search):
                # Extract business IDs from search results
                result_content = action_result.content
                business_ids = []
                if result_content and "businesses" in result_content:
                    business_ids = [
                        b.get("id") for b in result_content["businesses"] if "id" in b
                    ]

                # Create search result message
                search_content = {
                    "type": "search",
                    "query": action.query,
                    "business_ids": business_ids,
                    "total_results": len(business_ids),
                }

                message = {
                    "id": action_row.id,
                    "from_agent": action_row.data.agent_id,
                    "to_agent": None,  # Search doesn't have a specific recipient
                    "type": "search",
                    "content": search_content,
                    "created_at": action_row.created_at.isoformat(),
                    "round": round_number,
                    "business_ids": business_ids,  # Store for thread matching
                }
                messages.append(message)
        except Exception as e:
            print(f"Warning: Failed to parse action {action_row.id}: {e}")
            continue

    return messages


def _create_message_threads(customers, businesses, messages):
    """Create message threads from customers, businesses, and messages.

    A thread is one customer and one business in one round: agents are fresh each round
    and never see earlier rounds' messages.

    Returns:
        tuple: (threads_dict, threads_with_payments_set) where:
            - threads_dict: dict of thread_key -> thread data
            - threads_with_payments_set: set of thread_keys that have payments

    """
    threads = {}
    threads_with_payments = set()
    customer_by_agent_id = {c["id"]: c for c in customers}
    business_by_agent_id = {b["id"]: b for b in businesses}

    for message in messages:
        from_agent = message["from_agent"]
        to_agent = message.get("to_agent")

        # Handle search messages specially - they create threads with all matched businesses
        if message["type"] == "search" and "business_ids" in message:
            customer = customer_by_agent_id.get(from_agent)
            if customer:
                for business_id in message["business_ids"]:
                    business = business_by_agent_id.get(business_id)
                    if business:
                        thread_key = (
                            f"{customer['id']}-{business_id}-r{message['round']}"
                        )

                        if thread_key not in threads:
                            threads[thread_key] = {
                                "participants": {
                                    "customer": customer,
                                    "business": business,
                                },
                                "round": message["round"],
                                "messages": [],
                                "lastMessageTime": message["created_at"],
                                "utility": 0,  # Default utility
                            }

                        # Add search message to each relevant thread
                        thread_message = message.copy()
                        thread_message.pop(
                            "business_ids", None
                        )  # Remove internal field
                        threads[thread_key]["messages"].append(thread_message)
                        threads[thread_key]["lastMessageTime"] = message["created_at"]
        else:
            # Handle regular messages (SendMessage)
            customer = customer_by_agent_id.get(from_agent) or customer_by_agent_id.get(
                to_agent
            )
            business = business_by_agent_id.get(from_agent) or business_by_agent_id.get(
                to_agent
            )

            if customer and business:
                customer_id = customer["id"]
                business_id = business["id"]
                thread_key = f"{customer_id}-{business_id}-r{message['round']}"

                if thread_key not in threads:
                    threads[thread_key] = {
                        "participants": {"customer": customer, "business": business},
                        "round": message["round"],
                        "messages": [],
                        "lastMessageTime": message["created_at"],
                        "utility": 0,  # Default utility
                    }

                threads[thread_key]["messages"].append(message)
                threads[thread_key]["lastMessageTime"] = message["created_at"]

                # Track threads with payments (customer sending payment to business)
                if message["type"] == "payment" and from_agent == customer_id:
                    threads_with_payments.add(thread_key)

    return threads, threads_with_payments


def _thread_customer_share(analytics: MarketplaceAnalytics, thread) -> float:
    """Return the customer share of one thread, scored like one customer-round.

    The customer's value counts once if a payment in the thread meets the request; every
    payment in the thread is subtracted (`round_scores.score_customer_round`).
    """
    customer_id = thread["participants"]["customer"]["id"]
    business_id = thread["participants"]["business"]["id"]
    # A repeated proposal id overwrites the earlier one, as in round_scores
    proposals = {
        m["content"]["id"]: OrderProposal.model_validate(m["content"])
        for m in thread["messages"]
        if m["type"] == "order_proposal" and m["from_agent"] == business_id
    }
    purchases = [
        proposals[m["content"]["proposal_message_id"]]
        for m in thread["messages"]
        if m["type"] == "payment"
        and m["from_agent"] == customer_id
        and m["content"]["proposal_message_id"] in proposals
    ]
    customer = analytics.customer_agents[customer_id].customer
    value = round(2 * reference_price(customer), 2)
    return score_customer_round(
        value,
        [
            (p.total_price, meets_request(analytics, customer_id, business_id, p))
            for p in purchases
        ],
    ).customer_share


def _summarize_round(
    round_number: int | None,
    customers,
    businesses,
    threads,
    customer_rows: list[CustomerRoundRow],
    seller_rows: list[SellerRoundRow],
) -> dict[str, Any]:
    """Return customer, business and marketplace numbers for one round, or all rounds if None.

    Utility is the customer share for customers and revenue for businesses, from the round
    tables; proposal and payment counts come from the threads.
    """
    # Keep only the rows and threads of the round asked for
    if round_number is not None:
        customer_rows = [r for r in customer_rows if r.round == round_number]
        seller_rows = [r for r in seller_rows if r.round == round_number]
        threads = [t for t in threads if t["round"] == round_number]

    # Proposals sent by the business and payments sent by the customer, per thread
    proposals: dict[str, int] = defaultdict(int)
    payments: dict[str, int] = defaultdict(int)
    for thread in threads:
        customer_id = thread["participants"]["customer"]["id"]
        business_id = thread["participants"]["business"]["id"]
        for m in thread["messages"]:
            if m["type"] == "order_proposal" and m["from_agent"] == business_id:
                proposals[customer_id] += 1
                proposals[business_id] += 1
            elif m["type"] == "payment" and m["from_agent"] == customer_id:
                payments[customer_id] += 1
                payments[business_id] += 1

    customer_share: dict[str, float] = defaultdict(float)
    for row in customer_rows:
        customer_share[row.customer] += row.customer_share
    revenue: dict[str, float] = defaultdict(float)
    for row in seller_rows:
        revenue[row.seller] += row.revenue

    return {
        "customer_analytics": {
            c["id"]: {
                "utility": round(customer_share[c["id"]], 2),
                "payments_made": payments[c["id"]],
                "proposals_received": proposals[c["id"]],
            }
            for c in customers
        },
        "business_analytics": {
            b["id"]: {
                "utility": round(revenue[b["id"]], 2),
                "proposals_sent": proposals[b["id"]],
                "payments_received": payments[b["id"]],
            }
            for b in businesses
        },
        "marketplace_summary": {
            "total_utility": round(sum(r.customer_share for r in customer_rows), 2),
            "total_payments": sum(payments[c["id"]] for c in customers),
            "total_proposals": sum(proposals[c["id"]] for c in customers),
            "seller_share": round(sum(r.seller_share for r in customer_rows), 2),
            "lost": round(sum(r.lost for r in customer_rows), 2),
            "sales": sum(r.outcome == "sale" for r in customer_rows),
        },
    }


async def _build_marketplace_data(database: BaseDatabaseController) -> dict[str, Any]:
    """Return messages, per-round threads, seller messages, prices and per-round analytics."""
    customers = await _load_customers()
    businesses = await _load_businesses()
    messages = await _load_messages()
    threads_dict, threads_with_payments = _create_message_threads(
        customers, businesses, messages
    )
    customer_rows, seller_rows = await compute_round_tables(database)
    posts = await load_channel_posts(database)

    # Profiles only, for the request-matching rule of round_scores
    analytics = MarketplaceAnalytics(database)
    await analytics.load_data()
    for thread_key in threads_with_payments:
        thread = threads_dict[thread_key]
        thread["utility"] = _thread_customer_share(analytics, thread)

    # Convert to list and sort by lastMessageTime
    message_threads = list(threads_dict.values())
    message_threads.sort(key=lambda x: x["lastMessageTime"], reverse=True)

    rounds = sorted({row.round for row in customer_rows})
    summary_args = (customers, businesses, message_threads, customer_rows, seller_rows)
    return {
        "messages": messages,
        "messageThreads": message_threads,
        "analytics": _summarize_round(None, *summary_args),
        "analyticsByRound": {
            str(r): _summarize_round(r, *summary_args) for r in rounds
        },
        "rounds": rounds,
        "sellerMessages": [post.model_dump() for post in posts],
        "sellerRounds": [row.model_dump() for row in seller_rows],
    }


def create_analytics_app(
    database_name: str,
    db_type: str = "postgres",
    postgres_host: str = "localhost",
    postgres_port: int = 5432,
    postgres_password: str = "postgres",
) -> FastAPI:
    """Create FastAPI app for analytics with database connection.

    Args:
        database_name: PostgreSQL schema name or path to SQLite database file
        db_type: Type of database ("sqlite" or "postgres")
        postgres_host: PostgreSQL host (only used if db_type is "postgres")
        postgres_port: PostgreSQL port (only used if db_type is "postgres")
        postgres_password: PostgreSQL password (only used if db_type is "postgres")

    Returns:
        Configured FastAPI application

    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Manage database connection lifecycle."""
        global _db_controller

        if db_type == "sqlite":
            print("Connecting to SQLite database...", flush=True)
            print(f"Database path: {database_name}", flush=True)

            async with connect_to_sqlite_database(database_path=database_name) as db:
                _db_controller = db
                print("Database connection established", flush=True)
                print("UI API ready", flush=True)
                yield

            print("Database connection closed", flush=True)
        elif db_type == "postgres":
            print("Connecting to PostgreSQL database...", flush=True)
            print(f"Host: {postgres_host}:{postgres_port}", flush=True)
            print(f"Schema: {database_name}", flush=True)

            async with connect_to_postgresql_database(
                schema=database_name,
                host=postgres_host,
                port=postgres_port,
                password=postgres_password,
                mode="existing",
            ) as db:
                _db_controller = db
                print("Database connection established", flush=True)
                print("UI API ready", flush=True)
                yield

            print("Database connection closed", flush=True)
        else:
            raise ValueError(
                f"Unsupported database type: {db_type}. Must be 'sqlite' or 'postgres'."
            )

    app = FastAPI(
        title="Marketplace UI API",
        version="2.0.0",
        description="UI API for marketplace simulation data",
        lifespan=lifespan,
    )

    # Enable CORS for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes (all under /api prefix)
    @app.get("/api")
    def api_index():
        """Get basic info about the API."""
        return {
            "name": "Marketplace UI API",
            "version": "2.0.0",
            "database": "SQLite" if db_type == "sqlite" else "PostgreSQL",
            "endpoints": [
                "/api/customers",
                "/api/businesses",
                "/api/marketplace-data",
                "/api/health",
            ],
        }

    @app.get("/api/customers")
    async def get_customers():
        """Get all customers."""
        try:
            return await _load_customers()
        except Exception as e:
            print(f"Error: {e}")
            return {"error": "Unable to load customers"}

    @app.get("/api/businesses")
    async def get_businesses():
        """Get all businesses."""
        try:
            return await _load_businesses()
        except Exception as e:
            print(f"Error: {e}")
            return {"error": "Unable to load businesses"}

    @app.get("/api/marketplace-data")
    async def get_marketplace_data():
        """Get messages, threads, seller messages, prices and analytics, cached until rows change."""
        global _marketplace_data_cache
        try:
            if _db_controller is None:
                raise RuntimeError("Database controller not initialized")

            # Rows are only ever added, so unchanged counts mean unchanged data
            key = (
                await _db_controller.actions.count(),
                await _db_controller.logs.count(),
            )
            if _marketplace_data_cache is None or _marketplace_data_cache[0] != key:
                _marketplace_data_cache = (
                    key,
                    await _build_marketplace_data(_db_controller),
                )
            return _marketplace_data_cache[1]
        except Exception as e:
            print(f"Error: {e}")
            return {"error": "Unable to load messages"}

    @app.get("/api/health")
    def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "database": "connected" if _db_controller else "disconnected",
        }

    # Mount static files at root (for UI)
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


def run_ui_server(
    database_name: str,
    db_type: str = "postgres",
    postgres_host: str = "localhost",
    postgres_port: int = 5432,
    postgres_password: str = "postgres",
    ui_port: int = 5000,
    ui_host: str = "localhost",
):
    """Run the UI server.

    Args:
        database_name: PostgreSQL schema name or path to SQLite database file
        db_type: Type of database ("sqlite" or "postgres")
        postgres_host: PostgreSQL host (only used if db_type is "postgres")
        postgres_port: PostgreSQL port (only used if db_type is "postgres")
        postgres_password: PostgreSQL password (only used if db_type is "postgres")
        ui_port: Port for UI server
        ui_host: Host for UI server

    """
    import uvicorn

    print("Starting Marketplace UI Server...", flush=True)
    if db_type == "sqlite":
        print(f"Database: {database_name}", flush=True)
    else:
        print(f"Schema: {database_name}", flush=True)
    print(f"Server will be available at: http://{ui_host}:{ui_port}", flush=True)

    app = create_analytics_app(
        database_name=database_name,
        db_type=db_type,
        postgres_host=postgres_host,
        postgres_port=postgres_port,
        postgres_password=postgres_password,
    )

    uvicorn.run(
        app,
        host=ui_host,
        port=ui_port,
        log_level="info",
    )
