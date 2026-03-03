"""
Google ADK FastAPI Service with PyVegas Integration.
Dynamic Agent Gateway Implementation.

UPDATED: Multi-agent sub-agent architecture with:
- Persistent session management (conversation memory)
- Sequential troubleshooting with fix tracking
- Escalation logic
- LlmProcessRequest/Response contract alignment
"""

import os
import yaml
import asyncio
import json
import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from pyvegas.serve.google_adk.factory import VegasAdkApp

from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.slack_agent.agent import build_dynamic_agent


# ================================================================
# REQUEST / RESPONSE MODELS (matching api_contract.yml)
# ================================================================

class LlmProcessRequest(BaseModel):
    sessionId: str
    channelId: str
    userId: str
    appId: str
    userMessage: str


class LlmProcessResponse(BaseModel):
    sessionId: str
    finalResponse: str
    status: str


# ================================================================
# GLOBAL STATE
# ================================================================

GLOBAL_STATE = {
    "mcp_tools": [],
    "registry": {},
}

# Persistent session service — shared across ALL requests
# TODO: Replace with DatabaseSessionService (PostgreSQL) for UAT/PROD
# TODO: Replace with RedisSessionService for local dev
_SESSION_SERVICE = InMemorySessionService()

# Cache runners per app_id to avoid rebuilding on every request
_RUNNER_CACHE: dict[str, Runner] = {}

_INIT_TASK = None


# ================================================================
# REGISTRY & INITIALIZATION
# ================================================================

def load_registry():
    current_dir = Path(__file__).parent
    config_path = current_dir.parent / "config" / "agents_registry.yaml"

    print(f"Attempting to load registry from: {config_path.resolve()}")

    if not config_path.exists():
        raise FileNotFoundError(f"CRITICAL: Registry file not found at {config_path.resolve()}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


async def _perform_initialization():
    """The actual heavy lifting. Only runs exactly once."""
    print("[Cold Start] Fetching Registry and MCP Tools...")

    # 1. Load Registry
    GLOBAL_STATE["registry"] = load_registry()

    # 2. Fetch Tools
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")
    toolset = McpToolset(connection_params=StreamableHTTPServerParams(url=mcp_url))

    GLOBAL_STATE["mcp_tools"] = await toolset.get_tools()
    print(f"Successfully loaded {len(GLOBAL_STATE['mcp_tools'])} tools into global memory.")


async def ensure_initialized():
    """
    Lock-free guarantee that initialization happens once.
    Concurrent requests will await the exact same task.
    """
    global _INIT_TASK

    if _INIT_TASK is None:
        _INIT_TASK = asyncio.create_task(_perform_initialization())

    try:
        await _INIT_TASK
    except Exception as e:
        _INIT_TASK = None
        print(f"Initialization failed: {e}")
        raise HTTPException(status_code=503, detail="Gateway initialization failed. MCP Server may be down.")


# ================================================================
# APP INITIALIZATION
# ================================================================

_adk_app = VegasAdkApp(
    agents_dir="agents",
    web=False  # Disable standard UI as we are using webhook backend
)

app = _adk_app.create_app()

pyvegas_lifespan = app.router.lifespan_context


@asynccontextmanager
async def combined_lifespan(fastapi_app):
    """Executes our custom logic, then hands control back to PyVegas."""

    GLOBAL_STATE["registry"] = load_registry()
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")
    toolset = McpToolset(connection_params=StreamableHTTPServerParams(url=mcp_url))

    try:
        GLOBAL_STATE["mcp_tools"] = await toolset.get_tools()
        print(f"Successfully loaded {len(GLOBAL_STATE['mcp_tools'])} tools.")
    except Exception as e:
        print(f"Failed to load MCP tools. Error: {e}")

    if pyvegas_lifespan:
        async with pyvegas_lifespan(fastapi_app) as pyvegas_state:
            yield pyvegas_state
    else:
        yield


app.router.lifespan_context = combined_lifespan


# ================================================================
# HELPER: Build or retrieve a Runner for an app_id
# ================================================================

def _get_or_build_runner(app_id: str) -> Runner:
    """
    Build the agent + runner for a given app_id, or return
    cached version. The agent is built with filtered MCP tools
    based on allowed_catalogs from the registry.
    """
    if app_id in _RUNNER_CACHE:
        return _RUNNER_CACHE[app_id]

    registry = GLOBAL_STATE["registry"]
    all_tools = GLOBAL_STATE["mcp_tools"]

    app_config = registry.get("agents", {}).get(app_id)
    if not app_config:
        raise HTTPException(status_code=404, detail=f"Invalid App ID: {app_id}")

    # Filter tools by allowed catalogs
    allowed_catalogs = app_config.get("allowed_catalogs", [])
    allowed_prefixes = [
        registry["catalogs"][cat]
        for cat in allowed_catalogs
        if cat in registry.get("catalogs", {})
    ]

    filtered_tools = [
        t for t in all_tools
        if t.name.startswith(tuple(allowed_prefixes))
    ]

    agent = build_dynamic_agent(app_id, filtered_tools)

    runner = Runner(
        agent=agent,
        app_name=app_id,
        session_service=_SESSION_SERVICE,
    )

    _RUNNER_CACHE[app_id] = runner
    print(f"Built and cached runner for app_id={app_id} with {len(filtered_tools)} tools.")
    return runner


# ================================================================
# HELPER: Get or create a session with state
# ================================================================

async def _get_or_create_session(
    app_id: str,
    user_id: str,
    session_id: str,
    channel_id: str,
    user_message: str,
):
    """
    Resume an existing session (conversation memory) or create
    a new one with initialized state for the agent.

    Session state tracks:
    - app_id, user_id, channel_id: context identifiers
    - user_message: current turn's message
    - detected_intent: classified intent (persists across turns)
    - api_results: product API response data
    - kb_results: knowledge base search results
    - fixes_attempted: JSON list of all fixes suggested so far
    - current_step_index: which troubleshooting step we're on
    - ticket_id: AYS/ServiceNow ticket ID if escalated
    - final_response: last response given to user
    """
    # Try to get existing session
    existing = await _SESSION_SERVICE.get_session(
        app_name=app_id,
        user_id=user_id,
        session_id=session_id,
    )

    if existing:
        # Follow-up turn: update the message, preserve all other state
        existing.state["user_message"] = user_message
        return existing

    # New conversation: create session with initialized state
    session = await _SESSION_SERVICE.create_session(
        app_name=app_id,
        user_id=user_id,
        session_id=session_id,
        state={
            # Context
            "app_id": app_id,
            "user_id": user_id,
            "channel_id": channel_id,
            "session_id": session_id,
            "user_message": user_message,
            # Troubleshooting tracking
            "detected_intent": "{}",
            "api_results": "{}",
            "kb_results": "",
            "fixes_attempted": "[]",
            "current_step_index": "1",
            # Escalation
            "ticket_id": "",
            # Response
            "final_response": "",
            "original_message": user_message,
        },
    )
    return session


# ================================================================
# ENDPOINT: LLM Process (api_contract.yml aligned)
# ================================================================

@app.post("/process", response_model=LlmProcessResponse)
async def process_message(req: LlmProcessRequest):
    """
    Main entry point matching api_contract.yml.

    OUTBOUND: Java calls this with LlmProcessRequest.
    Returns LlmProcessResponse with sessionId, finalResponse, status.
    """
    start_total_time = time.perf_counter()

    try:
        await ensure_initialized()
    except HTTPException as e:
        return LlmProcessResponse(
            sessionId=req.sessionId,
            finalResponse="Service is temporarily unavailable. Please try again.",
            status="error",
        )

    # 1. Get or build the runner for this app
    try:
        runner = _get_or_build_runner(req.appId)
    except HTTPException as e:
        return LlmProcessResponse(
            sessionId=req.sessionId,
            finalResponse=f"Configuration error: {e.detail}",
            status="error",
        )

    # 2. Get or create session (preserves conversation history)
    session = await _get_or_create_session(
        app_id=req.appId,
        user_id=req.userId,
        session_id=req.sessionId,
        channel_id=req.channelId,
        user_message=req.userMessage,
    )

    # 3. Build the user message content
    content = types.Content(
        role="user",
        parts=[types.Part(text=req.userMessage)]
    )

    # 4. Run the agent
    final_response_text = "No response generated by the LLM."

    try:
        print(f"Executing agent for {req.appId} (Session: {req.sessionId})...")

        start_run_time = time.perf_counter()

        async for event in runner.run_async(
            user_id=req.userId,
            session_id=req.sessionId,
            new_message=content,
        ):
            if event.is_final_response():
                final_response_text = event.content.parts[0].text

        run_time = time.perf_counter() - start_run_time
        total_time = time.perf_counter() - start_total_time

        print(
            f"Metrics | App: {req.appId} | Session: {req.sessionId} | "
            f"LLM Run: {run_time:.4f}s | Total: {total_time:.4f}s"
        )

        return LlmProcessResponse(
            sessionId=req.sessionId,
            finalResponse=final_response_text,
            status="success",
        )

    except Exception as e:
        print(f"Agent execution failed: {e}")
        return LlmProcessResponse(
            sessionId=req.sessionId,
            finalResponse=f"Failed to process request: {str(e)}",
            status="error",
        )


# ================================================================
# ENDPOINT: Legacy Slack webhook (backward compatibility)
# ================================================================

@app.post("/slack/{app_id}/events")
@app.post("/slack/{app_id}/events/")
async def slack_webhook(app_id: str, request: Request):
    """
    Legacy Slack webhook endpoint.
    Converts Slack event payload into LlmProcessRequest
    and delegates to process_message.
    """
    try:
        await ensure_initialized()
    except HTTPException as e:
        return {"status": "error", "message": e.detail}

    payload = await request.json()

    event = payload.get("event", {})
    user_query = event.get("text", "")
    user_id = event.get("user", "unknown_user")
    channel_id = event.get("channel", "unknown_channel")

    # Build session_id from context (same thread = same session)
    session_id = f"{app_id}_{channel_id}_{user_id}"

    # Delegate to the contract-aligned endpoint
    req = LlmProcessRequest(
        sessionId=session_id,
        channelId=channel_id,
        userId=user_id,
        appId=app_id,
        userMessage=user_query,
    )

    response = await process_message(req)

    return {
        "status": response.status,
        "response": response.finalResponse,
    }


# ================================================================
# DIRECT TESTING
# ================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1556)
