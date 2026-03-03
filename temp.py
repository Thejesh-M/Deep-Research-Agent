Great questions. Let me address both.

**Runner Cache — The Problem With Your Current Setup:**

Right now in your existing code, every single request does this:

```python
# Current: lines 140-161 of your adk_service.py
filtered_tools = [t for t in all_tools if t.name.startswith(...)]
agent = build_dynamic_agent(app_id, filtered_tools)  # rebuilds agent
session_service = InMemorySessionService()             # new sessions
runner = Runner(agent=agent, ...)                      # new runner
```

That's expensive. For every Slack message, you're rebuilding the entire agent hierarchy (Coordinator + 7 sub-agents), re-filtering tools, and creating a new Runner. For 157 channels with potentially hundreds of messages per hour, this adds significant latency to every request.

But your concern is valid — you load agents **dynamically per app_id** because each app has different `allowed_catalogs` and therefore different tool sets. Artemis gets `artemis_user`, `artemis_business`. Salesforce gets `sf_user`, `sf_account`. The agent configuration is app-specific.

So the question is: **what actually changes between requests for the same app_id?**

The answer is: **nothing**. The registry YAML doesn't change between requests. The MCP tools don't change between requests. The agent instructions are the same. The only thing that changes is the session state (user message, conversation history), and that's handled by the session service, not the runner.

So caching runners per `app_id` is safe — the agent structure is identical for every request to the same app. What changes is the **session**, not the **agent**.

The cache invalidation concern is real though. Here's when the cache SHOULD be invalidated:

```python
# When does the cache go stale?
# 1. Registry YAML is updated (new app added, catalogs changed)
# 2. MCP server adds/removes tools
# 3. Agent prompts are updated (if externalized)

# Solution: TTL-based cache with manual refresh

import time

_RUNNER_CACHE: dict[str, tuple[Runner, float]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

def _get_or_build_runner(app_id: str) -> Runner:
    now = time.time()
    
    if app_id in _RUNNER_CACHE:
        runner, created_at = _RUNNER_CACHE[app_id]
        if now - created_at < CACHE_TTL_SECONDS:
            return runner
        # Expired — fall through to rebuild
        print(f"Runner cache expired for {app_id}, rebuilding.")
    
    # ... build agent and runner ...
    
    _RUNNER_CACHE[app_id] = (runner, now)
    return runner

# Plus a manual refresh endpoint for deployments
@app.post("/admin/reload")
async def reload_agents():
    """Call this after registry or MCP changes."""
    _RUNNER_CACHE.clear()
    GLOBAL_STATE["registry"] = load_registry()
    # Re-fetch MCP tools
    mcp_url = os.getenv("MCP_SERVER_URL", "...")
    toolset = McpToolset(connection_params=...)
    GLOBAL_STATE["mcp_tools"] = await toolset.get_tools()
    return {"status": "reloaded", "tools": len(GLOBAL_STATE["mcp_tools"])}
```

**Now, Semantic Cache — This Is the Bigger Win:**

The runner cache saves you from rebuilding agents. But the real latency killer is the **LLM calls themselves**. Every troubleshooting session makes multiple LLM calls (Coordinator reasoning, IntentAgent classification, TroubleshootAgent KB search, ResponseAgent synthesis). If 50 users in the same week ask "I can't access Artemis" and they all have status "inactive," you're making the same LLM calls with roughly the same inputs and getting roughly the same outputs.

Semantic cache sits **between the agent and the LLM**, caching responses for semantically similar inputs.

```
User: "I am unable to access Artemis"
User: "Can't login to Artemis"
User: "Artemis is not working for me"
User: "artemis access denied"

All of these should hit the SAME cache entry
because they're semantically identical.
```

Here's where to place it in your architecture:

```
Slack Message
    │
    ├─ 1. Check semantic cache (BEFORE any agent runs)
    │     Hit? → Return cached response immediately
    │     Miss? → Continue to agent
    │
    ├─ 2. Agent runs (Intent → API → KB → Response)
    │
    └─ 3. Store in semantic cache (AFTER agent completes)
          Key: embedding of (app_id + intent + api_results)
          Value: final response + KB steps
          TTL: configurable per intent
```

But here's the nuance — you can't just cache the **final response** blindly. Your troubleshooting is **user-specific** (the API results differ per user — one user might be "inactive," another "suspended"). So the cache needs to be **layered**:

```python
# semantic_cache.py

import numpy as np
import json
import time
import hashlib
from typing import Optional

class SemanticCache:
    """
    Two-layer semantic cache for V-Support:
    
    Layer 1 (L1): Intent-level cache
      - Caches the intent classification for similar messages
      - "Can't access Artemis" ≈ "Artemis not working" → same intent
      - Very high hit rate, saves the IntentAgent LLM call
    
    Layer 2 (L2): Resolution-level cache  
      - Caches the KB resolution for (intent + api_status) combos
      - "User not active" + status="inactive" → same KB steps
      - User-specific API results make the key unique
      - Moderate hit rate, saves KB search + response synthesis
    """
    
    def __init__(self, embedding_client, redis_client):
        self.embedder = embedding_client  # Vertex AI / Gemini embeddings
        self.redis = redis_client
        self.similarity_threshold = 0.92  # tune this
    
    # ── Layer 1: Intent Cache ─────────────────────────────
    
    async def get_cached_intent(
        self, app_id: str, user_message: str
    ) -> Optional[dict]:
        """
        Check if we've seen a semantically similar message 
        for this app before.
        
        Returns: {"intent": "User not active", "confidence": 0.95}
        or None on miss.
        """
        embedding = await self._get_embedding(user_message)
        cache_key_prefix = f"intent_cache:{app_id}"
        
        # Search for similar embeddings in Redis
        match = await self._vector_search(
            cache_key_prefix, embedding
        )
        
        if match and match["score"] >= self.similarity_threshold:
            print(f"[Cache L1 HIT] Intent for '{user_message[:40]}...'")
            return json.loads(match["value"])
        
        return None
    
    async def cache_intent(
        self, app_id: str, user_message: str, intent_result: dict,
        ttl: int = 3600  # 1 hour default
    ):
        """Store intent classification for future similar messages."""
        embedding = await self._get_embedding(user_message)
        cache_key = f"intent_cache:{app_id}:{self._hash(user_message)}"
        
        await self.redis.set(
            cache_key,
            json.dumps({
                "embedding": embedding.tolist(),
                "value": json.dumps(intent_result),
                "created_at": time.time(),
            }),
            ex=ttl,
        )
    
    # ── Layer 2: Resolution Cache ─────────────────────────
    
    async def get_cached_resolution(
        self, app_id: str, intent: str, api_status: str,
        step_index: int
    ) -> Optional[dict]:
        """
        Check if we have a cached resolution for this 
        (intent + api_status + step) combination.
        
        This is a DETERMINISTIC key, not semantic — because
        once we know the intent and the user's API status,
        the KB resolution is the same for everyone.
        
        Returns: {"kb_result": "...", "response": "Step 2: ..."}
        or None.
        """
        cache_key = (
            f"resolution:{app_id}:{intent}:"
            f"{api_status}:step_{step_index}"
        )
        
        cached = await self.redis.get(cache_key)
        if cached:
            print(f"[Cache L2 HIT] Resolution for {intent} step {step_index}")
            return json.loads(cached)
        
        return None
    
    async def cache_resolution(
        self, app_id: str, intent: str, api_status: str,
        step_index: int, resolution: dict,
        ttl: int = 86400  # 24 hours — KB doesn't change often
    ):
        """Store resolution for this intent + status + step combo."""
        cache_key = (
            f"resolution:{app_id}:{intent}:"
            f"{api_status}:step_{step_index}"
        )
        await self.redis.set(cache_key, json.dumps(resolution), ex=ttl)
    
    # ── Layer 0: Exact Match (fast path) ──────────────────
    
    async def get_exact_match(
        self, app_id: str, user_message: str
    ) -> Optional[str]:
        """
        Fastest path — exact message match.
        Catches repeated messages like "No" or "Yes" or 
        copy-pasted error messages.
        """
        cache_key = f"exact:{app_id}:{self._hash(user_message)}"
        return await self.redis.get(cache_key)
    
    # ── Helpers ───────────────────────────────────────────
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding from Vertex AI / Gemini."""
        # Use Gemini's text-embedding model
        result = await self.embedder.embed_content(
            model="text-embedding-004",
            content=text,
        )
        return np.array(result.embedding)
    
    async def _vector_search(
        self, prefix: str, query_embedding: np.ndarray
    ) -> Optional[dict]:
        """
        Search Redis for the most similar cached embedding.
        
        For production: Use Redis Vector Similarity Search (VSS)
        or Vertex AI Vector Search for better performance.
        """
        # Simplified brute-force for illustration
        # In production, use Redis VECTOR index or pgvector
        keys = await self.redis.keys(f"{prefix}:*")
        best_match = None
        best_score = 0.0
        
        for key in keys:
            data = json.loads(await self.redis.get(key))
            stored = np.array(data["embedding"])
            score = float(np.dot(query_embedding, stored) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored)
            ))
            if score > best_score:
                best_score = score
                best_match = {"score": score, "value": data["value"]}
        
        return best_match
    
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
```

**Now, integrating this into `adk_service.py`:**

```python
# In adk_service.py — updated process_message

_SEMANTIC_CACHE = SemanticCache(embedding_client, redis_client)

@app.post("/process", response_model=LlmProcessResponse)
async def process_message(req: LlmProcessRequest):
    
    # ... initialization ...
    
    # ── Layer 0: Exact match (sub-millisecond) ────────
    exact = await _SEMANTIC_CACHE.get_exact_match(
        req.appId, req.userMessage
    )
    if exact:
        return LlmProcessResponse(
            sessionId=req.sessionId,
            finalResponse=exact, status="success"
        )
    
    # ── Layer 1: Semantic intent cache ────────────────
    # Skip the IntentAgent LLM call if we've seen similar
    session = await _get_or_create_session(...)
    
    cached_intent = await _SEMANTIC_CACHE.get_cached_intent(
        req.appId, req.userMessage
    )
    if cached_intent:
        # Inject directly into session state
        session.state["detected_intent"] = json.dumps(cached_intent)
        # Agent will skip IntentAgent, Coordinator sees 
        # detected_intent already populated
    
    # ── Layer 2: Resolution cache ─────────────────────
    # After API results are known (either from cache or API call)
    # Check if we have a cached resolution for this combo
    if cached_intent:
        intent_name = cached_intent.get("intent", "")
        api_status = session.state.get("api_results", "")
        step = int(session.state.get("current_step_index", "1"))
        
        cached_resolution = await _SEMANTIC_CACHE.get_cached_resolution(
            req.appId, intent_name, 
            _extract_status(api_status), step
        )
        if cached_resolution:
            # Return directly — no LLM calls at all!
            return LlmProcessResponse(
                sessionId=req.sessionId,
                finalResponse=cached_resolution["response"],
                status="success",
            )
    
    # ── Cache miss: Run full agent pipeline ───────────
    runner = _get_or_build_runner(req.appId)
    
    final_response_text = ""
    async for event in runner.run_async(...):
        if event.is_final_response():
            final_response_text = event.content.parts[0].text
    
    # ── Store results in cache for future hits ────────
    # Cache the intent classification
    detected = session.state.get("detected_intent", "{}")
    await _SEMANTIC_CACHE.cache_intent(
        req.appId, req.userMessage, json.loads(detected)
    )
    
    # Cache the resolution
    await _SEMANTIC_CACHE.cache_resolution(
        req.appId,
        json.loads(detected).get("intent", ""),
        _extract_status(session.state.get("api_results", "")),
        int(session.state.get("current_step_index", "1")),
        {"response": final_response_text,
         "kb_result": session.state.get("kb_results", "")},
    )
    
    return LlmProcessResponse(
        sessionId=req.sessionId,
        finalResponse=final_response_text,
        status="success",
    )
```

**Here's what each cache layer saves you:**

| Layer | What it caches | Hit rate | Latency saved | When to use |
|---|---|---|---|---|
| **L0: Exact match** | Identical messages | High for "Yes"/"No" | ~100% (no LLM calls) | Confirmation responses, repeated errors |
| **L1: Semantic intent** | Similar messages → same intent | Very high | ~30% (skips IntentAgent) | "Can't access X" ≈ "X not working" |
| **L2: Resolution** | Same intent + same API status + same step | Moderate | ~80% (skips KB + synthesis) | 50 users all inactive on Artemis |

**But there's a critical caveat for conversational flows:**

You should **NOT cache** when:

- The response is user-specific and depends on API results that haven't been fetched yet for this user
- The conversation is mid-troubleshooting (step 2, step 3) — the cache key needs to include `step_index` to avoid returning step 1's answer for step 3
- The user is in an escalation flow — ticket creation must always be live
- The API results have changed (user was reactivated since last cache)

```python
# Cache bypass conditions
def _should_bypass_cache(session_state: dict) -> bool:
    """Don't use resolution cache in these cases."""
    return (
        # Mid-escalation — always live
        session_state.get("ticket_id", "") != ""
        # User explicitly said "no" — need fresh KB search
        or session_state.get("user_message", "").lower().strip() in 
           ["no", "nope", "didn't work", "still not working"]
    )
```

**For your Redis setup**, since you mentioned Redis for local and PostgreSQL for prod:

```python
# Session service swap (one line change)
import os

if os.getenv("ENV") == "production":
    from your_custom_session import PostgresSessionService
    _SESSION_SERVICE = PostgresSessionService(
        connection_string=os.getenv("PG_CONNECTION_STRING")
    )
elif os.getenv("ENV") == "uat":
    from your_custom_session import PostgresSessionService
    _SESSION_SERVICE = PostgresSessionService(
        connection_string=os.getenv("PG_CONNECTION_STRING")
    )
else:
    # Local dev — Redis for cache, InMemory for sessions
    _SESSION_SERVICE = InMemorySessionService()

# Semantic cache always uses Redis (even in prod)
import redis.asyncio as redis
_REDIS = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
_SEMANTIC_CACHE = SemanticCache(embedding_client, _REDIS)
```

Want me to build out the full `semantic_cache.py` with Redis Vector Search integration, or dive into the PostgreSQL session service implementation?
