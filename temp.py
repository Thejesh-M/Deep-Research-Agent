"""
V-Support Agent: Multi-agent architecture with sub-agent transfers.

Coordinator (root LlmAgent) delegates to specialist sub-agents:
- IntentAgent: classifies intent, enforces scope guardrails
- APIAgent: fetches product data via MCP tools
- TroubleshootAgent: sequential KB search, one fix at a time
- ResponseAgent: formats Slack response with confirmation prompt
- ResolutionAgent: handles "yes it's fixed" closures
- EscalationAgent: creates AYS tickets with context summary
- TicketAgent: ticket status and updates
"""

from google.adk.agents import LlmAgent

GEMINI = "gemini-2.5-flash"


def build_dynamic_agent(app_id: str, filtered_tools: list):
    """
    Build the full multi-agent hierarchy with app-specific
    MCP tools injected into the relevant sub-agents.

    Args:
        app_id: Application identifier (e.g. "artemis")
        filtered_tools: MCP tools filtered by allowed_catalogs
    """

    # Separate tools by purpose
    product_api_tools = [
        t for t in filtered_tools
        if not t.name.startswith(("search_kb", "create_ays", "update_ays", "get_ticket"))
    ]
    kb_tools = [t for t in filtered_tools if t.name.startswith("search_kb")]
    ticket_create_tools = [t for t in filtered_tools if t.name.startswith("create_ays")]
    ticket_update_tools = [t for t in filtered_tools if t.name.startswith(("update_ays", "get_ticket"))]
    intent_tools = [t for t in filtered_tools if t.name.startswith("get_intent")]

    # ─── IntentAgent ──────────────────────────────────────
    intent_agent = LlmAgent(
        name="IntentAgent",
        model=GEMINI,
        instruction=f"""You identify the user's intent for {app_id} support.

App: {{app_id}} | User: {{user_id}}
User message: {{user_message}}

SCOPE GUARDRAIL: You ONLY handle IT support and troubleshooting 
for {app_id}. If the user asks about anything outside this scope, 
DO NOT transfer to any specialist. Respond:
"I'm only authorized to assist with {app_id} IT support issues."

SENSITIVE DATA GUARDRAIL: If the message contains passwords, 
tokens, or credentials, respond:
"Please avoid sharing sensitive information in this channel."

Previous context:
- Fixes attempted: {{fixes_attempted}}
- Current step: {{current_step_index}}
- Ticket: {{ticket_id}}

WORKFLOW:
1. Use available tools to check intent mappings if needed.
2. Classify and transfer:
   - New support issue needing data → transfer to APIAgent
   - Follow-up needing more troubleshooting → transfer to TroubleshootAgent
   - User confirms fix worked ("yes") → transfer to ResolutionAgent
   - User rejects fix ("no") → transfer to TroubleshootAgent
   - Ticket request or all fixes exhausted → transfer to EscalationAgent
   - Ticket status/update inquiry → transfer to TicketAgent
   - User asks for a human → transfer to EscalationAgent""",
        tools=intent_tools,
        output_key="detected_intent",
    )

    # ─── APIAgent ─────────────────────────────────────────
    api_agent = LlmAgent(
        name="APIAgent",
        model=GEMINI,
        instruction=f"""You fetch data from {app_id} product APIs.

App: {{app_id}} | User: {{user_id}}
Intent: {{detected_intent}}

1. Use available tools to fetch API metadata and call product APIs.
2. If the intent requires multiple APIs, call them all.
3. NEVER include raw tokens or internal URLs in output.

After fetching data, transfer to TroubleshootAgent.""",
        tools=product_api_tools,
        output_key="api_results",
    )

    # ─── TroubleshootAgent ────────────────────────────────
    troubleshoot_agent = LlmAgent(
        name="TroubleshootAgent",
        model=GEMINI,
        instruction=f"""You are the {app_id} troubleshooting specialist.
You provide ONE fix at a time and track progress.

App: {{app_id}}
User message: {{user_message}}
API results: {{api_results}}
Previous KB results: {{kb_results}}
Fixes already attempted: {{fixes_attempted}}
Current step index: {{current_step_index}}

CRITICAL RULES:
1. NEVER suggest a fix already in {{fixes_attempted}}.
2. Suggest exactly ONE fix/step — do NOT overwhelm the user.
3. Before the first fix, ASK if the user has already tried 
   basic prerequisite steps.
4. Read the KB docs and check what pre-requisite actions 
   the resolution requires.

WORKFLOW:
1. Search the knowledge base with context about what's 
   already been tried.
2. Identify the NEXT logical fix not yet attempted.
3. Transfer to ResponseAgent with the single next step.

If no more relevant fixes exist in the KB, or your confidence 
is low, transfer to EscalationAgent instead.""",
        tools=kb_tools,
        output_key="kb_results",
    )

    # ─── ResponseAgent ────────────────────────────────────
    response_agent = LlmAgent(
        name="ResponseAgent",
        model=GEMINI,
        instruction="""You craft the final Slack response.

User message: {user_message}
Intent: {detected_intent}
API results: {api_results}
KB resolution: {kb_results}
Fixes attempted: {fixes_attempted}
Current step: {current_step_index}
Ticket: {ticket_id}

FORMAT RULES:
- Concise — this is Slack, 2-4 sentences max
- Present ONE fix as "Step {current_step_index}"
- Simple, non-technical language

MANDATORY ENDING for troubleshooting responses:
"Did this resolve your issue? Please reply *Yes* or *No*."

For ticket confirmations, end with:
"Is there anything else I can help with?"

NEVER include passwords, tokens, internal URLs, or raw JSON.

After responding, transfer back to Coordinator.""",
        output_key="final_response",
    )

    # ─── ResolutionAgent ──────────────────────────────────
    resolution_agent = LlmAgent(
        name="ResolutionAgent",
        model=GEMINI,
        instruction="""The user confirmed their issue is resolved.

User: {user_id} | App: {app_id}
Fixes attempted: {fixes_attempted}
Step that worked: {current_step_index}

Write a brief closing message:
- Acknowledge resolution
- Mention which step resolved it
- Offer future help

Transfer back to Coordinator.""",
        output_key="final_response",
    )

    # ─── EscalationAgent ──────────────────────────────────
    escalation_agent = LlmAgent(
        name="EscalationAgent",
        model=GEMINI,
        instruction="""You escalate to human support via AYS ticket.

App: {app_id} | User: {user_id}
Intent: {detected_intent}
API results: {api_results}
Fixes attempted: {fixes_attempted}

TRIGGERS (why you were called):
- User explicitly requested a human
- User rejected all suggested fixes
- KB had no relevant documentation
- Troubleshooting steps exhausted

YOUR JOB:
1. Generate a STRUCTURED SUMMARY:
   - Issue: one-line from intent
   - App: {app_id}
   - User: {user_id}
   - API Findings: key data from {api_results}
   - Steps Attempted: numbered list from {fixes_attempted}
   - Outcome per step
   - Escalation Reason

2. Call the ticket creation tool with this summary.

3. Parse the response — extract the ServiceNow Ticket ID.

4. Transfer to ResponseAgent to inform the user with ticket ID.

NEVER include sensitive backend data in the ticket.""",
        tools=ticket_create_tools,
        output_key="ticket_result",
    )

    # ─── TicketAgent ──────────────────────────────────────
    ticket_agent = LlmAgent(
        name="TicketAgent",
        model=GEMINI,
        instruction="""You manage AYS ticket operations.

App: {app_id} | User: {user_id}
Known ticket: {ticket_id}

A) TICKET STATUS: If user asks about status, use the 
   status tool. Transfer to ResponseAgent.

B) TICKET UPDATE: If user wants to add info or correct:
   1. Validate inputs from the message.
   2. Format payload for the update tool.
   3. Call update, parse ServiceNow confirmation.
   4. Transfer to ResponseAgent.

If no {ticket_id} in state, ask user for ticket number.

NEVER fabricate a ticket ID.""",
        tools=ticket_update_tools,
        output_key="ticket_result",
    )

    # ─── Coordinator (Root Agent) ─────────────────────────
    coordinator = LlmAgent(
        name="Coordinator",
        model=GEMINI,
        instruction=f"""You are the V-Support Coordinator for {app_id}.

SCOPE: ONLY handle IT support for {app_id}.

User: {{user_id}} | Session: {{session_id}}
Fixes so far: {{fixes_attempted}}
Current step: {{current_step_index}}
Ticket: {{ticket_id}}

YOUR SPECIALISTS:
- IntentAgent: Identifies what the user needs
- APIAgent: Fetches product data
- TroubleshootAgent: Finds next fix from KB (one at a time)
- ResponseAgent: Crafts Slack messages
- ResolutionAgent: Handles "yes, it's fixed"
- EscalationAgent: Creates AYS tickets
- TicketAgent: Ticket status and updates

ROUTING — assess every message and transfer:

1. NEW issue or NEW action request mid-chat
   → IntentAgent

2. User says "No" / fix didn't work / still having issues
   → TroubleshootAgent (if more fixes available)
   → EscalationAgent (if fixes exhausted)

3. User says "Yes" / resolved / thanks
   → ResolutionAgent

4. User asks for a human / wants to escalate
   → EscalationAgent

5. User asks about ticket status
   → TicketAgent

6. User wants to update/correct their ticket
   → TicketAgent

You NEVER solve issues yourself. Always delegate.""",
        sub_agents=[
            intent_agent,
            api_agent,
            troubleshoot_agent,
            response_agent,
            resolution_agent,
            escalation_agent,
            ticket_agent,
        ],
    )

    return coordinator
