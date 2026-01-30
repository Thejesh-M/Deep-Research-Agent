"""
LeadResearcher Agent - The orchestrator that coordinates the research process.

Responsibilities:
1. Analyze query complexity
2. Create research plan with subagent tasks
3. Save plan to memory
4. Synthesize subagent results
5. Decide if more research is needed
"""
import json
from datetime import datetime
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from src.state.schema import AgentState, ResearchPlan, SubagentTask
from src.utils.llm_provider import get_llm
from src.utils.memory import MemoryStore

LEAD_RESEARCHER_SYSTEM_PROMPT = """You are a Lead Research Agent coordinating a multi-agent research system.

## Your Role
You analyze user queries, create research strategies, and delegate to specialized subagents.

## Complexity Assessment Guidelines
- **Simple** (1 subagent, 3-10 tool calls): Direct fact-finding, single topic
- **Moderate** (2-4 subagents, 10-15 calls each): Comparisons, multi-faceted topics
- **Complex** (5-10+ subagents): Deep research, many interrelated aspects

## Delegation Principles
For each subagent, you MUST provide:
1. **Clear objective**: What specific information to find
2. **Search strategy**: Start broad, then narrow
3. **Output format**: How to structure findings
4. **Tool guidance**: Which tools to prioritize (web search, etc.)
5. **Boundaries**: What NOT to do, scope limits to prevent overlap

## Current Context
{memory_context}

## Current Date
{current_date}

## Instructions
Based on the user query and any existing context, create a detailed research plan.
Return a JSON object with this structure:
{{
    "query_complexity": "simple" | "moderate" | "complex",
    "estimated_subagents": <number>,
    "strategy": "<overall approach>",
    "subagent_tasks": [
        {{
            "task_id": "<unique_id>",
            "objective": "<clear objective>",
            "search_strategy": "broad" | "specific",
            "output_format": "<expected format>",
            "tool_guidance": "<tool recommendations>",
            "boundaries": "<scope limits>"
        }}
    ]
}}
"""

SYNTHESIS_PROMPT = """You are synthesizing research findings from multiple subagents.

## Subagent Results
## Subagent Results
{subagent_results}

## Current Date
{current_date}

## Current Research Plan
{research_plan}

## Iteration
This is iteration {iteration} of max {max_iterations}.

## Instructions
1. Analyze the findings from all subagents
2. Identify any gaps or contradictions
3. Decide if more research is needed

Return a JSON object:
{{
    "synthesis": "<combined findings summary>",
    "gaps": ["<gap1>", "<gap2>"],
    "contradictions": ["<if any>"],
    "needs_more_research": true/false,
    "reason": "<why more research is/isn't needed>",
    "next_tasks": [<if needs_more_research, define new SubagentTask objects>]
}}
"""

def lead_researcher_planning_node(state: AgentState) -> dict:
    """
    Initial planning phase - analyze query and create research plan.
    """
    messages = state.get("messages", [])
    memory_context = state.get("memory_context", "No prior context.")
    output_dir = state.get("output_dir", "./research_output")
    conversation_id = state.get("conversation_id", "default")
    
    # Get user query from last human message
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (isinstance(msg, tuple) and msg[0] == "user"):
            user_query = msg.content if hasattr(msg, 'content') else msg[1]
            break
    
    if not user_query:
        return {"messages": [AIMessage(content="No query found.")]}
    
    # Get LLM
    provider = state.get("provider", "openai")
    model = get_llm(provider=provider, temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", LEAD_RESEARCHER_SYSTEM_PROMPT),
        ("human", "Create a research plan for: {query}")
    ])
    
    chain = prompt | model
    response = chain.invoke({
        "memory_context": memory_context,
        "query": user_query,
        "current_date": datetime.now().isoformat()
    })
    
    # Parse response
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        # Create ResearchPlan
        subagent_tasks = [
            SubagentTask(**task) for task in data.get("subagent_tasks", [])
        ]
        
        research_plan = ResearchPlan(
            query_complexity=data.get("query_complexity", "moderate"),
            estimated_subagents=data.get("estimated_subagents", 2),
            strategy=data.get("strategy", ""),
            subagent_tasks=subagent_tasks
        )
        
        # Save to memory
        memory = MemoryStore(output_dir, conversation_id)
        plan_file = memory.save_plan(user_query, research_plan)
        
        return {
            "research_plan": research_plan,
            "subagent_tasks": subagent_tasks,
            "messages": [AIMessage(content=f"Research plan created with {len(subagent_tasks)} subagents. Plan saved to {plan_file}")],
            "iteration_count": 1
        }
        
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error creating research plan: {e}")]}


def lead_researcher_synthesis_node(state: AgentState) -> dict:
    """
    Synthesis phase - combine subagent results and decide next steps.
    """
    subagent_results = state.get("subagent_results", [])
    research_plan = state.get("research_plan")
    iteration = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 3)
    output_dir = state.get("output_dir", "./research_output")
    conversation_id = state.get("conversation_id", "default")
    all_sources = state.get("all_sources", [])
    
    # Format results for prompt
    results_text = ""
    for result in subagent_results:
        results_text += f"\n### {result.task_id}\n"
        results_text += f"**Confidence**: {result.confidence:.0%}\n"
        results_text += f"**Findings**: {result.findings}\n"
        results_text += f"**Gaps**: {', '.join(result.gaps) if result.gaps else 'None'}\n"
        
        # Collect sources
        for src in result.sources:
            if src not in all_sources:
                all_sources.append(src)
    
    # Get LLM
    provider = state.get("provider", "openai")
    model = get_llm(provider=provider, temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIS_PROMPT),
        ("human", "Synthesize the research findings and decide next steps.")
    ])
    
    chain = prompt | model
    response = chain.invoke({
        "subagent_results": results_text,
        "research_plan": research_plan.strategy if research_plan else "No plan",
        "iteration": iteration,
        "max_iterations": max_iterations,
        "current_date": datetime.now().isoformat()
    })
    
    # Parse response
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        synthesis = data.get("synthesis", "")
        needs_more = data.get("needs_more_research", False) and iteration < max_iterations
        
        # Save progress to memory
        memory = MemoryStore(output_dir, conversation_id)
        memory.update_progress(iteration, subagent_results, synthesis)
        
        if needs_more:
            # Create new subagent tasks with validation
            new_tasks = []
            for task_data in data.get("next_tasks", []):
                try:
                    # Validate required fields before creating
                    if all(k in task_data for k in ["task_id", "objective", "search_strategy", "output_format", "tool_guidance", "boundaries"]):
                        new_tasks.append(SubagentTask(**task_data))
                except Exception:
                    pass  # Skip invalid tasks
            
            if new_tasks:
                return {
                    "subagent_tasks": new_tasks,
                    "subagent_results": [],
                    "iteration_count": iteration + 1,
                    "research_complete": False,
                    "all_sources": all_sources,
                    "messages": [AIMessage(content=f"Iteration {iteration} complete. Starting iteration {iteration + 1} with {len(new_tasks)} new tasks.")]
                }
        
        # Research complete (either needs_more=False or no valid new tasks)
        # Create synthesis from results if LLM didn't provide one
        if not synthesis:
            synthesis = _create_synthesis_from_results(subagent_results)
        
        return {
            "research_complete": True,
            "all_sources": all_sources,
            "memory_context": synthesis,
            "messages": [AIMessage(content=f"Research complete after {iteration} iterations. Proceeding to citation.")]
        }
            
    except Exception as e:
        # Fallback: create synthesis from results
        synthesis = _create_synthesis_from_results(subagent_results)
        return {
            "research_complete": True,
            "all_sources": all_sources,
            "memory_context": synthesis,
            "messages": [AIMessage(content=f"Synthesis completed. Proceeding to citation.")]
        }


def _create_synthesis_from_results(subagent_results: List) -> str:
    """Create a synthesis from subagent results."""
    if not subagent_results:
        return "No research findings available."
    
    synthesis = "## Research Findings\n\n"
    for result in subagent_results:
        synthesis +=f"### {result.task_id}\n\n{result.findings}\n\n"
    return synthesis
