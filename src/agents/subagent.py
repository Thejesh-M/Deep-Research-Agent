"""
Subagent - Specialized research worker using ReAct agent pattern.

Features:
- Receives detailed task from LeadResearcher
- Uses LangGraph's prebuilt ReAct agent for tool execution
- Returns structured findings with sources
"""
import asyncio
import json
from datetime import datetime
from typing import List
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from src.state.schema import AgentState, SubagentTask, SubagentResult
from src.utils.llm_provider import get_llm
from src.tools.search import search_web, search_web_with_sources, deep_search_web

SUBAGENT_SYSTEM_PROMPT = """You are a specialized Research Subagent with a specific task.

## Your Task
**Objective**: {objective}
**Search Strategy**: {search_strategy}
**Output Format**: {output_format}
**Tool Guidance**: {tool_guidance}
**Boundaries**: {boundaries}

## Current Time and Date
{current_date}

## Deep Research Methodology

### 1. Initial Exploration (Start Broad)
- Begin with general queries to understand the landscape
- Use search_web or search_web_with_sources for initial exploration
- Evaluate the quality and relevance of each result

### 2. Counter-Questioning & Critical Thinking
After each search, ask yourself:
- "What if the opposite is true?"
- "What assumptions am I making?"
- "What alternative explanations exist?"
- "Who might disagree with this finding and why?"
- "What evidence contradicts this?"

### 3. Lead-Chasing & Deep Diving
- Identify interesting leads, contradictions, or gaps in search results
- Use deep_search_web to explore promising leads with follow-up queries
- Look for:
  * Contradictory information that needs reconciliation
  * Questions or uncertainties mentioned in sources
  * Technical details that need explanation
  * Edge cases or exceptions
  * Recent developments or updates

### 4. What-If Scenario Exploration
Generate and explore "what if" questions such as:
- "What if this technology/approach failed?"
- "What if the assumptions change?"
- "What if we look at this from a different industry/context?"
- "What if the timeline was different?"

### 5. Iterative Deepening
- Start with 3-5 broad searches
- Identify 2-3 key leads to chase
- Perform targeted follow-up searches (3-5 more)
- Synthesize contradictions and gaps
- One final round if critical information is still missing

## Available Tools
1. **search_web**: Basic web search with formatted text results
2. **search_web_with_sources**: Structured search with source tracking and scores
3. **deep_search_web**: Advanced search with lead-chasing and follow-up query support

## Research Principles
1. **Quality over quantity**: Better to deeply understand 3 sources than skim 10
2. **Source diversity**: Seek different perspectives and types of sources
3. **Evidence-based**: Always track sources for every claim
4. **Critical evaluation**: Question everything, including your own findings
5. **Gap awareness**: Explicitly note what you don't know

## Output Format
After completing your research, provide findings in this JSON format:
{{
    "findings": "<your detailed findings with critical analysis>",
    "sources": [
        {{"url": "<url>", "title": "<title>", "snippet": "<relevant excerpt>", "score": <relevance score>}}
    ],
    "confidence": <0.0 to 1.0>,
    "gaps": ["<information still missing>"],
    "contradictions": ["<any contradictory information found>"],
    "counter_questions": ["<questions you asked yourself during research>"],
    "leads_explored": ["<leads you chased and what you found>"]
}}
"""


async def execute_subagent_task(task: SubagentTask, provider: str = "openai") -> SubagentResult:
    """
    Execute a single subagent task using ReAct agent pattern.
    """
    model = get_llm(provider=provider, temperature=0.5)
    
    # Format the system prompt with task details
    system_prompt = SUBAGENT_SYSTEM_PROMPT.format(
        objective=task.objective,
        search_strategy=task.search_strategy,
        output_format=task.output_format,
        tool_guidance=task.tool_guidance,
        boundaries=task.boundaries,
        current_date=datetime.now().isoformat()
    )
    
    # Create ReAct agent with all search tools
    agent = create_react_agent(
        model=model,
        tools=[search_web, search_web_with_sources, deep_search_web],
        prompt=system_prompt
    )
    
    # Run the agent
    result = await asyncio.to_thread(
        agent.invoke,
        {"messages": [HumanMessage(content=f"Research task: {task.objective}")]}
    )
    
    # Extract final response
    final_message = result["messages"][-1]
    content = final_message.content if hasattr(final_message, 'content') else str(final_message)
    
    # Parse the response
    try:
        # Try to extract JSON from response
        json_content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_content)
        
        return SubagentResult(
            task_id=task.task_id,
            findings=data.get("findings", content),
            sources=data.get("sources", []),
            confidence=float(data.get("confidence", 0.7)),
            gaps=data.get("gaps", [])
        )
    except (json.JSONDecodeError, ValueError):
        # Non-JSON response - use the raw content as findings
        return SubagentResult(
            task_id=task.task_id,
            findings=content,
            sources=[],
            confidence=0.5,
            gaps=["Response was not in expected JSON format"]
        )
    except Exception as e:
        return SubagentResult(
            task_id=task.task_id,
            findings=content,
            sources=[],
            confidence=0.3,
            gaps=[f"Error parsing response: {e}"]
        )


async def run_subagents_parallel(tasks: List[SubagentTask], provider: str = "openai") -> List[SubagentResult]:
    """
    Run multiple subagent tasks in parallel using asyncio.gather.
    """
    coroutines = [execute_subagent_task(task, provider) for task in tasks]
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    
    # Filter out exceptions and convert to SubagentResult
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            valid_results.append(SubagentResult(
                task_id=tasks[i].task_id,
                findings=f"Error: {result}",
                sources=[],
                confidence=0.0,
                gaps=["Subagent execution failed"]
            ))
        else:
            valid_results.append(result)
    
    return valid_results


def subagent_executor_node(state: AgentState) -> dict:
    """
    LangGraph node that executes all subagent tasks in parallel.
    """
    from langchain_core.messages import AIMessage
    
    tasks = state.get("subagent_tasks", [])
    provider = state.get("provider", "openai")
    
    if not tasks:
        return {"messages": [AIMessage(content="No subagent tasks to execute.")]}
    
    # Run subagents in parallel
    results = asyncio.run(run_subagents_parallel(tasks, provider))
    
    return {
        "subagent_results": results,
        "messages": [AIMessage(content=f"Executed {len(results)} subagent tasks in parallel.")]
    }
