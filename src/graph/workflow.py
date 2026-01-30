"""
Multi-Agent Research Workflow

This workflow implements the orchestrator-worker pattern with:
- LeadResearcher (planning and synthesis)
- Parallel Subagents (research execution)
- CitationAgent (final attribution)
"""
from langgraph.graph import StateGraph, START, END

from src.state.schema import AgentState
from src.agents.lead_researcher import (
    lead_researcher_planning_node,
    lead_researcher_synthesis_node
)
from src.agents.subagent import subagent_executor_node
from src.agents.citation_agent import citation_agent_node


def should_continue_research(state: AgentState) -> str:
    """
    Routing function: decide whether to continue research loop or proceed to citation.
    """
    research_complete = state.get("research_complete", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if research_complete or iteration_count >= max_iterations:
        return "citation"
    else:
        return "continue"


def has_tasks(state: AgentState) -> str:
    """
    Routing function: check if there are subagent tasks to execute.
    """
    tasks = state.get("subagent_tasks", [])
    if tasks:
        return "execute"
    else:
        return "synthesize"


def build_graph():
    """
    Build the multi-agent research workflow graph.
    
    Flow:
    START -> lead_planning -> [has_tasks?]
                                |
                    [execute] -> subagent_executor -> lead_synthesis -> [should_continue?]
                                                                            |
                                                        [continue] -> lead_planning (loop)
                                                        [citation] -> citation_agent -> END
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("lead_planning", lead_researcher_planning_node)
    workflow.add_node("subagent_executor", subagent_executor_node)
    workflow.add_node("lead_synthesis", lead_researcher_synthesis_node)
    workflow.add_node("citation_agent", citation_agent_node)
    
    # Entry point
    workflow.add_edge(START, "lead_planning")
    
    # After planning, check if we have tasks
    workflow.add_conditional_edges(
        "lead_planning",
        has_tasks,
        {
            "execute": "subagent_executor",
            "synthesize": "lead_synthesis"  # Skip execution if no tasks
        }
    )
    
    # After subagent execution, go to synthesis
    workflow.add_edge("subagent_executor", "lead_synthesis")
    
    # After synthesis, decide whether to continue or cite
    workflow.add_conditional_edges(
        "lead_synthesis",
        should_continue_research,
        {
            "continue": "lead_planning",  # Loop back for more research
            "citation": "citation_agent"   # Proceed to final report
        }
    )
    
    # Citation agent ends the workflow
    workflow.add_edge("citation_agent", END)
    
    return workflow.compile()


def get_initial_state(query: str, conversation_id: str, output_dir: str = "./research_output", provider: str = "openai") -> dict:
    """
    Create initial state for the research workflow.
    """
    from langchain_core.messages import HumanMessage
    
    return {
        "messages": [HumanMessage(content=query)],
        "research_plan": None,
        "subagent_tasks": [],
        "subagent_results": [],
        "memory_context": "",
        "iteration_count": 0,
        "max_iterations": 3,
        "research_complete": False,
        "all_sources": [],
        "conversation_id": conversation_id,
        "output_dir": output_dir,
        "provider": provider
    }
