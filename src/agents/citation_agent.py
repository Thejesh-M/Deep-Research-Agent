"""
CitationAgent - Processes final research report to add proper citations.

Responsibilities:
1. Identify claims that need citations
2. Match claims to source documents
3. Insert inline citations
4. Generate bibliography
"""
import json
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from src.state.schema import AgentState
from src.utils.llm_provider import get_llm
from src.utils.memory import MemoryStore

CITATION_AGENT_PROMPT = """You are a Citation Agent responsible for adding proper academic-style citations to a research report.

## Research Synthesis
{synthesis}

## Available Sources
{sources}

## Current Date
{current_date}

## Instructions
1. Create a polished research report based on the synthesis
2. Add inline citations [1], [2], etc. for factual claims
3. Match each citation to the appropriate source
4. Ensure all major claims are attributed

Return a JSON object:
{{
    "report": "<full report with inline citations>",
    "citations_used": [<list of source indices used, 1-indexed>]
}}
"""

def citation_agent_node(state: AgentState) -> dict:
    """
    Process the research findings and add proper citations.
    """
    synthesis = state.get("memory_context", "")
    all_sources = state.get("all_sources", [])
    output_dir = state.get("output_dir", "./research_output")
    conversation_id = state.get("conversation_id", "default")
    
    sources_text = ""
    for i, src in enumerate(all_sources, 1):
        sources_text += f"[{i}] {src.get('title', 'Unknown')}: {src.get('url', 'No URL')}\n"
        if src.get('snippet'):
            sources_text += f"    Snippet: {src.get('snippet')[:200]}...\n"
    
    if not sources_text:
        sources_text = "No sources available."
    
    provider = state.get("provider", "openai")
    model = get_llm(provider=provider, temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", CITATION_AGENT_PROMPT),
        ("human", "Create the final cited research report.")
    ])
    
    chain = prompt | model
    response = chain.invoke({
        "synthesis": synthesis,
        "sources": sources_text,
        "current_date": datetime.now().isoformat()
    })
    
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        report = data.get("report", synthesis)
        citations_used = data.get("citations_used", [])

        # Filter sources to only those cited
        if citations_used:
            cited_sources = [all_sources[i-1] for i in citations_used if 0 < i <= len(all_sources)]
        else:
            cited_sources = all_sources
        
        # Save final report
        memory = MemoryStore(output_dir, conversation_id)
        report_file = memory.save_final_report(report, cited_sources)
        
        return {
            "messages": [AIMessage(content=f"# Research Report\n\n{report}\n\n---\n*Report saved to: {report_file}*")]
        }
        
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"# Research Report\n\n{synthesis}\n\n---\n*Note: Citation processing failed: {e}*")]
        }
