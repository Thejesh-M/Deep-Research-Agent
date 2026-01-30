from typing import TypedDict, List, Annotated, Optional, Any
import operator
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# Subagent Task Definition
class SubagentTask(BaseModel):
    """Defines a task for a subagent."""
    task_id: str = Field(description="Unique identifier for the task")
    objective: str = Field(description="Clear objective for the subagent")
    search_strategy: str = Field(description="Guidance on search approach: 'broad' or 'specific'")
    output_format: str = Field(description="Expected output format")
    tool_guidance: str = Field(description="Which tools to prioritize")
    boundaries: str = Field(description="What NOT to do / scope limits")

# Subagent Result
class SubagentResult(BaseModel):
    """Result returned by a subagent."""
    task_id: str
    findings: str
    sources: List[dict] = Field(default_factory=list)  # [{url, title, snippet}]
    confidence: float = Field(ge=0, le=1, description="Confidence in findings")
    gaps: List[str] = Field(default_factory=list, description="Information gaps identified")

# Research Plan
class ResearchPlan(BaseModel):
    """Structured research plan created by LeadResearcher."""
    query_complexity: str = Field(description="'simple', 'moderate', or 'complex'")
    estimated_subagents: int = Field(ge=1, le=15)
    strategy: str = Field(description="Overall research strategy")
    subagent_tasks: List[SubagentTask] = Field(default_factory=list)

class AgentState(TypedDict):
    """State for the multi-agent research system."""
    # Core messaging
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Research planning
    research_plan: Optional[ResearchPlan]
    
    # Subagent management
    subagent_tasks: List[SubagentTask]
    subagent_results: List[SubagentResult]
    
    # Memory & context
    memory_context: str  # Retrieved from memory file
    
    # Iteration control
    iteration_count: int
    max_iterations: int
    research_complete: bool
    
    # Sources for citation
    all_sources: List[dict]
    
    # Session info
    conversation_id: str
    output_dir: str  # Where to save markdown files
    provider: str  # LLM provider: 'openai', 'anthropic', 'google'

