"""
Memory module for persisting research plans and progress to markdown files.
"""
import os
from datetime import datetime
from typing import Optional, List
from src.state.schema import ResearchPlan, SubagentResult

class MemoryStore:
    """Manages markdown-based memory for research sessions."""
    
    def __init__(self, output_dir: str, conversation_id: str):
        self.output_dir = output_dir
        self.conversation_id = conversation_id
        self.plan_file = os.path.join(output_dir, f"research_plan_{conversation_id}.md")
        self.progress_file = os.path.join(output_dir, f"research_progress_{conversation_id}.md")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def save_plan(self, query: str, plan: ResearchPlan) -> str:
        """Save the research plan to markdown file."""
        content = f"""# Research Plan
**Generated**: {datetime.now().isoformat()}
**Query**: {query}
**Complexity**: {plan.query_complexity}
**Estimated Subagents**: {plan.estimated_subagents}

## Strategy
{plan.strategy}

## Subagent Tasks
"""
        for i, task in enumerate(plan.subagent_tasks, 1):
            content += f"""
### Task {i}: {task.task_id}
- **Objective**: {task.objective}
- **Search Strategy**: {task.search_strategy}
- **Output Format**: {task.output_format}
- **Tool Guidance**: {task.tool_guidance}
- **Boundaries**: {task.boundaries}
"""
        
        with open(self.plan_file, 'w') as f:
            f.write(content)
        
        return self.plan_file
    
    def update_progress(self, iteration: int, results: List[SubagentResult], synthesis: str = "") -> str:
        """Update the progress file with subagent results."""
        # Read existing content or start new
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                content = f.read()
        else:
            content = f"""# Research Progress
**Session**: {self.conversation_id}
**Started**: {datetime.now().isoformat()}

---
"""
        
        # Add new iteration
        content += f"""
## Iteration {iteration}
**Time**: {datetime.now().strftime('%H:%M:%S')}

### Subagent Results
"""
        for result in results:
            content += f"""
#### {result.task_id}
**Confidence**: {result.confidence:.0%}

**Findings**:
{result.findings}

**Sources**:
"""
            for src in result.sources:
                content += f"- [{src.get('title', 'Unknown')}]({src.get('url', '#')})\n"
            
            if result.gaps:
                content += "\n**Gaps Identified**:\n"
                for gap in result.gaps:
                    content += f"- {gap}\n"
        
        if synthesis:
            content += f"""
### Synthesis
{synthesis}
"""
        
        content += "\n---\n"
        
        with open(self.progress_file, 'w') as f:
            f.write(content)
        
        return self.progress_file
    
    def get_context(self) -> str:
        """Retrieve context from memory files for context window management."""
        context = ""
        
        if os.path.exists(self.plan_file):
            with open(self.plan_file, 'r') as f:
                context += f"## Current Research Plan\n{f.read()}\n\n"
        
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                progress = f.read()
                # If progress is too long, summarize (keep last 2 iterations)
                sections = progress.split("## Iteration")
                if len(sections) > 3:
                    context += f"## Research Progress (Last 2 Iterations)\n## Iteration{'## Iteration'.join(sections[-2:])}"
                else:
                    context += f"## Research Progress\n{progress}"
        
        return context
    
    def save_final_report(self, report: str, sources: List[dict]) -> str:
        """Save the final research report with citations."""
        report_file = os.path.join(self.output_dir, f"final_report_{self.conversation_id}.md")
        
        content = f"""# Research Report
**Generated**: {datetime.now().isoformat()}

{report}

---

## Sources
"""
        for i, src in enumerate(sources, 1):
            content += f"[{i}] [{src.get('title', 'Unknown Source')}]({src.get('url', '#')})\n"
        
        with open(report_file, 'w') as f:
            f.write(content)
        
        return report_file
