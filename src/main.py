"""
Deep Research Agent - Multi-Agent Research System

Usage:
    python -m src.main "Your research query here"
    python -m src.main "Your query" --provider openai --output ./output
"""
import sys
import uuid
import argparse
import dotenv
from src.graph.workflow import build_graph, get_initial_state

dotenv.load_dotenv(".env.example")

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Deep Research System")
    parser.add_argument("query", help="Research query")
    parser.add_argument("--provider", default="openai", 
                        choices=["openai", "anthropic", "google"],
                        help="LLM provider to use")
    parser.add_argument("--output", default="./research_output",
                        help="Directory to save research outputs")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum research iterations")
    
    args = parser.parse_args()
    
    conversation_id = str(uuid.uuid4())[:8]
    print(f"\n{'='*60}")
    print(f"Session ID: {conversation_id}")
    print(f"Provider: {args.provider}")
    print(f"Output: {args.output}")
    print(f"Query: {args.query}")
    print(f"{'='*60}\n")
    
    # Build graph
    graph = build_graph()
    
    # Create initial state
    initial_state = get_initial_state(
        query=args.query,
        conversation_id=conversation_id,
        output_dir=args.output,
        provider=args.provider
    )
    initial_state["max_iterations"] = args.max_iterations
    
    print("Starting research...\n")
    
    for event in graph.stream(initial_state):
        for node_name, state_update in event.items():
            print(f"\n--- {node_name.upper()} ---")
            
            if "messages" in state_update:
                for msg in state_update["messages"]:
                    content = getattr(msg, "content", str(msg))
                    if len(content) > 500:
                        print(f"{content[:500]}...")
                    else:
                        print(content)
            
            if "research_plan" in state_update and state_update["research_plan"]:
                plan = state_update["research_plan"]
                print(f"\n📋 Research Plan:")
                print(f"   Complexity: {plan.query_complexity}")
                print(f"   Subagents: {plan.estimated_subagents}")
                print(f"   Strategy: {plan.strategy[:100]}...")
            
            if "subagent_results" in state_update:
                results = state_update["subagent_results"]
                print(f"\n📊 Subagent Results: {len(results)} completed")
                for r in results:
                    print(f"   - {r.task_id}: {r.confidence:.0%} confidence")
    
    print(f"\n{'='*60}")
    print(f"Research complete!")
    print(f"Check: {args.output}/final_report_{conversation_id}.md")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
