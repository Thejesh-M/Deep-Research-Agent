import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool
from typing import List, Dict

# Load environment variables
load_dotenv()

def get_tavily_client():
    """Get or create Tavily client instance."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    return TavilyClient(api_key=api_key)

@tool
def search_web(query: str, max_results: int = 15) -> str:
    """
    Search the web for information about a topic using Tavily API.
    Returns search results as formatted text with sources.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)
    """
    try:
        tavily_client = get_tavily_client()
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced"
        )
        
        results_text = f"Search results for: {query}\n\n"
        for i, result in enumerate(response.get("results", []), 1):
            results_text += f"{i}. {result.get('title', 'Untitled')}\n"
            results_text += f"   URL: {result.get('url', '')}\n"
            results_text += f"   {result.get('content', '')}\n\n"
        
        return results_text if response.get("results") else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def search_web_with_sources(query: str, max_results: int = 15, search_depth: str = "advanced") -> List[Dict]:
    """
    Search the web and return structured results with source information using Tavily.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)
        search_depth: "basic" or "advanced" - advanced provides deeper, more comprehensive results
    
    Returns:
        List of dicts with: title, url, content, score
    """
    try:
        tavily_client = get_tavily_client()
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_raw_content=False,
            include_answer=True
        )
        
        results = []
        
        if response.get("answer"):
            results.append({
                "title": "AI Summary",
                "url": "",
                "content": response["answer"],
                "score": 1.0,
                "type": "answer"
            })
        
        for result in response.get("results", []):
            results.append({
                "title": result.get("title", "Unknown"),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "type": "source"
            })
        
        return results if results else [{"title": "No Results", "url": "", "content": "No results found.", "score": 0.0, "type": "error"}]
    except Exception as e:
        return [{"title": "Search Error", "url": "", "content": str(e), "score": 0.0, "type": "error"}]

@tool
def deep_search_web(query: str, follow_up_queries: List[str] = None) -> Dict:
    """
    Perform deep web search with automatic follow-up queries and lead-chasing.
    This tool uses Tavily's advanced search and automatically explores related topics.
    
    Args:
        query: The main search query
        follow_up_queries: Optional list of follow-up queries to explore
    
    Returns:
        Dict with: main_results, follow_up_results, leads, summary
    """
    try:
        tavily_client = get_tavily_client()
        main_response = tavily_client.search(
            query=query,
            search_depth="advanced",
            include_answer=True
        )
        
        results = {
            "main_query": query,
            "main_answer": main_response.get("answer", ""),
            "main_results": main_response.get("results", []),
            "follow_up_results": [],
            "leads": []
        }
        
        # Extract potential leads from main results
        leads = []
        for result in main_response.get("results", [])[:3]:
            content = result.get("content", "")
            if "?" in content or "however" in content.lower() or "but" in content.lower():
                leads.append(content[:200] + "...")
        
        results["leads"] = leads
        
        if follow_up_queries:
            for follow_up in follow_up_queries[:3]:
                follow_response = tavily_client.search(
                    query=follow_up,
                    search_depth="advanced"
                )
                results["follow_up_results"].append({
                    "query": follow_up,
                    "results": follow_response.get("results", [])
                })
        
        return results
    except Exception as e:
        return {
            "main_query": query,
            "error": str(e),
            "main_results": [],
            "follow_up_results": [],
            "leads": []
        }
