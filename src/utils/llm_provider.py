"""
LLM Provider abstraction for multi-provider support.
"""
import os
from typing import Optional, List, Any
from langchain_core.language_models import BaseChatModel

def get_llm(
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> BaseChatModel:
    """
    Get an LLM instance based on provider.
    
    Args:
        provider: 'openai', 'anthropic', or 'google'
        model: Model name (uses defaults if not specified)
        temperature: Sampling temperature
        **kwargs: Additional provider-specific arguments
    
    Returns:
        A LangChain chat model instance
    """
    provider = provider.lower()
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = model or "gpt-4o"
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)
    
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = model or "claude-sonnet-4-20250514"
        return ChatAnthropic(model=model, temperature=temperature, **kwargs)
    
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = model or "gemini-2.0-flash"
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, **kwargs)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'anthropic', or 'google'.")

def get_default_provider() -> str:
    """Get the default provider based on available API keys."""
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    elif os.getenv("GOOGLE_API_KEY"):
        return "google"
    else:
        raise ValueError("No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY.")
