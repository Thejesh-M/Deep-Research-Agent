# Deep Research Agent

A multi-agent deep research system built with LangGraph.

## Features

- **LeadResearcher**: Orchestrator that analyzes complexity and delegates tasks
- **Parallel Subagents**: Execute research tasks concurrently with deep research methodology
- **Tavily Search**: Advanced web search with AI summaries and relevance scoring
- **Deep Research**: Counter-questioning, lead-chasing, and what-if scenario exploration
- **CitationAgent**: Adds proper source attribution
- **Multi-provider**: Supports OpenAI, Anthropic, and Google

## Architecture

```
+----------------+      +------------------+       +------------------+
|   User Query   | ---> |  LeadResearcher  | <---> |   Memory Store   |
+----------------+      |    (Planning)    |       | (Plan/Context)   |
                        +------------------+       +------------------+
                                 |                         ^
                                 v                         |
                        +------------------+               |
                        | SubagentExecutor |               |
                        |    (Parallel)    |               |
                        +------------------+               |
                                 |                         |
           +---------------------+---------------------+   |
           |                     |                     |   |
           v                     v                     v   |
+------------------+    +------------------+    +------------------+
|  Tavily Search   |    |  Tavily Search   |    |  Deep Research   |
|   (Subagent 1)   |    |   (Subagent 2)   |    |   (Subagent N)   |
+------------------+    +------------------+    +------------------+
           |                     |                     |   |
           +---------------------+---------------------+   |
                                 |                         |
                                 v                         |
                        +------------------+               |
                        |  LeadResearcher  | <-------------+
                        |   (Synthesis)    |
                        +------------------+      +------------------+
                                 |                |   Memory Store   |
                                 v                | (Progress/Report)|
                        +------------------+      +------------------+
                        |  CitationAgent   | <---> |                  |
                        |  (Final Report)  |       +------------------+
                        +------------------+
                                 |
                                 v
                                END
                                        (Loop if needed)
```

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Configure environment variables:
```bash
Edit .env and add your API keys:
# - TAVILY_API_KEY (required for web search)
# - OPENAI_API_KEY or GOOGLE_API_KEY (required for LLM)
```

## Usage

```bash
uv run python -m src.main "Your research query" --provider openai
```

## Options

- `--provider`: LLM provider (openai, anthropic, google)
- `--output`: Output directory for reports
- `--max-iterations`: Maximum research iterations
