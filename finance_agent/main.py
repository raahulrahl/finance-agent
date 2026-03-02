"""Finance Agent - AI Financial Analyst."""

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from textwrap import dedent
from typing import Any

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global agent instance
agent: Agent | None = None
_initialized = False
_init_lock = asyncio.Lock()


class AgentNotInitializedError(RuntimeError):
    """Raised when agent is accessed before initialization."""

    pass


def load_config() -> dict:
    """Load agent configuration from project root."""
    # Try multiple possible locations for agent_config.json
    possible_paths = [
        Path(__file__).parent.parent / "agent_config.json",  # Project root
        Path(__file__).parent / "agent_config.json",  # Same directory as main.py
        Path.cwd() / "agent_config.json",  # Current working directory
    ]

    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except (PermissionError, json.JSONDecodeError) as e:
                print(f"⚠️  Error reading {config_path}: {type(e).__name__}")
                continue
            except Exception as e:
                print(f"⚠️  Unexpected error reading {config_path}: {type(e).__name__}")
                continue

    # If no config found or readable, create a minimal default
    print("⚠️  No agent_config.json found, using default configuration")
    return {
        "name": "finance-agent",
        "description": "AI financial analyst agent",
        "version": "1.0.0",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
            "proxy_urls": ["127.0.0.1"],
            "cors_origins": ["*"],
        },
        "environment_variables": [
            {"key": "OPENROUTER_API_KEY", "description": "OpenRouter API key for LLM calls", "required": True},
        ],
    }


async def initialize_agent() -> None:
    """Initialize the finance agent with proper model and tools."""
    global agent

    # Get API keys from environment
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4o")

    if not openrouter_api_key:
        error_msg = (
            "No API key provided. Set OPENROUTER_API_KEY environment variable.\n"
            "Get your API key from: https://openrouter.ai/keys"
        )
        raise ValueError(error_msg)

    # Initialize tools
    # enabling common financial data features
    finance_tools = YFinanceTools()

    search_tools = DuckDuckGoTools()

    # Create the finance agent
    agent = Agent(
        name="Finance Agent",
        model=OpenRouter(
            id=model_name,
            api_key=openrouter_api_key,
            cache_response=True,
            supports_native_structured_outputs=True,
        ),
        tools=[finance_tools, search_tools],
        description=dedent("""\
            You are a seasoned AI Financial Analyst.
            Your goal is to provide accurate financial data, market analysis, and investment insights.
        """),
        instructions=dedent("""\
            CRITICAL FORMATTING RULES - FOLLOW EXACTLY:

            1. **NEVER output raw JSON or mixed content** - Only use clean markdown
            2. **ALWAYS create complete, properly formatted markdown tables** like this:
               | Metric | Value |
               |--------|-------|
               | Price | $100.00 |
               | P/E | 25.5 |
            3. **NEVER use incomplete tables or broken formatting**
            4. **ALWAYS complete your response before ending**
            5. **Use simple bullet points with hyphens (-) for text content**
            6. **Keep responses concise and well-structured**

            CONTENT GUIDELINES:
            - Start with a clear title using # heading
            - Present financial data in complete markdown tables
            - Use bullet points for news and analysis
            - End with a brief summary
            - Include disclaimer about financial advice

            ABSOLUTELY FORBIDDEN:
            - Raw JSON output
            - Incomplete tables
            - Mixed formatting
            - Truncated responses
        """),
        add_datetime_to_context=True,
        markdown=True,
        debug_mode=True,
    )
    print("✅ Finance Agent initialized")


async def run_agent(messages: list[dict[str, str]]) -> Any:
    """Run the agent with the given messages."""
    global agent
    if not agent:
        raise AgentNotInitializedError

    # Run the agent and get response
    return await agent.arun(messages)  # type: ignore[invalid-await]


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming agent messages with lazy initialization."""
    global _initialized

    # Lazy initialization on first call
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing Finance Agent...")
            await initialize_agent()
            _initialized = True

    # Run the async agent
    result = await run_agent(messages)
    return result


async def cleanup() -> None:
    """Clean up any resources."""
    print("🧹 Cleaning up Finance Agent resources...")


def main():
    """Run the main entry point for the Finance Agent."""
    parser = argparse.ArgumentParser(description="Bindu Finance Agent")
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "openai/gpt-4o"),
        help="Model ID for OpenRouter (env: MODEL_NAME)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to agent_config.json (optional)",
    )
    args = parser.parse_args()

    # Set environment variables if provided via CLI
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("🤖 Finance Agent - AI Financial Analyst")
    print("📈 Capabilities: Stock data, Market News, Financial Tables")

    # Load configuration
    config = load_config()

    try:
        # Bindufy and start the agent server
        print("🚀 Starting Bindu Finance Agent server...")
        print(f"🌐 Server will run on: {config.get('deployment', {}).get('url', 'http://127.0.0.1:3773')}")
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Finance Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup on exit
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
