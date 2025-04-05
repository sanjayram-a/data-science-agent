# agent_setup.py
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings # To access the configured LLM
from typing import List, Optional
import traceback # For printing detailed errors

# Import the function that defines the tools
from agent_tools import setup_agent_tools

# Global variable to hold the initialized agent
_agent: Optional[ReActAgent] = None

def initialize_agent() -> Optional[ReActAgent]:
    """
    Initializes the ReActAgent with the configured LLM and tools.
    Assumes Settings.llm has been configured previously (e.g., by llm_utils.configure_gemini).
    """
    global _agent
    llm = Settings.llm # Get the LLM configured in LlamaIndex settings

    if llm is None:
        print("Error: LLM not configured in LlamaIndex Settings. Cannot initialize agent.")
        print("Please configure the LLM first (e.g., using llm_utils.configure_gemini).")
        _agent = None
        return None

    try:
        print("Setting up agent tools...")
        tools = setup_agent_tools()
        if not tools:
            print("Error: No tools were generated for the agent.")
            _agent = None
            return None

        print(f"Initializing ReActAgent with {len(tools)} tools...")
        # You might want to adjust verbosity
        agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)
        _agent = agent # Store the initialized agent globally
        print("ReActAgent initialized successfully.")
        return agent
    except Exception as e:
        print(f"Error initializing agent: {e}")
        traceback.print_exc()
        _agent = None
        return None

def get_agent() -> Optional[ReActAgent]:
    """Returns the globally initialized agent."""
    # If not initialized, try initializing it now
    if _agent is None:
        print("Agent not initialized. Attempting initialization...")
        initialize_agent()
    return _agent