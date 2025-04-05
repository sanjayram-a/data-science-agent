# llm_utils.py
import os
import google.generativeai as genai
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from typing import Optional, Tuple

# Global variables to hold configured models (consider a class-based approach for better state management)
_genai_model = None
_llama_index_llm = None

def configure_gemini(api_key: str) -> Tuple[Optional[genai.GenerativeModel], Optional[Gemini]]:
    """Configures Google Gemini API for both direct use and LlamaIndex."""
    global _genai_model, _llama_index_llm
    if not api_key:
        print("Error: Gemini API Key is required.")
        return None, None
    try:
        genai.configure(api_key=api_key)
        # Configure LlamaIndex settings for Gemini
        # Using a common model like gemini-pro, adjust if needed
        llama_llm = Gemini(model_name="models/gemini-2.0-flash", api_key=api_key)
        Settings.llm = llama_llm
        _llama_index_llm = llama_llm # Store for potential later use

        # Also configure a separate Gemini model for potential direct calls (e.g., insights)
        genai_model = genai.GenerativeModel('models/gemini-2.0-flash')
        _genai_model = genai_model # Store for potential later use

        print("Gemini API Key configured successfully for genai and LlamaIndex.")
        return genai_model, llama_llm
    except Exception as e:
        print(f"Error configuring Gemini: {str(e)}")
        _genai_model = None
        _llama_index_llm = None
        Settings.llm = None # Reset LlamaIndex setting on error
        return None, None

def get_configured_models() -> Tuple[Optional[genai.GenerativeModel], Optional[Gemini]]:
    """Returns the globally configured Gemini models."""
    return _genai_model, _llama_index_llm

def generate_insights_with_gemini(prompt: str) -> str:
    """Generates insights using the configured Gemini model."""
    genai_model, _ = get_configured_models()
    if not genai_model:
        return "Error: Gemini model not configured. Please configure the API key first."

    if not prompt:
        return "Error: Prompt cannot be empty."

    try:
        print(f"Generating insights with Gemini for prompt: '{prompt[:100]}...'")
        # Use the configured genai.GenerativeModel instance
        response = genai_model.generate_content(prompt)
        # Access the text content safely
        insight = response.text if hasattr(response, 'text') else str(response)
        print("Insights generated successfully.")
        return insight
    except Exception as e:
        print(f"Error generating insights with Gemini: {e}")
        return f"Error generating insights: {str(e)}"

# Example of how to potentially load API key from environment variable
# def configure_from_env():
#     api_key = os.getenv("GEMINI_API_KEY")
#     if api_key:
#         configure_gemini(api_key)
#     else:
#         print("Warning: GEMINI_API_KEY environment variable not set.")