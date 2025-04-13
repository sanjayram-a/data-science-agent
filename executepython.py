import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from typing import List, Dict, Any, Optional, Union, Tuple
import traceback
from io import StringIO


def execute_python_code(code: str, df_input: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Executes user-provided Python code in a restricted environment.
    The code has access to pandas (pd), numpy (np), plotly.express (px),
    plotly.graph_objects (go), matplotlib.pyplot (plt), seaborn (sns),
    and the currently active dataframe as 'df'.

    Returns a dictionary containing:
    - 'stdout': Captured print statements.
    - 'fig': The last generated Plotly/Matplotlib figure (if any).
    - 'df_output': The modified 'df' variable after execution (if it's a DataFrame).
    - 'error': Error message if execution failed.
    """
    st.warning("⚠️ Executing arbitrary Python code can be risky. Ensure the code is trusted.")

    # Use the *active* dataframe passed to the function
    df_local = df_input.copy() if df_input is not None else None

    # Create a StringIO object to capture stdout
    output_capture = StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_capture

    result_fig = None
    result_df_output = None
    error_message = None
    returned_object = None # Capture the return value of the last expression if any

    try:
        # Create a restricted local namespace
        local_namespace = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'px': px,
            'go': go,
            'df': df_local # Provide the dataframe copy in the namespace
            # Add other safe libraries if needed
        }
        global_namespace = {} # Keep globals empty for more restriction

        # Execute the code
        # Using compile and exec allows capturing the result of the last expression
        compiled_code = compile(code, '<string>', 'exec')
        exec(compiled_code, global_namespace, local_namespace)

        # --- Retrieve results from the namespace ---
        # 1. Get the modified DataFrame ('df' variable)
        if 'df' in local_namespace and isinstance(local_namespace['df'], pd.DataFrame):
            result_df_output = local_namespace['df'] # The user might have modified df directly

        # 2. Look for explicitly assigned figures (e.g., fig = px.histogram(...))
        # Check common figure variable names or the last created plotly/matplotlib figure
        potential_fig_vars = ['fig', 'figure', 'plot']
        for var_name in potential_fig_vars:
             if var_name in local_namespace:
                  obj = local_namespace[var_name]
                  if isinstance(obj, (go.Figure, plt.Figure)):
                      result_fig = obj
                      break # Found a figure

        # If no named figure found, check the last active matplotlib figure
        if result_fig is None and plt.get_fignums(): # Check if any matplotlib figures exist
             result_fig = plt.gcf() # Get current figure

        # plt.show() in user code blocks execution in Streamlit, so we capture gcf() if needed.
        # User should ideally assign the figure to a variable like 'fig'.

    except Exception as e:
        error_message = f"Execution Error: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        # Close matplotlib figures created within exec to prevent display issues
        plt.close('all')


    # Get the captured stdout
    stdout_str = output_capture.getvalue()

    return {
        "stdout": stdout_str,
        "fig": result_fig,
        "df_output": result_df_output, # Return the potentially modified df
        "error": error_message
    }

# --- Gemini Insight Generation ---
def generate_insights_with_gemini(prompt: str) -> str:
    """Uses the configured Gemini model to generate text based on a prompt."""
    if not st.session_state.gemini_model:
        return "Gemini model not configured. Please add your API key."
    try:
        response = st.session_state.gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating insights with Gemini: {str(e)}")
        return f"Error generating insights: {str(e)}"