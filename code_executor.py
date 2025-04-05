# code_executor.py
import pandas as pd
import io
import sys
import traceback
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

def execute_python_code(code: str, df_input: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Executes arbitrary Python code provided by the user.
    Captures stdout, matplotlib plots, and any resulting DataFrame named 'df_result'.
    """
    output_buffer = io.StringIO()
    sys.stdout = output_buffer
    sys.stderr = output_buffer # Capture errors too

    # Make the input dataframe available to the executed code as 'df'
    local_vars = {'df': df_input.copy() if df_input is not None else None, 'pd': pd, 'plt': plt}
    result = {"stdout": "", "fig": None, "df_output": None, "error": None}

    # Store the original plt state
    original_figure = plt.gcf() # Get current figure
    original_axes = plt.gca()   # Get current axes
    original_show = plt.show # Store original show function

    # Override plt.show() to capture the figure instead of displaying it
    captured_fig = None
    def capture_show(*args, **kwargs):
        nonlocal captured_fig
        captured_fig = plt.gcf() # Capture the current figure when show() is called
        # Don't actually show it, just capture
        print("[Plot generated and captured]") # Add a note to stdout

    plt.show = capture_show

    try:
        # Execute the code
        exec(code, globals(), local_vars)

        # Check if a plot was generated and captured
        if captured_fig and captured_fig != original_figure: # Check if a new figure was created and captured
             result["fig"] = captured_fig
             # Important: Close the captured figure to prevent it from interfering with subsequent plots
             plt.close(captured_fig)
        else:
             # If no explicit plt.show() was called, check if the last operation created a plot figure
             current_fig = plt.gcf()
             if plt.get_fignums() and current_fig != original_figure: # Check if there are figures and the current one is new
                 result["fig"] = current_fig
                 print("[Plot generated and captured (implicit)]")
                 plt.close(current_fig) # Close it after capturing

        # Check if a dataframe named 'df_result' was created or modified
        if 'df_result' in local_vars and isinstance(local_vars['df_result'], pd.DataFrame):
            result["df_output"] = local_vars['df_result']
            print("[DataFrame 'df_result' captured]")

    except Exception as e:
        print(f"\n--- Error executing code ---")
        traceback.print_exc(file=output_buffer)
        result["error"] = str(e)
    finally:
        # Restore stdout and plt.show
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        plt.show = original_show
        # Restore the original figure/axes state if necessary
        if plt.get_fignums() and plt.gcf() != original_figure:
             plt.figure(original_figure.number) # Switch back to original figure context if it exists
        elif not plt.get_fignums() and original_figure.get_axes():
             # If all figures were closed, but original had axes, maybe recreate a figure?
             # This part is tricky, might be better to just ensure cleanup.
             pass
        # Ensure any lingering plots are closed if they weren't captured
        current_fig_after = plt.gcf()
        if result["fig"] is None and plt.get_fignums() and current_fig_after != original_figure:
             print("[Closing uncaptured plot]")
             plt.close(current_fig_after)


    result["stdout"] = output_buffer.getvalue()
    output_buffer.close()

    # Clean up None values if not set
    result = {k: v for k, v in result.items() if v is not None}

    return result