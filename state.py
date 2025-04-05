# state.py
import pandas as pd
from typing import Optional, Dict, Any
import plotly.graph_objects as go

# --- Application State Variables ---
current_df: Optional[pd.DataFrame] = None
current_df_cleaned: Optional[pd.DataFrame] = None
current_profile: Optional[Dict[str, Any]] = None
current_profile_cleaned: Optional[Dict[str, Any]] = None
# last_plot_fig: Optional[go.Figure] = None # Replaced by filename tracking
last_plot_filename: Optional[str] = None # Stores the relative path in static/
last_executed_code_output: Dict[str, Any] = {}
model_results: Dict[str, Any] = {}
clustering_results: Optional[Dict[str, Any]] = None
file_name: Optional[str] = None

# --- State Management Functions ---

def has_current_df() -> bool:
    """Checks if a dataframe is currently loaded."""
    global current_df
    return current_df is not None

def get_current_df(cleaned: bool = False) -> Optional[pd.DataFrame]:
    """Gets the current dataframe (original or cleaned)."""
    return current_df_cleaned if cleaned and current_df_cleaned is not None else current_df

def set_current_df(df: Optional[pd.DataFrame], name: Optional[str] = None, is_cleaned: bool = False, profile: Optional[Dict[str, Any]] = None):
    """Sets the current dataframe (original or cleaned) and its profile."""
    global current_df, current_df_cleaned, current_profile, current_profile_cleaned, file_name
    if is_cleaned:
        current_df_cleaned = df
        current_profile_cleaned = profile
    else:
        current_df = df
        current_profile = profile
        file_name = name # Store original file name
        # Reset cleaned state when a new original file is loaded
        current_df_cleaned = None
        current_profile_cleaned = None
        # Reset other results as well
        reset_results()

def get_current_profile(cleaned: bool = False) -> Optional[Dict[str, Any]]:
    """Gets the current data profile (original or cleaned)."""
    return current_profile_cleaned if cleaned and current_profile_cleaned is not None else current_profile

def get_file_name() -> Optional[str]:
    """Gets the original uploaded file name."""
    return file_name

def get_current_df_name() -> Optional[str]:
    """Gets the name of the current dataframe."""
    return file_name

# def set_last_plot_fig(fig: Optional[go.Figure]): # Replaced by filename tracking
#     """Stores the last generated plot figure."""
#     global last_plot_fig
#     last_plot_fig = fig

# def get_last_plot_fig() -> Optional[go.Figure]: # Replaced by filename tracking
#     """Gets the last generated plot figure."""
#     return last_plot_fig

def set_last_plot_filename(filename: Optional[str]):
    """Stores the filename (relative path in static/) of the last generated plot."""
    global last_plot_filename
    last_plot_filename = filename

def get_last_plot_filename() -> Optional[str]:
    """Gets the filename (relative path in static/) of the last generated plot."""
    return last_plot_filename

def set_last_executed_code_output(output: Dict[str, Any]):
    """Stores the output of the last code execution."""
    global last_executed_code_output
    last_executed_code_output = output

def get_last_executed_code_output() -> Dict[str, Any]:
    """Gets the output of the last code execution."""
    return last_executed_code_output

def set_model_results(results: Dict[str, Any]):
    """Stores model training results."""
    global model_results
    model_results = results

def get_model_results() -> Dict[str, Any]:
    """Gets model training results."""
    return model_results

def set_clustering_results(results: Optional[Dict[str, Any]]):
    """Stores clustering results."""
    global clustering_results
    clustering_results = results

def get_clustering_results() -> Optional[Dict[str, Any]]:
    """Gets clustering results."""
    return clustering_results

def reset_results():
    """Resets plot, execution, model, and clustering results."""
    global last_plot_filename, last_executed_code_output, model_results, clustering_results
    # last_plot_fig = None # Removed
    last_plot_filename = None
    last_executed_code_output = {}
    model_results = {}
    clustering_results = None

def get_full_state_summary() -> Dict[str, Any]:
    """Returns a dictionary summarizing the current state (excluding large data)."""
    # Be careful not to serialize large objects like dataframes or figures here
    return {
        "file_name": file_name,
        "original_rows": len(current_df) if current_df is not None else 0,
        "original_cols": len(current_df.columns) if current_df is not None else 0,
        "cleaned_rows": len(current_df_cleaned) if current_df_cleaned is not None else 0,
        "cleaned_cols": len(current_df_cleaned.columns) if current_df_cleaned is not None else 0,
        "original_profile_exists": current_profile is not None,
        "cleaned_profile_exists": current_profile_cleaned is not None,
        # "plot_exists": last_plot_fig is not None, # Replaced
        "plot_filename_exists": last_plot_filename is not None,
        "execution_output_exists": bool(last_executed_code_output),
        "model_results_exist": bool(model_results),
        "clustering_results_exist": clustering_results is not None,
    }