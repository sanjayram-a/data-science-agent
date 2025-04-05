# agent_tools.py
import pandas as pd
import json
import traceback
import io
import base64
import matplotlib.pyplot as plt # Import needed for code execution context and plot saving
from typing import List, Dict, Any, Optional

# LlamaIndex imports
from llama_index.core.tools import FunctionTool

# Import functions from our modules
from data_utils import generate_data_profile, load_dataframe # Assuming load_dataframe might be needed indirectly or for context
from cleaning import clean_data
from visualization import create_plot
from modeling import train_model
from clustering import perform_clustering
from code_executor import execute_python_code
# Import state management functions
import state

# State is now managed by the 'state' module. Tools will call functions from state.py

# --- Agent Tool Definitions ---

def setup_agent_tools() -> List[FunctionTool]:
    """Defines the tools the ReActAgent can use."""

    # --- Tool: Get Data Profile ---
    def get_data_profile_tool(use_cleaned_data: bool = False) -> str:
        """
        Provides a summary profile of the currently loaded dataset (either original or cleaned).
        Includes information like column names, types, missing values, basic statistics for numeric columns,
        and value counts for categorical columns.
        Args:
            use_cleaned_data (bool): If True, profile the cleaned dataset, otherwise profile the original. Defaults to False.
        """
        print(f"Tool: get_data_profile_tool called (use_cleaned_data={use_cleaned_data})")
        profile = state.get_current_profile(cleaned=use_cleaned_data)
        if profile:
            # Convert profile dict to a JSON string for the agent
            try:
                # Use convert_numpy_types if necessary before dumping
                # profile = convert_numpy_types(profile) # Assuming profile might contain numpy types
                profile_str = json.dumps(profile, indent=2)
                return f"Data profile:\n{profile_str}"
            except Exception as e:
                 # Fallback to simple string representation if JSON fails
                 print(f"Warning: Could not dump profile to JSON: {e}. Returning string representation.")
                 return f"Data profile:\n{str(profile)}"
        else:
            df_state = "cleaned" if use_cleaned_data else "original"
            return f"Error: No {df_state} data profile available. Load data first."

    # --- Tool: Get DataFrame Head ---
    def get_dataframe_head_tool(num_rows: int = 5, use_cleaned_data: bool = False) -> str:
        """
        Returns the first N rows (head) of the currently active pandas DataFrame (original or cleaned).
        Args:
            num_rows (int): The number of rows to return. Defaults to 5.
            use_cleaned_data (bool): If True, use the cleaned dataset, otherwise use the original. Defaults to False.
        """
        print(f"Tool: get_dataframe_head_tool called (num_rows={num_rows}, use_cleaned_data={use_cleaned_data})")
        df = state.get_current_df(cleaned=use_cleaned_data)
        if df is not None:
            try:
                return df.head(num_rows).to_markdown() # Return as markdown table
            except Exception as e:
                return f"Error converting dataframe head to markdown: {e}"
        else:
            df_state = "cleaned" if use_cleaned_data else "original"
            return f"Error: No {df_state} dataset loaded."

    # --- Tool: Clean Data ---
    def clean_data_tool_wrapper(options: str) -> str:
        """
        Cleans the currently loaded dataset based on the provided options specified as a JSON string.
        Updates the internal 'cleaned' dataset state. Returns a summary of cleaning steps performed or an error message.
        Example options JSON: '{"drop_duplicates": true, "handle_missing": "median", "handle_outliers": "clip", "outlier_method": "iqr"}'
        Available options:
            - drop_duplicates (bool): Remove duplicate rows. Default: true.
            - handle_missing (str): Method for missing values ('mean', 'median', 'mode', 'fill', 'drop_rows', 'drop_cols', 'none'). Default: 'median'.
            - missing_numeric_fill (float/int): Value to fill numeric NaNs if handle_missing='fill'. Default: 0.
            - missing_categorical_fill (str): Value to fill categorical NaNs if handle_missing='fill'. Default: 'Unknown'.
            - missing_col_drop_threshold (float): Threshold (0-1) to drop columns with missing values if handle_missing='drop_cols'. Default: 0.5.
            - handle_outliers (str/bool): Method for outliers ('clip', 'remove', false). Default: 'clip'.
            - outlier_method (str): Method to detect outliers ('iqr', 'zscore'). Default: 'iqr'.
            - iqr_multiplier (float): IQR multiplier if outlier_method='iqr'. Default: 1.5.
            - zscore_threshold (float): Z-score threshold if outlier_method='zscore'. Default: 3.
            - fix_datatypes (bool): Attempt automatic conversion of object columns to numeric/datetime. Default: true.
        Args:
            options (str): A JSON string containing the cleaning options.
        """
        print(f"Tool: clean_data_tool_wrapper called with options: {options}")
        df_original = state.get_current_df(cleaned=False) # Always clean the original data
        if df_original is None:
            return "Error: No original dataset loaded to clean."

        try:
            cleaning_options = json.loads(options)
            if not isinstance(cleaning_options, dict):
                raise ValueError("Options must be a JSON object (dictionary).")
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for cleaning options."
        except ValueError as e:
            return f"Error in options format: {e}"
        except Exception as e:
            return f"Error parsing options: {e}"

        try:
            cleaned_df, steps = clean_data(df_original, cleaning_options)
            if cleaned_df is not None:
                # Generate profile for the cleaned data
                original_profile = state.get_current_profile()
                file_name_base = state.get_file_name() or 'data' # Use state function
                cleaned_profile = generate_data_profile(cleaned_df, file_name=f"{file_name_base}_cleaned")
                # Update the application state using state module
                state.set_current_df(cleaned_df, is_cleaned=True, profile=cleaned_profile)
                steps_summary = "\n".join(steps)
                return f"Data cleaning successful. Cleaned data is now available.\nCleaning Steps:\n{steps_summary}"
            else:
                # clean_data returns steps even on failure (e.g., if df was None initially)
                steps_summary = "\n".join(steps)
                return f"Data cleaning failed. Steps attempted/Messages:\n{steps_summary}"
        except Exception as e:
            traceback.print_exc()
            return f"Error during data cleaning process: {str(e)}"

    # --- Tool: Create Plot ---
    def create_plot_tool_wrapper(plot_type: str, x_col: str, y_col: Optional[str] = None, color_by: Optional[str] = None, title: Optional[str] = None, use_cleaned_data: bool = True) -> str:
        """
        Generates a plot using Plotly based on the specified parameters and the active dataset (cleaned by default).
        Stores the plot figure internally. Returns a confirmation message or error.
        The plot can be retrieved or displayed using other mechanisms (not directly returned by this tool).
        Args:
            plot_type (str): Type of plot (e.g., 'histogram', 'scatter', 'bar', 'box', 'line', 'heatmap', 'pie').
            x_col (str): Column name for the X-axis.
            y_col (Optional[str]): Column name for the Y-axis (required for some plot types like scatter, line).
            color_by (Optional[str]): Column name to color the plot markers by.
            title (Optional[str]): Custom title for the plot.
            use_cleaned_data (bool): If True, use the cleaned dataset, otherwise use the original. Defaults to True.
        """
        print(f"Tool: create_plot_tool_wrapper called (plot_type={plot_type}, x={x_col}, y={y_col}, color={color_by}, cleaned={use_cleaned_data})")
        df = state.get_current_df(cleaned=use_cleaned_data)
        df_state = "cleaned" if use_cleaned_data else "original"
        if df is None:
            return f"Error: No {df_state} dataset loaded to create a plot."

        try:
            # Add any extra kwargs if needed, maybe passed as a JSON string? For now, keep it simple.
            kwargs = {}
            fig = create_plot(df, plot_type, x_col, y_col, color_by, title, **kwargs)

            if fig:
                # state.set_last_plot_fig(fig) # Removed: We now track the filename after saving
                # Save the plot to HTML
                try:
                    # Sanitize column names for filename
                    safe_x = "".join(c if c.isalnum() else "_" for c in x_col)
                    safe_y = "_" + "".join(c if c.isalnum() else "_" for c in y_col) if y_col else ""
                    # Save in the root directory first, app.py will move it to static/
                    plot_filename = f"plot_{plot_type}_{safe_x}{safe_y}.html"
                    fig.write_html(plot_filename, auto_open=False)
                    # --- Update state with the filename ---
                    state.set_last_plot_filename(plot_filename)
                    # --- Return confirmation message ---
                    return f"Successfully created {plot_type} plot for '{x_col}' (and '{y_col}' if applicable). Plot saved as {plot_filename}. View it on the Chart Analysis page."
                except Exception as save_e:
                     print(f"Warning: Could not save plot to HTML: {save_e}")
                     # Ensure state doesn't hold a filename if save failed
                     state.set_last_plot_filename(None)
                     return f"Failed to save {plot_type} plot as HTML file. Error: {save_e}"
            else:
                # create_plot function prints warnings/errors internally
                return f"Failed to create {plot_type} plot. Check logs or previous messages for details (e.g., missing columns, wrong data types)."
        except Exception as e:
            traceback.print_exc()
            return f"Error creating plot: {str(e)}"

    # --- Tool: Train Model ---
    def train_model_tool_wrapper(target_col: str, model_type: str = 'auto', features: Optional[str] = None, test_size: float = 0.2, use_cleaned_data: bool = True) -> str:
        """
        Trains a machine learning model (regression or classification) on the active dataset (cleaned by default).
        Stores the results (metrics, model pipeline) internally.
        Args:
            target_col (str): The name of the target variable column.
            model_type (str): Type of model ('auto', 'regression', 'classification'). Default: 'auto'.
            features (Optional[str]): JSON string list of feature column names. If None, use all columns except target. Example: '["col1", "col2"]'.
            test_size (float): Proportion of the data to use for the test set (0.0 to 1.0). Default: 0.2.
            use_cleaned_data (bool): If True, use the cleaned dataset, otherwise use the original. Defaults to True.
        """
        print(f"Tool: train_model_tool_wrapper called (target={target_col}, type={model_type}, cleaned={use_cleaned_data})")
        df = state.get_current_df(cleaned=use_cleaned_data)
        df_state = "cleaned" if use_cleaned_data else "original"
        if df is None:
            return f"Error: No {df_state} dataset loaded for training."

        feature_list = None
        if features:
            try:
                feature_list = json.loads(features)
                if not isinstance(feature_list, list):
                    raise ValueError("Features must be a JSON list of strings.")
                # Basic validation if feature names are strings
                if not all(isinstance(item, str) for item in feature_list):
                     raise ValueError("All items in the features list must be strings (column names).")
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for features list."
            except ValueError as e:
                return f"Error in features format: {e}"
            except Exception as e:
                return f"Error parsing features: {e}"

        try:
            results, message = train_model(df, target_col, model_type, feature_list, test_size)
            if results:
                # Store results (excluding the potentially large pipeline object for the message)
                results_summary = {k: v for k, v in results.items() if k != 'model_pipeline'}
                state.set_model_results(results) # Store full results internally via state module
                # Safely dump summary to JSON
                try:
                    summary_str = json.dumps(results_summary, indent=2)
                except TypeError: # Handle potential non-serializable items if any slip through
                    summary_str = str(results_summary)
                return f"Model training successful.\nSummary:\n{summary_str}\nFull results stored internally."
            else:
                return f"Model training failed: {message}" # Message contains the error
        except Exception as e:
            traceback.print_exc()
            return f"Error during model training process: {str(e)}"

    # --- Tool: Perform Clustering ---
    def perform_clustering_tool_wrapper(n_clusters: int = 3, features: Optional[str] = None, use_cleaned_data: bool = True) -> str:
        """
        Performs K-Means clustering on the active dataset (cleaned by default).
        Stores the results (cluster labels, sizes, pipeline) internally.
        Args:
            n_clusters (int): The number of clusters to find. Default: 3.
            features (Optional[str]): JSON string list of feature column names. If None, use only numeric columns by default. Example: '["col1", "col2"]'.
            use_cleaned_data (bool): If True, use the cleaned dataset, otherwise use the original. Defaults to True.
        """
        print(f"Tool: perform_clustering_tool_wrapper called (k={n_clusters}, cleaned={use_cleaned_data})")
        df = state.get_current_df(cleaned=use_cleaned_data)
        df_state = "cleaned" if use_cleaned_data else "original"
        if df is None:
            return f"Error: No {df_state} dataset loaded for clustering."

        feature_list = None
        if features:
            try:
                feature_list = json.loads(features)
                if not isinstance(feature_list, list):
                    raise ValueError("Features must be a JSON list of strings.")
                if not all(isinstance(item, str) for item in feature_list):
                     raise ValueError("All items in the features list must be strings (column names).")
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for features list."
            except ValueError as e:
                return f"Error in features format: {e}"
            except Exception as e:
                return f"Error parsing features: {e}"

        try:
            results, message = perform_clustering(df, n_clusters, feature_list)
            if results:
                 # Store results (excluding the potentially large pipeline object for the message)
                results_summary = {k: v for k, v in results.items() if k != 'clustering_pipeline'}
                state.set_clustering_results(results) # Store full results internally via state module
                 # Safely dump summary to JSON
                try:
                    summary_str = json.dumps(results_summary, indent=2)
                except TypeError:
                    summary_str = str(results_summary)
                return f"Clustering successful.\nSummary:\n{summary_str}\nFull results stored internally."
            else:
                return f"Clustering failed: {message}" # Message contains the error
        except Exception as e:
            traceback.print_exc()
            return f"Error during clustering process: {str(e)}"

    # --- Tool: Execute Python Code ---
    def execute_python_code_tool_wrapper(code: str, use_cleaned_data: bool = True) -> str:
        """
        Executes arbitrary Python code. The active DataFrame (cleaned by default) is available as a variable named 'df'.
        Any print statements are captured as stdout. Matplotlib plots created with plt.show() are captured.
        A DataFrame named 'df_result' created in the code will also be captured.
        Returns a summary of the execution (stdout, whether a plot or df_result was captured, and any errors).
        Args:
            code (str): The Python code string to execute.
            use_cleaned_data (bool): If True, provide the cleaned dataset as 'df', otherwise provide the original. Defaults to True.
        """
        print(f"Tool: execute_python_code_tool_wrapper called (cleaned={use_cleaned_data})")
        df = state.get_current_df(cleaned=use_cleaned_data)
        df_state = "cleaned" if use_cleaned_data else "original"
        current_df_for_exec = df.copy() if df is not None else None # Pass a copy

        if current_df_for_exec is None:
            # Allow execution even without a dataframe, df will be None in the code
            print(f"Warning: No {df_state} dataset loaded, 'df' variable in executed code will be None.")

        try:
            # Make sure matplotlib is imported if user code uses plt
            # Also import pandas and numpy for convenience
            code_to_run = "import matplotlib.pyplot as plt\nimport pandas as pd\nimport numpy as np\n" + code
            result = execute_python_code(code_to_run, current_df_for_exec) # Pass the copy
            state.set_last_executed_code_output(result) # Store result internally via state module

            # Format the output string for the agent
            output_lines = []
            if result.get("stdout"):
                output_lines.append("--- Stdout ---")
                # Limit stdout length for brevity in agent response
                stdout_preview = result['stdout'][:1000]
                output_lines.append(stdout_preview)
                if len(result['stdout']) > 1000:
                    output_lines.append("... (stdout truncated)")
                output_lines.append("--------------")
            if result.get("fig"):
                output_lines.append("--- Plot Captured ---")
                output_lines.append("A matplotlib plot was generated and stored.")
                # Optionally save the plot here as well
                try:
                    plot_filename = "executed_code_plot.png" # Save as png for easier display
                    # Ensure the figure object is valid before saving
                    if hasattr(result['fig'], 'savefig'):
                         result['fig'].savefig(plot_filename)
                         output_lines.append(f"Plot saved as {plot_filename}.")
                         plt.close(result['fig']) # Close figure after saving
                    else:
                         output_lines.append("(Warning: Captured plot object was not a valid figure)")
                except Exception as save_e:
                    output_lines.append(f"(Warning: Failed to save plot: {save_e})")
                output_lines.append("-------------------")

            if result.get("df_output") is not None:
                 output_lines.append("--- DataFrame 'df_result' Captured ---")
                 output_lines.append(f"Shape: {result['df_output'].shape}")
                 try:
                     output_lines.append("Head:\n" + result['df_output'].head().to_markdown())
                 except Exception as md_e:
                     print(f"Warning: Could not convert df_result head to markdown: {md_e}")
                     output_lines.append("Head:\n" + str(result['df_output'].head()))
                 output_lines.append("------------------------------------")
                 # Potentially update state with df_result? Needs careful consideration.
                 # For now, just report its capture.

            if result.get("error"):
                output_lines.append(f"--- Execution Error ---")
                # Limit error length
                error_preview = result["error"][:1000]
                output_lines.append(error_preview)
                if len(result["error"]) > 1000:
                    output_lines.append("... (error truncated)")
                output_lines.append("---------------------")

            if not output_lines:
                 return "Code executed successfully. No stdout, plot, or 'df_result' captured."
            else:
                 return "\n".join(output_lines)

        except Exception as e:
            traceback.print_exc()
            error_msg = f"Critical error during code execution wrapper: {str(e)}\n{traceback.format_exc()}"
            state.set_last_executed_code_output({"error": error_msg, "stdout": ""}) # Store error via state module
            # Limit length of critical error message returned to agent
            return error_msg[:1500]


    # --- Create FunctionTool objects ---
    tools = [
        FunctionTool.from_defaults(fn=get_data_profile_tool),
        FunctionTool.from_defaults(fn=get_dataframe_head_tool),
        FunctionTool.from_defaults(fn=clean_data_tool_wrapper),
        FunctionTool.from_defaults(fn=create_plot_tool_wrapper),
        FunctionTool.from_defaults(fn=train_model_tool_wrapper),
        FunctionTool.from_defaults(fn=perform_clustering_tool_wrapper),
        FunctionTool.from_defaults(fn=execute_python_code_tool_wrapper),
    ]
    return tools

# (Removed placeholder state functions and example update_state function)