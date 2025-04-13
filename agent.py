import json
import streamlit as st
import time
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from typing import List,Optional
from helper import get_active_df, get_active_profile,generate_data_profile,convert_numpy_types
from datacleaning import clean_data
from createplot import create_plot
from trainmodel import train_model
from performclustering import perform_clustering
from executepython import execute_python_code

# --- LlamaIndex Agent Setup ---
# Define Tools for the Agent
def setup_agent_tools() -> List[FunctionTool]:
    """Defines the tools the ReActAgent can use."""
    # Ensure tool functions are defined or imported before this point

    # Tool wrapper functions: These adapt our existing functions to be used as tools.
    # They retrieve the current df from session state when called by the agent.
    def get_data_profile_tool() -> str:
        """
        Provides a summary profile of the currently active dataset,
        including column names, types, missing values, basic statistics for numeric columns,
        and value counts for categorical columns.
        Use this to understand the data structure and content.
        """
        df = get_active_df()
        if df is None:
            return "No dataset is currently loaded."
        profile = get_active_profile() # Use the profile of the active df
        if profile:
            # Convert profile dict to a readable string format for the LLM, handling NumPy types
            return json.dumps(convert_numpy_types(profile), indent=2)
        else:
            # Generate on the fly if profile isn't available for active df
             profile_generated = generate_data_profile(df)
             if profile_generated:
                  return json.dumps(convert_numpy_types(profile_generated), indent=2)
             else:
                  return "Could not generate data profile."

    def get_dataframe_head_tool(num_rows: int = 5) -> str:
        """
        Displays the first N rows (default is 5) of the currently active dataset.
        Useful for quickly inspecting the data values and structure.
        """
        df = get_active_df()
        if df is None:
            return "No dataset is currently loaded."
        return df.head(num_rows).to_markdown()

    def clean_data_tool_wrapper(options: str) -> str:
        """
        Cleans the *original* uploaded dataset based on the provided options specified as a JSON string.
        Updates the 'cleaned' dataset view. Available options include:
        'drop_duplicates' (bool, default: true),
        'handle_missing' (str: 'mean', 'median', 'mode', 'fill', 'drop_rows', 'drop_cols', 'none', default: 'median'),
        'missing_numeric_fill' (numeric, used if handle_missing='fill'),
        'missing_categorical_fill' (str, default: 'Unknown', used if handle_missing='fill'),
        'handle_outliers' (str: 'clip', 'remove', False, default: 'clip'),
        'outlier_method' (str: 'iqr', 'zscore', default: 'iqr'),
        'iqr_multiplier' (float, default: 1.5),
        'zscore_threshold' (float, default: 3),
        'fix_datatypes' (bool, default: true).
        Example JSON: '{"handle_missing": "median", "handle_outliers": "clip"}'
        This tool modifies the internal state, creating a cleaned version of the data.
        It returns a summary of the cleaning steps performed.
        After cleaning, you might want to use 'get_data_profile' again to see the changes.
        """
        df = st.session_state.df # Always clean the original dataframe
        if df is None:
            return "No original dataset loaded to clean."
        try:
            # Parse the JSON string into a dictionary
            cleaning_options = json.loads(options)
        except json.JSONDecodeError:
            return "Invalid options format. Please provide options as a valid JSON string."

        cleaned_df, steps = clean_data(df, cleaning_options)

        if cleaned_df is not None:
            st.session_state.df_cleaned = cleaned_df
            st.session_state.data_profile_cleaned = generate_data_profile(cleaned_df)
            st.session_state.current_df_display = 'cleaned' # Switch view to cleaned
            # Need to trigger rerun in the main loop after agent response
            st.session_state.agent_action_rerun = True # Flag for rerun
            return f"Data cleaning performed. Steps:\n- " + "\n- ".join(steps) + "\nThe active dataset is now the cleaned version."
        else:
            return "Data cleaning failed."

    def create_plot_tool_wrapper(plot_type: str, x_col: str, y_col: Optional[str] = None, color_by: Optional[str] = None, title: Optional[str] = None) -> str:
        """
        Generates a plot of the specified type using the currently active dataset.
        Supported plot_types: 'histogram', 'scatter', 'bar', 'box', 'line', 'pie', 'heatmap', 'density_heatmap', 'violin'.
        Requires 'x_col'. 'y_col' is required for scatter, line, density_heatmap, and optional for bar (aggregates) and box/violin.
        'color_by' is an optional column name to color the plot markers.
        'title' is an optional title for the plot.
        The plot is generated and displayed in the 'Analysis Results' tab.
        Returns a confirmation message or an error.
        """
        df = get_active_df()
        if df is None:
            return "No dataset loaded to create a plot."

        # Simple validation
        if x_col not in df.columns: return f"Error: X-axis column '{x_col}' not found."
        if y_col and y_col not in df.columns: return f"Error: Y-axis column '{y_col}' not found."
        if color_by and color_by not in df.columns: return f"Error: Color-by column '{color_by}' not found."

        valid_plots = ['histogram', 'scatter', 'bar', 'box', 'line', 'pie', 'heatmap', 'density_heatmap', 'violin']
        if plot_type not in valid_plots:
            return f"Error: Invalid plot_type '{plot_type}'. Choose from: {', '.join(valid_plots)}"

        # Generate a unique key for the plot based on parameters
        plot_key = f"{plot_type}_{x_col}_{y_col}_{color_by}_{time.time():.0f}" # Add timestamp to ensure uniqueness

        # Call the plotting function
        fig = create_plot(df, plot_type, x_col, y_col, color_by, title)

        if fig:
            # Store the figure in session state to be displayed by Streamlit
            st.session_state.analysis_results["plots"][plot_key] = {
                "figure": fig,
                "title": title or f"{plot_type.capitalize()} of {x_col}{' vs '+y_col if y_col else ''}",
                "params": {'plot_type': plot_type, 'x_col': x_col, 'y_col': y_col, 'color_by': color_by, 'title': title}
            }
            st.session_state.agent_action_rerun = True # Flag for rerun
            return f"Successfully generated '{plot_type}' plot. It is available in the 'Analysis Results' tab."
        else:
            return f"Failed to generate '{plot_type}' plot. Check columns and plot type requirements."

    def train_model_tool_wrapper(target_col: str, model_type: str = 'auto', features: Optional[str] = None, test_size: float = 0.2) -> str:
        """
        Trains a machine learning model (Regression or Classification) on the currently active dataset.
        'target_col' is the name of the column to predict.
        'model_type' can be 'auto', 'regression', or 'classification'. 'auto' tries to guess based on the target column.
        'features' (optional) is a comma-separated string of column names to use as input features (e.g., "col1,col2,col3"). If None, all other columns are used.
        'test_size' (optional, default 0.2) is the proportion of data used for testing.
        Stores the results (metrics, best model, feature importances) which can be viewed in the 'Model Results' tab.
        Returns a summary message of the training outcome.
        """
        df = get_active_df()
        if df is None:
            return "No dataset loaded for model training."

        feature_list = None
        if features:
             feature_list = [f.strip() for f in features.split(',') if f.strip()]
             if not feature_list:
                 return "Error: Features string provided but resulted in an empty list."

        results, message = train_model(df, target_col, model_type, feature_list, test_size)

        if results:
            st.session_state.model_results = results # Store the entire results dict
            st.session_state.agent_action_rerun = True # Flag for rerun
            return f"Model training successful. {message}. Results are in the 'Model Results' tab."
        else:
            return f"Model training failed. Reason: {message}"

    def perform_clustering_tool_wrapper(n_clusters: int = 3, features: Optional[str] = None) -> str:
        """
        Performs K-Means clustering on the numeric columns of the currently active dataset.
        'n_clusters' (default 3) specifies the number of clusters to find.
        'features' (optional) is a comma-separated string of *numeric* column names to use for clustering (e.g., "num_col1,num_col2"). If None, all numeric columns are used.
        Stores the results (cluster assignments, summaries, inertia) which can be viewed in the 'Clustering Results' tab.
        Returns a summary message of the clustering outcome.
        """
        df = get_active_df()
        if df is None:
            return "No dataset available for clustering."

        feature_list = None
        if features:
             feature_list = [f.strip() for f in features.split(',') if f.strip()]
             if not feature_list:
                 return "Error: Features string provided but resulted in an empty list."

        results, message = perform_clustering(df, n_clusters, feature_list)

        if results:
            st.session_state.clustering_results = results
            st.session_state.agent_action_rerun = True # Flag for rerun
            return f"Clustering successful. {message}. Results are in the 'Clustering Results' tab."
        else:
            return f"Clustering failed. Reason: {message}"

    def execute_python_code_tool_wrapper(code: str) -> str:
        """
        Executes the provided Python code string.
        The code has access to pandas (pd), numpy (np), plotly (px, go), matplotlib (plt), seaborn (sns),
        and the current dataframe as 'df'.
        WARNING: Executes arbitrary code. Use with caution.
        If the code generates a plot (assign it to a variable named 'fig') or modifies the 'df', these might be captured.
        Returns the captured standard output (print statements) and errors, if any.
        Any generated plot or modified DataFrame will be displayed under the 'Code Execution' area after execution.
        """
        df = get_active_df()
        # It's generally safer *not* to let the agent directly modify the main df via code exec.
        # Pass a copy, and let the user decide if they want to update the main df based on the output.
        # However, the current execute_python_code returns the modified df, so we can capture it.

        execution_output = execute_python_code(code, df) # Pass the active df

        # Store the entire output dict
        st.session_state.last_executed_code_output = execution_output
        st.session_state.agent_action_rerun = True # Flag for rerun

        # Construct the response message for the agent
        response = ""
        if execution_output["stdout"]:
            response += "Captured Output:\n```\n" + execution_output["stdout"] + "\n```\n"
        if execution_output["fig"]:
            response += "A plot was generated and is available in the 'Code Execution' area.\n"
        if execution_output["df_output"] is not None and df is not None:
             # Check if the dataframe was actually modified
             if not df.equals(execution_output["df_output"]):
                 response += "The DataFrame 'df' was modified by the code. The result is available in the 'Code Execution' area. You may want to apply these changes permanently.\n" # User action needed
             else:
                 response += "The DataFrame 'df' was accessed but not significantly modified.\n"
        elif execution_output["df_output"] is not None and df is None:
             response += "A DataFrame 'df' was created by the code. The result is available in the 'Code Execution' area.\n"

        if execution_output["error"]:
            response += "Execution Error:\n```\n" + execution_output["error"] + "\n```"
        elif not response:
             response = "Code executed successfully with no major output captured."

        return response

    # Create FunctionTool objects
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

def initialize_agent():
    """Initializes the ReActAgent if LLM is configured."""
    if st.session_state.llm and not st.session_state.agent:
        tools = setup_agent_tools()
        # Create a system prompt that includes context about the data
        system_prompt = """
        You are an AI Data Science Assistant. Your goal is to help users analyze and understand their data.
        You have access to a dataset uploaded by the user and a set of tools to interact with it.
        The user will interact with you in a chat interface.
        When using tools:
        - Always operate on the 'currently active dataset'. The user can switch between the original and cleaned version if cleaning is performed.
        - Before performing complex tasks like modeling or plotting, it's often helpful to use 'get_data_profile_tool' or 'get_dataframe_head_tool' to understand the data structure, column names, and types.
        - Be precise with column names when calling tools.
        - If asked to clean data, use the 'clean_data_tool_wrapper' and specify the cleaning methods in the JSON options argument. This creates a 'cleaned' version.
        - If asked to plot, use 'create_plot_tool_wrapper' with appropriate parameters. The plot will appear in the 'Analysis Results' tab.
        - If asked to model, use 'train_model_tool_wrapper'. Results appear in the 'Model Results' tab.
        - If asked to cluster, use 'perform_clustering_tool_wrapper'. Results appear in the 'Clustering Results' tab.
        - If asked to perform a custom task,pandas related task, can use 'execute_python_code_tool_wrapper', but be mindful of its risks and limitations. Explain what the code does.
        - When unsure about column names or data types, ask the user for clarification or use the profiling tool first.
        - Provide clear explanations of your actions and results. Refer the user to the appropriate tabs where results (plots, models, etc.) are displayed.

        Current Active Dataset Context:
        {data_context}
        """
        # Placeholder for dynamic context
        data_context_str = "No data loaded yet"
        active_df = get_active_df()
        if active_df is not None:
             profile = get_active_profile()
             if profile:
                  context_items = {
                    "File Name": profile.get("file_name"),
                    "Current View": st.session_state.current_df_display.capitalize(),
                    "Rows": profile.get("num_rows"),
                    "Columns": profile.get("num_columns"),
                    "Column Names": profile.get("column_names"),
                    "Column Types": profile.get("column_types"),
                    "Missing Value Summary (Count)": profile.get("missing_values_count")
                  }
                  # Limit context length if needed, handling NumPy types
                  data_context_str = json.dumps(convert_numpy_types(context_items), indent=2)
             else:
                 data_context_str = f"Dataset '{st.session_state.file_name}' loaded ({len(active_df)} rows, {len(active_df.columns)} columns). Profile not generated yet."


        try:
            st.session_state.agent = ReActAgent.from_tools(
                tools=tools,
                llm=st.session_state.llm,
                verbose=True, # Set to True to see agent's thought process in console/logs
                system_prompt=system_prompt.format(data_context=data_context_str),# Inject context
                max_iterations=40
            )
            st.sidebar.success("Data Science Agent initialized!")
        except Exception as e:
            st.sidebar.error(f"Error initializing agent: {e}")
            st.session_state.agent = None
    elif not st.session_state.llm:
         st.sidebar.warning("LLM not configured. Agent cannot be initialized.")