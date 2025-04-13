import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import sys
import time
import json
import re
from typing import List, Dict, Any, Optional, Union, Tuple
import traceback # For better error logging
from helper import convert_numpy_types,generate_data_profile,get_active_df,get_active_profile
# LLM and Agent Imports
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings
from llama_index.llms.google_genai import GoogleGenAI # Use LlamaIndex's Gemini integration
# from llama_index.llms.openai import OpenAI # Alternative LLM
# from llama_index.llms.langchain import LangChainLLM # If using LangChain wrappers
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from agent import initialize_agent
# Plotting
import plotly.express as px
import plotly.graph_objects as go
from datacleaning import clean_data
from createplot import create_plot
from executepython import execute_python_code
# Scikit-learn imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans

# Utilities
from io import StringIO

# Set page config
st.set_page_config(page_title="Data Science Agent", layout="wide")

# --- Session State Initialization ---
# Use keys consistently
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_cleaned' not in st.session_state: # Store cleaned df separately
    st.session_state.df_cleaned = None
if 'current_df_display' not in st.session_state: # Which df is being worked on
    st.session_state.current_df_display = 'original' # 'original' or 'cleaned'
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {"plots": {}, "insights": ""}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'data_profile' not in st.session_state: # Profile of the original data
    st.session_state.data_profile = None
if 'data_profile_cleaned' not in st.session_state: # Profile of the cleaned data
    st.session_state.data_profile_cleaned = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'gemini_model' not in st.session_state: # Separate Gemini model for insights
    st.session_state.gemini_model = None
if 'last_executed_code_output' not in st.session_state:
    st.session_state.last_executed_code_output = {"stdout": "", "fig": None, "df_output": None, "error": None}




# --- App title and description ---
st.title("🤖 DataScience Agent")
st.markdown("""
Welcome to your AI-powered data science assistant! Upload your dataset and ask me to:
- **Explore & Understand:** Get summaries, profiles, and answer questions about your data.
- **Clean Data:** Handle missing values, duplicates, outliers, and more.
- **Visualize:** Create various plots like histograms, scatter plots, bar charts, etc.
- **Build Models:** Train regression or classification models and evaluate them.
- **Find Patterns:** Perform clustering analysis.
- **Execute Code:** Run custom Python code for specific tasks.
""")
st.info("⚠️ Please enter your API keys in the sidebar to enable AI features.")


# --- API Key Setup ---
with st.sidebar.container():
    st.header("🔑 API Keys")
    # openai_api_key = st.text_input("OpenAI API Key (Optional)", type="password", key="openai_key")
    gemini_api_key = st.text_input("Google Gemini API Key*", type="password", key="gemini_key")

    if gemini_api_key and not st.session_state.llm:
        try:
            genai.configure(api_key=gemini_api_key)
            # Configure LlamaIndex settings for Gemini
            Settings.llm = GoogleGenAI(model_name="models/gemini-2.0-flash", api_key=gemini_api_key)
            st.session_state.llm = Settings.llm # Store the LLM for agent use
            # Also configure a separate Gemini model for potential direct calls
            st.session_state.gemini_model =GoogleGenAI('models/gemini-2.0-flash', api_key=gemini_api_key)
            st.sidebar.success("Gemini API Key configured!")
        except Exception as e:
            st.sidebar.error(f"Error configuring Gemini: {str(e)}")
            st.session_state.llm = None
            st.session_state.gemini_model = None
    # elif openai_api_key and not st.session_state.llm:
    #     try:
    #         # Configure LlamaIndex settings for OpenAI
    #         Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_api_key) # Or "gpt-4"
    #         st.session_state.llm = Settings.llm
    #         st.sidebar.success("OpenAI API Key configured!")
    #     except Exception as e:
    #         st.sidebar.error(f"Error configuring OpenAI: {str(e)}")
    #         st.session_state.llm = None

# --- Gemini Insight Generation ---
def generate_insights_with_gemini(prompt: str) -> str:
    """Uses the configured Gemini model to generate text based on a prompt."""
    if not st.session_state.gemini_model:
        return "Gemini model not configured. Please add your API key."
    try:
        response = st.session_state.gemini_model.complete(prompt)
        return str(response)
    except Exception as e:
        st.error(f"Error generating insights with Gemini: {str(e)}")
        return f"Error generating insights: {str(e)}"



# --- File Upload ---
st.sidebar.header("💾 Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"], key="file_uploader")

# Load data and generate profile
if uploaded_file is not None:
    # Check if it's a new file or if df is None
    load_new_file = (st.session_state.file_name != uploaded_file.name) or (st.session_state.df is None)
    if load_new_file:
        try:
            st.session_state.file_name = uploaded_file.name
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)

            st.session_state.data_profile = generate_data_profile(st.session_state.df)
            # Reset other states when a new file is loaded
            st.session_state.df_cleaned = None
            st.session_state.current_df_display = 'original'
            st.session_state.analysis_results = {"plots": {}, "insights": ""}
            st.session_state.model_results = {}
            st.session_state.clustering_results = None
            st.session_state.chat_history = []
            st.session_state.data_profile_cleaned = None
            st.session_state.last_executed_code_output = {"stdout": "", "fig": None, "df": None, "error": None}

            st.sidebar.success(f"Loaded '{uploaded_file.name}'")
            st.rerun() # Rerun to update the UI immediately

        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            # Reset states on error
            st.session_state.df = None
            st.session_state.file_name = None
            st.session_state.data_profile = None
            # ... reset other states as above ...

initialize_agent()
# --- Main App Layout ---
if st.session_state.df is None:
    st.warning("Please upload a CSV or Excel file to begin.")
else:
    st.success(f"Working with: **{st.session_state.file_name}**")

    # --- Data Display Options ---
    if st.session_state.df_cleaned is not None:
        display_option = st.radio(
            "Select DataFrame view:",
            ('Original', 'Cleaned'),
            index=1 if st.session_state.current_df_display == 'cleaned' else 0,
            key='df_view_selector',
            horizontal=True
        )
        # Update the active display based on radio button
        new_display_state = display_option.lower()
        if new_display_state != st.session_state.current_df_display:
            st.session_state.current_df_display = new_display_state
            # Update agent context if it exists
            if st.session_state.agent:
                 initialize_agent() # Re-initialize agent with new context
            st.rerun() # Rerun to refresh views and potentially agent context

    # Get the active DataFrame and Profile for display
    active_df = get_active_df()
    active_profile = get_active_profile()

    # --- Tabs for Different Functionalities ---
    tab_titles = [
        "📄 Data Overview",
        "✨ Agent Chat",
        "🧹 Data Cleaning",
        "📊 Analysis Results",
        "🤖 Model Results",
        "🧩 Clustering Results",
        "🐍 Code Execution"
        ]
    tabs = st.tabs(tab_titles)

    # --- Tab 1: Data Overview ---
    with tabs[0]:
        st.header(f"Data Overview ({st.session_state.current_df_display.capitalize()} View)")
        if active_df is not None:
            st.write(f"Shape: {active_df.shape[0]} rows, {active_df.shape[1]} columns")

            # Display Head/Tail
            st.subheader("Preview Data")
            num_rows_display = st.slider("Number of rows to display", 1, 50, 5, key=f"rows_display_{st.session_state.current_df_display}")
            display_head = st.checkbox("Show Head", value=True, key=f"show_head_{st.session_state.current_df_display}")
            if display_head:
                st.dataframe(active_df.head(num_rows_display))
            else:
                 st.dataframe(active_df.tail(num_rows_display))

            # Display Data Profile
            st.subheader("Data Profile")
            if active_profile:
                 # Display profile in a more structured way
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Number of Rows", active_profile.get("num_rows", "N/A"))
                    st.metric("Number of Columns", active_profile.get("num_columns", "N/A"))
                    st.metric("Duplicate Rows", active_profile.get("duplicate_rows", "N/A"))
                    st.metric("Memory Usage", active_profile.get("memory_usage", "N/A"))
                with col2:
                    st.write("**Column Types:**")
                    st.json(active_profile.get("column_types", {}), expanded=False)

                st.write("**Missing Values (%):**")
                missing_perc = active_profile.get("missing_values_percentage", {})
                st.dataframe(pd.Series(missing_perc).rename("Missing %").sort_values(ascending=False))


                numeric_summary = active_profile.get("numeric_summary", {})
                if numeric_summary:
                    st.subheader("Numeric Column Summary")
                    # Transpose describe output for better readability
                    st.dataframe(pd.DataFrame(numeric_summary).T) # Display transposed describe()

                categorical_summary = active_profile.get("categorical_summary", {})
                if categorical_summary:
                     st.subheader("Categorical Column Summary (Top 5 Values)")
                     for col, summary in categorical_summary.items():
                         with st.expander(f"Column: {col}"):
                            if "error" in summary:
                                st.error(summary["error"])
                            else:
                                st.write(f"Unique Values: {summary.get('unique_values', 'N/A')}")
                                st.write("Top 5 Values:")
                                st.dataframe(pd.Series(summary.get('top_5_values', {})).rename('Count'))

            else:
                st.info("Generating profile...")
                if st.button("Generate Profile Now"):
                     profile = generate_data_profile(active_df)
                     if profile:
                        if st.session_state.current_df_display == 'cleaned':
                             st.session_state.data_profile_cleaned = profile
                        else:
                             st.session_state.data_profile = profile
                        st.rerun()
                     else:
                         st.error("Failed to generate profile.")

            # Option to generate insights using Gemini
            st.subheader("🤖 Generate Data Insights (Experimental)")
            if st.button("Ask Gemini for Insights", key="gemini_insight_button"):
                 if active_profile and st.session_state.gemini_model:
                      insight_prompt = f"""
                      Analyze the following data profile and provide key insights, potential issues, and suggestions for analysis.
                      Be concise and focus on actionable points.

                      Data Profile:
                      {json.dumps(convert_numpy_types(active_profile), indent=2)}
                      """
                      with st.spinner("Generating insights with Gemini..."):
                          insights = generate_insights_with_gemini(insight_prompt)
                          st.session_state.analysis_results["insights"] = insights
                          st.rerun() # Display the insight below
                 elif not st.session_state.gemini_model:
                      st.warning("Gemini model not configured. Please add your API key.")
                 else:
                     st.warning("Data profile not available. Generate profile first.")

            if st.session_state.analysis_results.get("insights"):
                 st.markdown("**Gemini Insights:**")
                 st.markdown(st.session_state.analysis_results["insights"])


    # --- Tab 2: Agent Chat ---
    with tabs[1]:
        st.header("💬 Chat with Data Science Agent")

        if st.session_state.agent:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # User input
            if prompt := st.chat_input("Ask the agent about your data... (e.g., 'Clean the data using median imputation', 'Plot histogram of Age', 'Train a classification model for Target')"):
                # Add user message to history and display
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get agent response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    try:
                        with st.spinner("Agent is thinking..."):
                             # Update agent context before query if needed (e.g., if df view changed outside chat)
                             # initialize_agent() # Consider if context needs frequent updates

                             response = st.session_state.agent.chat(prompt)
                             full_response = str(response) # Agent response is typically text

                        message_placeholder.markdown(full_response)

                        # Check if a tool action requires a rerun to update UI
                        if st.session_state.get("agent_action_rerun", False):
                            st.session_state.agent_action_rerun = False # Reset flag
                            # Give a small delay for the user to read the message before rerun
                            time.sleep(1)
                            st.rerun()

                    except Exception as e:
                        full_response = f"An error occurred: {str(e)}"
                        message_placeholder.error(full_response)
                        st.error(traceback.format_exc())


                # Add agent response to history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        else:
            st.warning("Agent not initialized. Please ensure the LLM API key is entered correctly in the sidebar.")

    # --- Tab 3: Data Cleaning ---
    with tabs[2]:
        st.header("🧹 Data Cleaning Configuration")
        st.write("Configure and apply cleaning steps manually, or ask the agent via chat.")
        st.info(f"Cleaning will be applied to the **original** data ('{st.session_state.file_name}'). Results will be shown in the 'Cleaned' view.")

        if st.session_state.df is not None:
             with st.form("cleaning_form"):
                st.subheader("Cleaning Options")
                # Use columns for layout
                col1, col2 = st.columns(2)

                with col1:
                    opt_drop_duplicates = st.checkbox("Drop Duplicate Rows", value=True)
                    opt_fix_dtypes = st.checkbox("Attempt Data Type Fixes (Numeric/Date)", value=True)
                    opt_handle_missing = st.selectbox(
                        "Handle Missing Values",
                        options=['none', 'median', 'mean', 'mode', 'fill', 'drop_rows', 'drop_cols'],
                        index=1 # Default to median
                    )
                    opt_missing_numeric_fill = None
                    opt_missing_categorical_fill = None
                    if opt_handle_missing == 'fill':
                         opt_missing_numeric_fill = st.number_input("Fill Numeric NA with:", value=0)
                         opt_missing_categorical_fill = st.text_input("Fill Categorical NA with:", value="Unknown")
                    elif opt_handle_missing == 'drop_cols':
                         opt_missing_col_drop_threshold = st.slider("Drop Column if Missing > Threshold", 0.0, 1.0, 0.5, 0.05)


                with col2:
                    opt_handle_outliers = st.selectbox(
                        "Handle Outliers",
                        options=[False, 'clip', 'remove'], # False means no outlier handling
                        index=1 # Default to clip
                    )
                    opt_outlier_method = None
                    opt_iqr_multiplier = None
                    opt_zscore_threshold = None
                    if opt_handle_outliers:
                         opt_outlier_method = st.radio("Outlier Detection Method", ['iqr', 'zscore'], index=0, horizontal=True)
                         if opt_outlier_method == 'iqr':
                             opt_iqr_multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
                         else:
                             opt_zscore_threshold = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.1)

                submitted = st.form_submit_button("Apply Cleaning Steps")

                if submitted:
                     cleaning_options = {
                        "drop_duplicates": opt_drop_duplicates,
                        "handle_missing": opt_handle_missing,
                        "missing_numeric_fill": opt_missing_numeric_fill,
                        "missing_categorical_fill": opt_missing_categorical_fill,
                        "missing_col_drop_threshold": opt_missing_col_drop_threshold if opt_handle_missing == 'drop_cols' else None,
                        "handle_outliers": opt_handle_outliers,
                        "outlier_method": opt_outlier_method,
                        "iqr_multiplier": opt_iqr_multiplier,
                        "zscore_threshold": opt_zscore_threshold,
                        "fix_datatypes": opt_fix_dtypes
                    }
                     # Remove None values from options to avoid issues
                     cleaning_options = {k: v for k, v in cleaning_options.items() if v is not None or k in ['missing_numeric_fill']}


                     with st.spinner("Applying cleaning steps..."):
                         cleaned_df, steps = clean_data(st.session_state.df, cleaning_options)

                     if cleaned_df is not None:
                         st.session_state.df_cleaned = cleaned_df
                         st.session_state.data_profile_cleaned = generate_data_profile(cleaned_df)
                         st.session_state.current_df_display = 'cleaned' # Switch view
                         st.success("Data cleaning performed successfully!")
                         st.write("Summary of steps:")
                         st.markdown("- " + "\n- ".join(steps))
                         st.info("Switched view to 'Cleaned' data. You can compare in the 'Data Overview' tab or switch back using the radio button.")
                         # No need for rerun here as state changes trigger it implicitly after button press? Let's add one just in case.
                         time.sleep(0.5) # Short delay
                         st.rerun()

                     else:
                         st.error("Data cleaning failed.")
        else:
             st.warning("Load data first to enable cleaning options.")


    # --- Tab 4: Analysis Results (Plots) ---
    with tabs[3]:
        st.header("📊 Analysis Results")
        st.info("Plots generated by the agent or manually will appear here.")

        if st.session_state.analysis_results["plots"]:
            for plot_key, plot_data in st.session_state.analysis_results["plots"].items():
                 with st.container(border=True):
                    st.subheader(plot_data["title"])
                    st.plotly_chart(plot_data["figure"], use_container_width=True)
                    # Expander for plot details
                    with st.expander("Plot Details"):
                        st.json(plot_data["params"])
                    # Add button to remove plot?
                    if st.button(f"Remove Plot '{plot_data['title']}'", key=f"remove_{plot_key}"):
                        del st.session_state.analysis_results["plots"][plot_key]
                        st.rerun()

        else:
            st.write("No plots generated yet.")

        # Manual Plotting Interface (Optional - can rely on agent)
        with st.expander("Manual Plotting Tool"):
            st.write("Create plots manually using the active dataset.")
            if active_df is not None:
                 plot_cols = active_df.columns.tolist()
                 plot_type = st.selectbox("Plot Type", ['histogram', 'scatter', 'bar', 'box', 'line', 'pie', 'heatmap', 'density_heatmap', 'violin'], key="manual_plot_type")
                 x_col = st.selectbox("X-axis Column", plot_cols, key="manual_x")
                 y_col = None
                 if plot_type in ['scatter', 'bar', 'box', 'line', 'density_heatmap', 'violin']: # y optional for box/violin/bar
                     y_col = st.selectbox("Y-axis Column (Optional for Box/Violin/Bar)", [None] + plot_cols, key="manual_y")
                 color_by = st.selectbox("Color By (Optional)", [None] + plot_cols, key="manual_color")
                 custom_title = st.text_input("Custom Plot Title (Optional)", key="manual_title")

                 if st.button("Generate Manual Plot"):
                      with st.spinner("Generating plot..."):
                          fig = create_plot(active_df, plot_type, x_col, y_col, color_by, custom_title)
                          if fig:
                             plot_key = f"manual_{plot_type}_{x_col}_{y_col}_{color_by}_{time.time():.0f}"
                             st.session_state.analysis_results["plots"][plot_key] = {
                                "figure": fig,
                                "title": custom_title or f"Manual {plot_type.capitalize()} of {x_col}{' vs '+y_col if y_col else ''}",
                                "params": {'plot_type': plot_type, 'x_col': x_col, 'y_col': y_col, 'color_by': color_by, 'title': custom_title}
                            }
                             st.success("Plot generated!")
                             st.rerun() # Rerun to display the plot above
                          else:
                             st.error("Failed to generate plot.")
            else:
                 st.warning("Load data first.")

    # --- Tab 5: Model Results ---
    with tabs[4]:
        st.header("🤖 Model Training Results")

        if st.session_state.model_results:
            results = st.session_state.model_results
            st.subheader("Training Summary")
            st.write(f"**Problem Type:** {results.get('problem_type', 'N/A').capitalize()}")
            if results.get('classification_type'):
                st.write(f"**Classification Type:** {results['classification_type'].capitalize()}")
            st.write(f"**Target Variable:** {results['parameters'].get('target_col', 'N/A')}")
            st.write(f"**Features Used:** {'All (except target)' if not results['parameters'].get('features') else ', '.join(results['parameters']['features'])}")
            st.write(f"**Test Set Size:** {results['parameters'].get('test_size', 'N/A'):.0%}")
            st.write(f"**Best Performing Model:** {results.get('best_model', 'N/A')}")

            st.subheader("Preprocessing Steps Applied:")
            st.markdown("- " + "\n- ".join(results.get('preprocessing_steps', ['None'])))
            if 'target_encoding' in results:
                 st.write("**Target Variable Encoding:**")
                 st.json(results['target_encoding'])


            st.subheader("Model Performance Metrics")
            all_metrics_data = []
            for model_name, model_data in results.get('models', {}).items():
                metrics = model_data.get('metrics', {})
                metrics['Model'] = model_name
                metrics['Training Time (s)'] = model_data.get('training_time_seconds', 'N/A')
                all_metrics_data.append(metrics)

            if all_metrics_data:
                metrics_df = pd.DataFrame(all_metrics_data)
                # Set Model as index and reorder columns for better display
                metrics_df = metrics_df.set_index('Model')
                # Dynamically find metric columns (excluding time)
                metric_cols = [col for col in metrics_df.columns if col not in ['Training Time (s)']]
                ordered_cols = metric_cols + ['Training Time (s)']
                metrics_df = metrics_df[ordered_cols]
                st.dataframe(metrics_df.style.format("{:.4f}", subset=metric_cols).format("{:.3f}", subset=['Training Time (s)'])) # Format numeric metrics


            # Display Confusion Matrix and Feature Importance
            for model_name, model_data in results.get('models', {}).items():
                 with st.expander(f"Details for: {model_name}"):
                    st.write(f"**Metrics for {model_name}:**")
                    st.json(model_data.get('metrics', {}))

                    # Confusion Matrix
                    if 'confusion_matrix' in model_data:
                         st.write("**Confusion Matrix:**")
                         cm = np.array(model_data['confusion_matrix'])
                         labels = model_data.get('confusion_matrix_labels')
                         if labels and len(labels) == cm.shape[0]:
                             try:
                                cm_fig = px.imshow(cm, text_auto=True, aspect="auto",
                                                   labels=dict(x="Predicted Label", y="True Label", color="Count"),
                                                   x=labels, y=labels,
                                                   title=f"Confusion Matrix - {model_name}",
                                                   color_continuous_scale='Blues')
                                cm_fig.update_xaxes(side="top")
                                st.plotly_chart(cm_fig, use_container_width=True)
                             except Exception as cm_plot_e:
                                 st.error(f"Could not plot confusion matrix: {cm_plot_e}")
                                 st.text(str(cm)) # Display as text if plot fails
                         else:
                             st.text(str(cm)) # Display raw matrix if labels mismatch or missing
                             if not labels: st.caption("Labels for matrix axes not available.")
                             elif len(labels) != cm.shape[0]: st.caption(f"Label count ({len(labels)}) doesn't match matrix dimensions ({cm.shape[0]}).")


                    # Feature Importance
                    if 'feature_importance' in model_data:
                        st.write("**Feature Importance:**")
                        imp_data = model_data['feature_importance']
                        imp_df = pd.DataFrame({'Feature': imp_data['features'], 'Importance': imp_data['importance']})
                        # Limit displayed features for readability
                        imp_df_display = imp_df.head(15) # Show top 15

                        try:
                            # Create bar chart for feature importance
                            fig_imp = px.bar(imp_df_display, x='Importance', y='Feature', orientation='h',
                                        title=f"Top {len(imp_df_display)} Feature Importances - {model_name}")
                            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}) # Show most important at top
                            st.plotly_chart(fig_imp, use_container_width=True)
                        except Exception as fi_plot_e:
                            st.error(f"Could not plot feature importances: {fi_plot_e}")
                            st.dataframe(imp_df_display) # Fallback to dataframe


        else:
            st.write("No models trained yet. Ask the agent or use a manual tool (if implemented).")

    # --- Tab 6: Clustering Results ---
    with tabs[5]:
        st.header("🧩 Clustering Results (K-Means)")

        if st.session_state.clustering_results:
            results = st.session_state.clustering_results
            params = results.get('parameters', {})
            summary = results.get('cluster_summary', {})
            assignments = results.get('cluster_assignments') # DF with scaled data + labels

            st.subheader("Clustering Summary")
            st.write(f"**Number of Clusters (k):** {params.get('n_clusters', 'N/A')}")
            st.write(f"**Features Used:** {', '.join(results.get('features_used', ['N/A']))}")
            st.write(f"**Inertia (Sum of squared distances):** {results.get('inertia', 'N/A'):.2f}")

            st.subheader("Preprocessing Steps Applied:")
            st.markdown("- " + "\n- ".join(results.get('preprocessing_steps', ['None'])))

            st.subheader("Cluster Sizes and Centers")
            size_data = {k: v.get('size', 0) for k, v in summary.items()}
            perc_data = {k: v.get('percentage', 0) for k, v in summary.items()}
            centers_data = {k: v.get('center_coordinates (original_scale)', {}) for k, v in summary.items()}

            # Display sizes using metrics or a small bar chart
            st.write("**Cluster Sizes:**")
            cols = st.columns(len(size_data))
            for i, (label, size) in enumerate(size_data.items()):
                 with cols[i]:
                      st.metric(label, f"{size} ({perc_data.get(label, 0)}%)")

            st.write("**Cluster Centers (Original Feature Scale):**")
            centers_df = pd.DataFrame(centers_data).T # Transpose for better view (clusters as rows)
            st.dataframe(centers_df.style.format("{:.2f}"))

            # Display detailed stats per cluster
            st.subheader("Detailed Cluster Statistics (Original Feature Scale)")
            for label, stats in summary.items():
                 with st.expander(f"{label} (Size: {stats['size']})"):
                     st.write(f"**Center Coordinates:**")
                     st.json(stats['center_coordinates (original_scale)'])
                     st.write(f"**Member Statistics:**")
                     member_stats = stats.get('member_statistics (original_scale)')
                     if member_stats:
                         st.dataframe(pd.DataFrame(member_stats).T) # Display describe() output for cluster members
                     else:
                         st.write("No members in this cluster or statistics not available.")


            # Optional: Visualize clusters (e.g., Scatter plot if 2 features used)
            if assignments is not None and len(results.get('features_used', [])) >= 2:
                 st.subheader("Cluster Visualization (Example using first 2 features)")
                 features_used = results.get('features_used')
                 try:
                    # Use scaled data for visualization if desired, or map back to original
                    # Let's use original data for easier interpretation if possible
                    active_df_for_plot = get_active_df()
                    plot_df = active_df_for_plot.loc[assignments.index].copy() # Get original data rows
                    plot_df['cluster'] = assignments['cluster'] # Add cluster label

                    fig_clusters = px.scatter(plot_df,
                                              x=features_used[0],
                                              y=features_used[1],
                                              color='cluster',
                                              color_continuous_scale=px.colors.qualitative.Plotly, # Use qualitative scale for clusters
                                              title=f'Cluster Visualization ({features_used[1]} vs {features_used[0]})',
                                              hover_data=plot_df.columns) # Show all data on hover
                    st.plotly_chart(fig_clusters, use_container_width=True)
                 except Exception as plot_e:
                      st.warning(f"Could not generate cluster visualization: {plot_e}")

        else:
            st.write("No clustering performed yet. Ask the agent or use a manual tool (if implemented).")

    # --- Tab 7: Code Execution ---
    with tabs[6]:
        st.header("🐍 Execute Python Code")
        st.warning("⚠️ **Caution:** Executing code can modify data or cause errors. Use code from trusted sources.")
        st.markdown("""
        You can run custom Python code snippets here. The code has access to:
        - `pd` (pandas), `np` (numpy)
        - `px` (plotly express), `go` (plotly graph objects)
        - `plt` (matplotlib pyplot), `sns` (seaborn)
        - `df`: A *copy* of the currently active DataFrame (`{}`)
        """.format(f"{st.session_state.current_df_display.capitalize()} View: '{st.session_state.file_name}'" if active_df is not None else "No data loaded"))

        default_code = """# Example: Calculate value counts for a column
# Replace 'YourColumnName' with an actual column name from your data
# column_to_analyze = 'YourColumnName'
# if column_to_analyze in df.columns:
#     print(f"Value counts for {column_to_analyze}:")
#     print(df[column_to_analyze].value_counts())
# else:
#     print(f"Column '{column_to_analyze}' not found.")

# Example: Create a plot and assign to 'fig'
# import plotly.express as px
# fig = px.histogram(df, x='YourNumericColumnName') # Replace with your column
# print("Plot generated and assigned to 'fig'") # Optional print statement

# The modified 'df' or the figure assigned to 'fig' will be shown below after execution.
"""
        code_input = st.text_area("Enter Python Code:", value=default_code, height=300)

        if st.button("Execute Code"):
            active_df_copy = get_active_df() # Get the active df
            if active_df_copy is not None:
                 with st.spinner("Executing code..."):
                     execution_output = execute_python_code(code_input, active_df_copy)
                     st.session_state.last_executed_code_output = execution_output # Store result
                 # Rerun to display output below
                 st.rerun()
            else:
                st.error("No data loaded to execute code on.")

        # Display output from the last execution (triggered by rerun)
        last_output = st.session_state.last_executed_code_output
        if last_output:
             st.subheader("Execution Output:")
             if last_output["error"]:
                 st.error("**Error:**")
                 st.code(last_output["error"], language='bash') # Use code block for traceback

             if last_output["stdout"]:
                  st.write("**Printed Output:**")
                  st.text(last_output["stdout"])

             if last_output["fig"]:
                  st.write("**Generated Plot:**")
                  try:
                     if isinstance(last_output["fig"], go.Figure):
                         st.plotly_chart(last_output["fig"], use_container_width=True)
                     elif isinstance(last_output["fig"], plt.Figure):
                          st.pyplot(last_output["fig"])
                     else:
                          st.write("Output figure type not recognized for display.")
                  except Exception as fig_display_e:
                      st.error(f"Error displaying figure: {fig_display_e}")

             if last_output.get("df_output") is not None:
              st.write("**Resulting DataFrame ('df'):**")
              # Access using .get() here too for consistency, though the check above handles it
              st.dataframe(last_output.get("df_output"))
              # Option to apply this df_output to the cleaned df state? Needs careful consideration.
              # Maybe add a button: "Update Cleaned Data with this Result"
              if st.button("Replace 'Cleaned Data' with this Result"):
                  # Use .get() again when retrieving the value to store
                  df_to_update = last_output.get("df_output")
                  if df_to_update is not None:
                      st.session_state.df_cleaned = df_to_update
                      st.session_state.data_profile_cleaned = generate_data_profile(st.session_state.df_cleaned)
                      st.session_state.current_df_display = 'cleaned'
                      st.success("Cleaned data updated with code execution result.")
                      # Clear last output to avoid accidental re-apply?
                      # st.session_state.last_executed_code_output = {}
                      time.sleep(1)
                      st.rerun()



             # Clear the output after displaying once? Or keep it until next execution?
             # Keeping it seems reasonable.


