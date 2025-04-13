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

# LLM and Agent Imports
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings
from llama_index.llms.google_genai import GoogleGenAI # Use LlamaIndex's Gemini integration
# from llama_index.llms.openai import OpenAI # Alternative LLM
# from llama_index.llms.langchain import LangChainLLM # If using LangChain wrappers
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# Plotting
import plotly.express as px
import plotly.graph_objects as go

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
import base64

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

# --- Helper function to get the currently active DataFrame ---
def get_active_df():
    """Returns the currently active DataFrame (original or cleaned)"""
    if st.session_state.current_df_display == 'cleaned' and st.session_state.df_cleaned is not None:
        return st.session_state.df_cleaned
    elif st.session_state.df is not None:
        return st.session_state.df
    else:
        return None

def get_active_profile():
    """Returns the profile of the currently active DataFrame"""
    if st.session_state.current_df_display == 'cleaned' and st.session_state.data_profile_cleaned is not None:
        return st.session_state.data_profile_cleaned
    elif st.session_state.data_profile is not None:
        return st.session_state.data_profile
    else:
        return None

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

# --- File Upload ---
st.sidebar.header("💾 Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"], key="file_uploader")

# Function to generate data profile
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

def generate_data_profile(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Generates a dictionary containing profiling information about the dataframe."""
    if df is None:
        return None
    try:
        profile = {
            "file_name": st.session_state.get('file_name', 'N/A'),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "column_types": df.dtypes.astype(str).to_dict(),
            "missing_values_count": df.isnull().sum().to_dict(),
            "missing_values_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1e6:.2f} MB",
            "numeric_summary": {},
            "categorical_summary": {}
        }

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        if numeric_cols:
            profile["numeric_summary"] = df[numeric_cols].describe().round(2).to_dict()

        for col in categorical_cols:
            try:
                counts = df[col].value_counts()
                summary = {
                    "unique_values": counts.nunique(),
                    "top_5_values": counts.head(5).to_dict()
                }
                profile["categorical_summary"][col] = summary
            except Exception as e:
                profile["categorical_summary"][col] = {"error": f"Could not compute summary: {str(e)}"}

        return profile
    except Exception as e:
        st.error(f"Error generating data profile: {e}")
        return None

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


# --- Data Cleaning Functions ---
# (Keep the clean_data function as provided in the prompt, it's quite good)
def clean_data(df: pd.DataFrame, options: Optional[Dict[str, Any]] = None) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Clean the dataset based on specified options"""
    if df is None:
        return None, ["No dataset loaded"]

    cleaned_df = df.copy()
    cleaning_steps = []

    if options is None: # Default options if none provided
        options = {
            "drop_duplicates": True,
            "handle_missing": "median", # Changed default to median for numeric
            "missing_numeric_fill": None, # Specific value if handle_missing is 'fill'
            "missing_categorical_fill": "Unknown", # Default fill for categorical
            "handle_outliers": "clip", # Options: False, 'clip', 'remove'
            "outlier_method": "iqr", # Options: 'iqr', 'zscore'
            "iqr_multiplier": 1.5,
            "zscore_threshold": 3,
            "fix_datatypes": True # Attempts to convert object columns
        }

    st.write("--- Cleaning Data ---")
    st.write("Options used:", options) # Show options being used

    # Drop duplicates
    if options.get("drop_duplicates", False):
        original_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        dropped_rows = original_rows - len(cleaned_df)
        if dropped_rows > 0:
            step = f"Removed {dropped_rows} duplicate rows."
            cleaning_steps.append(step)
            st.write(f"✔️ {step}")

    # Handle missing values
    missing_method = options.get("handle_missing", "none").lower()
    numeric_cols = cleaned_df.select_dtypes(include=np.number).columns
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns

    if missing_method != "none":
        total_missing_before = cleaned_df.isnull().sum().sum()
        if total_missing_before > 0:
            if missing_method == "drop_rows":
                original_rows = len(cleaned_df)
                cleaned_df = cleaned_df.dropna()
                dropped_rows = original_rows - len(cleaned_df)
                step = f"Dropped {dropped_rows} rows with any missing values."
                cleaning_steps.append(step)
                st.write(f"✔️ {step}")
            elif missing_method == "drop_cols":
                 # Drop columns with high percentage of missing values (e.g., > 50%)
                 threshold = options.get("missing_col_drop_threshold", 0.5)
                 cols_to_drop = cleaned_df.columns[cleaned_df.isnull().mean() > threshold]
                 if not cols_to_drop.empty:
                     cleaned_df = cleaned_df.drop(columns=cols_to_drop)
                     step = f"Dropped columns with >{threshold*100}% missing values: {', '.join(cols_to_drop)}."
                     cleaning_steps.append(step)
                     st.write(f"✔️ {step}")
                     # Re-evaluate numeric/categorical columns
                     numeric_cols = cleaned_df.select_dtypes(include=np.number).columns
                     categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns

            elif missing_method in ["mean", "median", "mode", "fill"]:
                 # Numeric Imputation
                 num_imputed_cols = []
                 for col in numeric_cols:
                     if cleaned_df[col].isnull().any():
                         if missing_method == "mean":
                             fill_value = cleaned_df[col].mean()
                         elif missing_method == "median":
                             fill_value = cleaned_df[col].median()
                         elif missing_method == "mode":
                              fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
                         elif missing_method == "fill":
                             fill_value = options.get("missing_numeric_fill", 0) # Default to 0 if not specified

                         cleaned_df[col].fillna(fill_value, inplace=True)
                         num_imputed_cols.append(col)
                 if num_imputed_cols:
                     step = f"Filled missing numeric values in ({', '.join(num_imputed_cols)}) using '{missing_method}' strategy."
                     cleaning_steps.append(step)
                     st.write(f"✔️ {step}")

                 # Categorical Imputation
                 cat_imputed_cols = []
                 cat_fill_method = "mode" if missing_method == "mode" else "fill" # Use mode if main method is mode, else use 'fill' strategy
                 for col in categorical_cols:
                     if cleaned_df[col].isnull().any():
                          if cat_fill_method == "mode":
                              fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "Unknown"
                          else: # fill strategy
                              fill_value = options.get("missing_categorical_fill", "Unknown")

                          cleaned_df[col].fillna(fill_value, inplace=True)
                          cat_imputed_cols.append(col)
                 if cat_imputed_cols:
                     strategy_desc = "mode" if cat_fill_method == "mode" else f"value '{fill_value}'"
                     step = f"Filled missing categorical values in ({', '.join(cat_imputed_cols)}) using {strategy_desc}."
                     cleaning_steps.append(step)
                     st.write(f"✔️ {step}")

            total_missing_after = cleaned_df.isnull().sum().sum()
            if total_missing_before > 0 and total_missing_after == 0:
                 st.write("✔️ All missing values handled.")
            elif total_missing_after > 0:
                 st.warning(f"⚠️ Still {total_missing_after} missing values remaining after handling.")


    # Fix data types (basic attempt)
    if options.get("fix_datatypes", False):
        converted_cols = []
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            try:
                # Attempt numeric conversion
                converted = pd.to_numeric(cleaned_df[col], errors='coerce')
                # If a significant portion converts, and few NaNs are introduced compared to original non-numeric
                if converted.notna().sum() > 0.8 * cleaned_df[col].notna().sum() and \
                   converted.isna().sum() - cleaned_df[col].isna().sum() < 0.1 * len(cleaned_df):
                     # Check if it's integer-like
                    if (converted.dropna() % 1 == 0).all():
                        # Try converting to nullable integer type
                         try:
                            cleaned_df[col] = converted.astype(pd.Int64Dtype())
                            converted_cols.append(f"{col} (to Integer)")
                            continue # Move to next column
                         except: pass # If Int64Dtype fails, fallback to float/object

                    cleaned_df[col] = converted
                    converted_cols.append(f"{col} (to Numeric/Float)")
                    continue # Move to next column
            except: pass # Ignore errors during numeric conversion attempt

            try:
                # Attempt datetime conversion (only if it looks like a date)
                 # More robust date check
                if cleaned_df[col].astype(str).str.contains(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})|(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})').any():
                    original_non_na = cleaned_df[col].notna().sum()
                    converted_dt = pd.to_datetime(cleaned_df[col], errors='coerce')
                    # Convert only if most non-NA values parse correctly
                    if converted_dt.notna().sum() > 0.8 * original_non_na:
                        cleaned_df[col] = converted_dt
                        converted_cols.append(f"{col} (to Datetime)")
                        continue
            except: pass # Ignore errors during datetime conversion

        if converted_cols:
             step = f"Attempted automatic data type conversion for columns: {', '.join(converted_cols)}."
             cleaning_steps.append(step)
             st.write(f"✔️ {step}")

    # Handle outliers
    outlier_handling = options.get("handle_outliers", False)
    if outlier_handling:
        outlier_method = options.get("outlier_method", "iqr").lower()
        numeric_cols_for_outliers = cleaned_df.select_dtypes(include=np.number).columns
        outliers_handled_cols = []

        for col in numeric_cols_for_outliers:
            original_min = cleaned_df[col].min()
            original_max = cleaned_df[col].max()
            num_outliers = 0

            if outlier_method == "iqr":
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                multiplier = options.get("iqr_multiplier", 1.5)
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
            elif outlier_method == "zscore":
                threshold = options.get("zscore_threshold", 3)
                mean = cleaned_df[col].mean()
                std = cleaned_df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            else: # No valid method
                continue

            outlier_mask = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
            num_outliers = outlier_mask.sum()

            if num_outliers > 0:
                if outlier_handling == 'clip':
                    cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                    action = "Clipped"
                elif outlier_handling == 'remove':
                     # Note: Removing rows might affect other columns. Usually clipping or winsorizing is preferred.
                     # This removes the entire row where an outlier is found in this column.
                     cleaned_df = cleaned_df[~outlier_mask]
                     action = "Removed rows containing"
                else: # False or invalid option
                    continue

                outliers_handled_cols.append(f"{col} ({num_outliers} outliers)")
                step = f"{action} {num_outliers} outliers in '{col}' using {outlier_method.upper()} method (bounds: {lower_bound:.2f} - {upper_bound:.2f})."
                cleaning_steps.append(step)
                st.write(f"✔️ {step}")

    if not cleaning_steps:
        cleaning_steps.append("No cleaning steps were applied based on the provided options.")
        st.write("No cleaning steps were applied based on the provided options.")

    st.write("--- Cleaning Finished ---")
    return cleaned_df, cleaning_steps

# --- Data Visualization Functions ---
# (Keep the create_plot function as provided in the prompt, it's solid)
def create_plot(df: pd.DataFrame, plot_type: str, x_col: str, y_col: Optional[str] = None, color_by: Optional[str] = None, title: Optional[str] = None, **kwargs) -> Optional[go.Figure]:
    """Create a plotly plot based on specified parameters"""
    if df is None or x_col not in df.columns:
        st.warning(f"Cannot create plot. Dataframe is missing or column '{x_col}' not found.")
        return None
    if y_col and y_col not in df.columns:
        st.warning(f"Cannot create plot. Column '{y_col}' not found.")
        return None
    if color_by and color_by not in df.columns:
        st.warning(f"Cannot create plot. Color-by column '{color_by}' not found.")
        color_by = None # Ignore color_by if column doesn't exist

    fig = None
    plot_type = plot_type.lower()
    base_title = title or f"{plot_type.capitalize()} Plot"

    try:
        if plot_type == "histogram":
            fig = px.histogram(df, x=x_col, color=color_by, title=f"{base_title}: Distribution of {x_col}", **kwargs)
        elif plot_type == "scatter":
            if y_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_by, title=f"{base_title}: {y_col} vs {x_col}", hover_data=df.columns, **kwargs)
            else:
                st.warning("Scatter plot requires both X and Y columns.")
                return None
        elif plot_type == "bar":
            if y_col: # Aggregate y_col by x_col
                agg_func = kwargs.get('agg_func', 'mean') # Allow specifying agg func, default mean
                try:
                    # Handle potential non-numeric y_col for certain aggregations like 'count' or 'nunique'
                    if agg_func in ['count', 'size']:
                        temp_df = df.groupby(x_col).size().reset_index(name=y_col)
                    elif agg_func == 'nunique':
                        temp_df = df.groupby(x_col)[y_col].nunique().reset_index()
                    else: # Attempt numeric aggregation
                        if pd.api.types.is_numeric_dtype(df[y_col]):
                             temp_df = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                        else:
                             st.warning(f"Cannot perform '{agg_func}' on non-numeric column '{y_col}'. Using count instead.")
                             temp_df = df.groupby(x_col).size().reset_index(name='count')
                             y_col = 'count' # Update y_col to 'count'
                except Exception as agg_e:
                     st.error(f"Could not aggregate '{y_col}' by '{x_col}' using '{agg_func}': {agg_e}. Using count instead.")
                     temp_df = df.groupby(x_col).size().reset_index(name='count')
                     y_col = 'count' # Update y_col to 'count'

                fig = px.bar(temp_df, x=x_col, y=y_col, title=f"{base_title}: {agg_func.capitalize()} of {y_col} by {x_col}", color=color_by if color_by in temp_df.columns else None, **kwargs) # Color by only if present after aggregation
            else: # Count occurrences of x_col
                counts = df[x_col].value_counts().reset_index()
                counts.columns = [x_col, 'count']
                fig = px.bar(counts, x=x_col, y='count', title=f"{base_title}: Count of {x_col}", **kwargs)
        elif plot_type == "box":
             # Box plot can work with only x (for distribution) or x and y
            fig = px.box(df, x=x_col, y=y_col, color=color_by, title=f"{base_title}: Box Plot of {y_col if y_col else x_col}", **kwargs)
        elif plot_type == "line":
            if y_col:
                 # Ensure x is sortable (numeric or datetime)
                if pd.api.types.is_numeric_dtype(df[x_col]) or pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    temp_df = df.sort_values(by=x_col)
                    fig = px.line(temp_df, x=x_col, y=y_col, color=color_by, title=f"{base_title}: {y_col} vs {x_col}", markers=kwargs.get('markers', False), **kwargs)
                else:
                    st.warning(f"Line plot requires a numeric or datetime X-axis. '{x_col}' is {df[x_col].dtype}.")
                    return None
            else:
                st.warning("Line plot requires both X and Y columns.")
                return None
        elif plot_type == "pie":
            counts = df[x_col].value_counts().reset_index()
            counts.columns = [x_col, 'count']
            max_slices = kwargs.get('max_slices', 10) # Limit slices for readability
            if len(counts) > max_slices:
                 # Group smallest slices into 'Other'
                 counts_top = counts.nlargest(max_slices-1, 'count')
                 other_count = counts['count'][max_slices-1:].sum()
                 counts_top = pd.concat([counts_top, pd.DataFrame({x_col: ['Other'], 'count': [other_count]})], ignore_index=True)
                 counts = counts_top

            fig = px.pie(counts, values='count', names=x_col, title=f"{base_title}: Distribution of {x_col}", hole=kwargs.get('hole', 0), **kwargs) # Add hole option for donut chart
        elif plot_type == "heatmap":
             # Typically used for correlation matrix or pivot table visualization
             # Let's default to correlation matrix of numeric columns
             numeric_cols = df.select_dtypes(include=np.number).columns
             if len(numeric_cols) > 1:
                 corr_matrix = df[numeric_cols].corr()
                 fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                 title=f"{base_title}: Correlation Matrix of Numeric Features",
                                 color_continuous_scale=kwargs.get('color_scale', 'RdBu_r')) # Example color scale
             else:
                 st.warning("Heatmap (correlation) requires at least two numeric columns.")
                 return None
        elif plot_type == "density_heatmap":
            if y_col:
                fig = px.density_heatmap(df, x=x_col, y=y_col, title=f"{base_title}: Density Heatmap of {y_col} vs {x_col}", **kwargs)
            else:
                st.warning("Density Heatmap requires both X and Y columns.")
                return None
        elif plot_type == "violin":
            fig = px.violin(df, x=x_col, y=y_col, color=color_by, box=kwargs.get('box', True), points=kwargs.get('points', "all"), title=f"{base_title}: Violin Plot of {y_col if y_col else x_col}", **kwargs)


        if fig:
            fig.update_layout(
                template="plotly_white",
                title_x=0.5, # Center title
                xaxis_title=x_col,
                yaxis_title=y_col if y_col else ("Count" if plot_type in ["histogram", "bar"] else None),
                legend_title_text=color_by if color_by else None
            )
    except Exception as e:
        st.error(f"Error creating {plot_type} plot: {str(e)}")
        st.error(traceback.format_exc()) # Print detailed traceback
        return None

    return fig

# --- Machine Learning Functions ---
# (Keep the train_model function as provided, it's well-structured)
def train_model(df: pd.DataFrame, target_col: str, model_type: str = 'auto', features: Optional[List[str]] = None, test_size: float = 0.2, random_state: int = 42) -> Tuple[Optional[Dict[str, Any]], str]:
    """Train a machine learning model based on the data and target column"""
    if df is None:
        return None, "Dataset not available."
    if target_col not in df.columns:
        return None, f"Target column '{target_col}' not found in the dataset."

    results = {'parameters': {'target_col': target_col, 'model_type': model_type, 'features': features, 'test_size': test_size, 'random_state': random_state}}

    try:
        # Select features or use all columns except target
        if features:
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                return None, f"Specified features not found: {', '.join(missing_features)}"
            if target_col in features:
                 return None, f"Target column '{target_col}' cannot be included in the features."
            X = df[features].copy()
        else:
            X = df.drop(columns=[target_col]).copy()

        y = df[target_col].copy()

        # --- Preprocessing ---
        preprocessing_steps = []

        # Handle Categorical Features (using one-hot encoding for simplicity here)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dummy_na=False) # Handle NA during get_dummies if imputation didn't catch it
            preprocessing_steps.append(f"Applied one-hot encoding to: {', '.join(categorical_cols)}.")

        # Handle Datetime Features (extract basic features)
        datetime_cols = X.select_dtypes(include=['datetime64[ns]']).columns
        if not datetime_cols.empty:
            for col in datetime_cols:
                X[f'{col}_year'] = X[col].dt.year
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_day'] = X[col].dt.day
                X[f'{col}_dayofweek'] = X[col].dt.dayofweek
            X = X.drop(columns=datetime_cols)
            preprocessing_steps.append(f"Extracted year, month, day, dayofweek from: {', '.join(datetime_cols)}.")


        # Ensure all columns are numeric before imputation/scaling
        non_numeric_final = X.select_dtypes(exclude=np.number).columns
        if not non_numeric_final.empty:
             return None, f"Preprocessing failed. Non-numeric columns remain after encoding: {', '.join(non_numeric_final)}. Consider cleaning data first."

        # Handle missing values (Imputation)
        if X.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='median') # Use median as a robust default
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            preprocessing_steps.append("Applied median imputation for missing values.")
        else:
            preprocessing_steps.append("No missing values found in features.")

        # Scaling (optional but good practice, especially for linear models)
        # scaler = StandardScaler()
        # X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        # preprocessing_steps.append("Applied StandardScaler to features.")
        # Using X directly for now, as RF is less sensitive, but scaling could be added

        results['preprocessing_steps'] = preprocessing_steps

        # --- Model Type Determination ---
        if model_type == 'auto':
             # Improved auto-detection
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15: # Threshold for continuous vs discrete
                problem_type = 'regression'
            elif pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or pd.api.types.is_bool_dtype(y) or y.nunique() <= 15:
                 problem_type = 'classification'
                 # Check for binary vs multiclass
                 if y.nunique() == 2:
                      results['classification_type'] = 'binary'
                 elif y.nunique() > 2:
                     results['classification_type'] = 'multiclass'
                 else:
                     return None, "Target column has only one unique value. Cannot train a model."
                 # Convert target to numerical labels for classification models if it's not already
                 if not pd.api.types.is_numeric_dtype(y):
                     le = LabelEncoder()
                     y = le.fit_transform(y)
                     results['target_encoding'] = dict(zip(le.classes_, le.transform(le.classes_)))
                     preprocessing_steps.append(f"Encoded target variable '{target_col}' using LabelEncoder.")
            else:
                return None, f"Could not automatically determine model type for target '{target_col}' with dtype {y.dtype} and {y.nunique()} unique values."
        elif model_type in ['regression', 'classification']:
            problem_type = model_type
            if problem_type == 'classification':
                 if y.nunique() == 2:
                      results['classification_type'] = 'binary'
                 elif y.nunique() > 2:
                     results['classification_type'] = 'multiclass'
                 else:
                     return None, "Target column has only one unique value. Cannot train a model."
                 if not pd.api.types.is_numeric_dtype(y):
                      le = LabelEncoder()
                      y = le.fit_transform(y)
                      results['target_encoding'] = dict(zip(le.classes_, le.transform(le.classes_)))
                      preprocessing_steps.append(f"Encoded target variable '{target_col}' using LabelEncoder.")
        else:
             return None, f"Invalid model_type specified: '{model_type}'. Choose 'auto', 'regression', or 'classification'."

        results['problem_type'] = problem_type

        # --- Data Splitting ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if problem_type=='classification' and results.get('classification_type') else None) # Stratify for classification


        # --- Model Training & Evaluation ---
        models_trained = {}
        if problem_type == 'regression':
            model_classes = {
                'Linear Regression': LinearRegression(),
                'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1) # Use more cores
            }
            metric_names = ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'R-squared']
            metrics_to_calc = [mean_squared_error, lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), mean_absolute_error, r2_score]
            higher_is_better = [False, False, False, True] # R2 higher is better, errors lower is better

            for name, model in model_classes.items():
                start_time = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                end_time = time.time()

                model_metrics = {}
                for m_name, m_func in zip(metric_names, metrics_to_calc):
                    try:
                        model_metrics[m_name] = m_func(y_test, y_pred)
                    except Exception as metric_e:
                         model_metrics[m_name] = f"Error: {metric_e}"

                model_result = {'metrics': model_metrics, 'training_time_seconds': round(end_time - start_time, 3)}

                # Feature importance for Tree-based models
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    model_result['feature_importance'] = {
                        'features': X.columns[indices].tolist(),
                        'importance': importances[indices].tolist()
                    }
                models_trained[name] = model_result

            # Determine best model (lower RMSE is better)
            best_model_name = min(models_trained.keys(), key=lambda k: models_trained[k]['metrics'].get('Root Mean Squared Error', float('inf')))


        elif problem_type == 'classification':
            model_classes = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1),
                'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
            }
            metric_names = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)']
            # Use average='weighted' for multiclass, default 'binary' is fine for binary
            avg_strategy = 'weighted' if results.get('classification_type') == 'multiclass' else 'binary'
            metrics_to_calc = [accuracy_score,
                               lambda yt, yp: precision_score(yt, yp, average=avg_strategy, zero_division=0),
                               lambda yt, yp: recall_score(yt, yp, average=avg_strategy, zero_division=0),
                               lambda yt, yp: f1_score(yt, yp, average=avg_strategy, zero_division=0)]
            higher_is_better = [True, True, True, True] # All classification metrics higher is better

            for name, model in model_classes.items():
                start_time = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None # For ROC etc if needed later
                end_time = time.time()

                model_metrics = {}
                for m_name, m_func in zip(metric_names, metrics_to_calc):
                    try:
                         model_metrics[m_name] = m_func(y_test, y_pred)
                    except Exception as metric_e:
                         model_metrics[m_name] = f"Error: {metric_e}"

                model_result = {'metrics': model_metrics, 'training_time_seconds': round(end_time - start_time, 3)}

                # Confusion Matrix
                try:
                    cm = confusion_matrix(y_test, y_pred)
                    model_result['confusion_matrix'] = cm.tolist()
                    # Include labels if encoding was done
                    if 'target_encoding' in results:
                         labels = list(results['target_encoding'].keys())
                         # Ensure labels match matrix dimensions if LabelEncoder was used
                         encoded_labels = list(results['target_encoding'].values())
                         # Sort labels based on their encoded value to match confusion_matrix output
                         sorted_labels = [label for _, label in sorted(zip(encoded_labels, labels))]
                         model_result['confusion_matrix_labels'] = sorted_labels
                    else:
                         # Try to get unique sorted labels directly from y_test if no encoding map
                         model_result['confusion_matrix_labels'] = sorted(pd.Series(y_test).unique().tolist())

                except Exception as cm_e:
                    model_result['confusion_matrix'] = f"Error generating confusion matrix: {cm_e}"


                # Feature importance for Tree-based models
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    model_result['feature_importance'] = {
                        'features': X.columns[indices].tolist(),
                        'importance': importances[indices].tolist()
                    }

                models_trained[name] = model_result

            # Determine best model (higher Accuracy is better)
            best_model_name = max(models_trained.keys(), key=lambda k: models_trained[k]['metrics'].get('Accuracy', float('-inf')))

        results['models'] = models_trained
        results['best_model'] = best_model_name

        return results, f"Model training completed successfully. Best model based on {'RMSE' if problem_type=='regression' else 'Accuracy'}: {best_model_name}"

    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        st.error(traceback.format_exc())
        return None, f"Error during model training: {str(e)}"


# --- Clustering Function ---
# (Keep the perform_clustering function as provided, it's suitable)
def perform_clustering(df: pd.DataFrame, n_clusters: int = 3, features: Optional[List[str]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """Perform K-means clustering on the dataset"""
    if df is None:
        return None, "Dataset not available."

    results = {'parameters': {'n_clusters': n_clusters, 'features': features}}
    preprocessing_steps = []

    try:
        # Select features or use all numeric columns
        if features:
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                return None, f"Specified features not found: {', '.join(missing_features)}"
            numeric_df = df[features].copy()
             # Check if selected features are numeric
            non_numeric_selected = numeric_df.select_dtypes(exclude=np.number).columns
            if not non_numeric_selected.empty:
                 return None, f"Clustering requires numeric features. Non-numeric columns selected: {', '.join(non_numeric_selected)}"
        else:
            numeric_df = df.select_dtypes(include=np.number).copy()

        if numeric_df.empty:
            return None, "No numeric columns available for clustering. Please select numeric features or ensure the dataframe has numeric columns."

        results['features_used'] = numeric_df.columns.tolist()

        # Handle missing values
        if numeric_df.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='median') # Use median
            X = imputer.fit_transform(numeric_df)
            preprocessing_steps.append("Applied median imputation for missing values.")
        else:
            X = numeric_df.values # Use values directly if no NA
            preprocessing_steps.append("No missing values found in selected features.")

        # Scale the data (important for K-Means)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        preprocessing_steps.append("Applied StandardScaler to features.")
        results['preprocessing_steps'] = preprocessing_steps

        # Perform K-means clustering
        if n_clusters <= 1:
            return None, "Number of clusters (n_clusters) must be greater than 1."
        if n_clusters >= len(X_scaled):
             return None, f"Number of clusters ({n_clusters}) cannot be equal to or greater than the number of samples ({len(X_scaled)})."

        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42) # Improved initialization
        clusters = kmeans.fit_predict(X_scaled)

        # Add clusters to the original dataframe (or a copy with selected features)
        # Create a result df containing original index, cluster labels, and potentially scaled features
        clustered_data_with_labels = pd.DataFrame(X_scaled, columns=numeric_df.columns, index=numeric_df.index)
        clustered_data_with_labels['cluster'] = clusters
        # Maybe also include original non-scaled numeric features for easier interpretation later?
        # result_df = df.loc[numeric_df.index].copy() # Get original rows corresponding to numeric_df
        # result_df['cluster'] = clusters

        # Get cluster centers (in original scale)
        centers_scaled = kmeans.cluster_centers_
        centers_original = scaler.inverse_transform(centers_scaled)

        # Calculate cluster statistics
        cluster_stats = {}
        cluster_sizes = pd.Series(clusters).value_counts().sort_index()
        total_samples = len(clusters)

        for i in range(n_clusters):
            cluster_label = f'Cluster {i}'
            size = cluster_sizes.get(i, 0) # Handle potential empty clusters if k is too large? (unlikely with k-means++)
            percentage = round(size / total_samples * 100, 2) if total_samples > 0 else 0

            # Get stats for original numeric data within this cluster
            cluster_members_original = numeric_df[clusters == i]
            numeric_summary = {}
            if not cluster_members_original.empty:
                 numeric_summary = cluster_members_original.describe().round(2).to_dict()


            cluster_stats[cluster_label] = {
                'size': size,
                'percentage': percentage,
                'center_coordinates (original_scale)': {col: centers_original[i, j] for j, col in enumerate(numeric_df.columns)},
                'member_statistics (original_scale)': numeric_summary
                # 'center_coordinates (scaled)': {col: centers_scaled[i, j] for j, col in enumerate(numeric_df.columns)} # Optional: include scaled centers
            }

        results['cluster_assignments'] = clustered_data_with_labels # DF with scaled data + labels
        # results['clustered_dataframe_original_features'] = result_df # DF with original data + labels
        results['cluster_summary'] = cluster_stats
        results['inertia'] = kmeans.inertia_ # Sum of squared distances of samples to their closest cluster center

        return results, f"K-Means Clustering completed successfully with {n_clusters} clusters."

    except Exception as e:
        st.error(f"Error performing clustering: {str(e)}")
        st.error(traceback.format_exc())
        return None, f"Error performing clustering: {str(e)}"

# --- Python Code Execution Function ---
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


# --- Initialize Agent ---
# Call this after LLM setup and potentially after data loading to inject context
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