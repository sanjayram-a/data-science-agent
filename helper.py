import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple

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