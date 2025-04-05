# data_utils.py
import pandas as pd
import numpy as np
import io
from typing import Dict, Any, Optional

# Function to generate data profile (modified to remove Streamlit dependencies)
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

def generate_data_profile(df: pd.DataFrame, file_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Generates a dictionary containing profiling information about the dataframe."""
    if df is None:
        return None
    try:
        profile = {
            "file_name": file_name or 'N/A',
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
                    "unique_values": counts.nunique(), # Corrected: use nunique() on the Series
                    "top_5_values": counts.head(5).to_dict()
                }
                profile["categorical_summary"][col] = summary
            except Exception as e:
                profile["categorical_summary"][col] = {"error": f"Could not compute summary: {str(e)}"}

        # Convert numpy types for better serialization if needed later
        profile = convert_numpy_types(profile)
        return profile
    except Exception as e:
        print(f"Error generating data profile: {e}") # Log error instead of using st.error
        return None

def load_dataframe(uploaded_file: io.BytesIO, file_name: str) -> Optional[pd.DataFrame]:
    """Loads a dataframe from an uploaded file stream."""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            print(f"Unsupported file type: {file_name}")
            return None
        print(f"Successfully loaded dataframe from {file_name}")
        return df
    except Exception as e:
        print(f"Error loading file {file_name}: {str(e)}")
        return None