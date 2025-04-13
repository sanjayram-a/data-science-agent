import pandas as pd
import streamlit as st
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

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
