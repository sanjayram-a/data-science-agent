import pandas as pd
import streamlit as st
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import plotly.express as px
import plotly.graph_objects as go
import traceback # For better error logging


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