# visualization.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

def create_plot(df: pd.DataFrame, plot_type: str, x_col: str, y_col: Optional[str] = None, color_by: Optional[str] = None, title: Optional[str] = None, **kwargs) -> Optional[go.Figure]:
    """Create a plotly plot based on specified parameters"""
    if df is None:
        print(f"Warning: Cannot create plot. Dataframe is missing.")
        return None
    if x_col not in df.columns:
        print(f"Warning: Cannot create plot. Column '{x_col}' not found.")
        return None
    if y_col and y_col not in df.columns:
        print(f"Warning: Cannot create plot. Column '{y_col}' not found.")
        return None
    if color_by and color_by not in df.columns:
        print(f"Warning: Color-by column '{color_by}' not found. Ignoring color_by.")
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
                print("Warning: Scatter plot requires both X and Y columns.")
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
                             print(f"Warning: Cannot perform '{agg_func}' on non-numeric column '{y_col}'. Using count instead.")
                             temp_df = df.groupby(x_col).size().reset_index(name='count')
                             y_col = 'count' # Update y_col to 'count'
                except Exception as agg_e:
                     print(f"Error: Could not aggregate '{y_col}' by '{x_col}' using '{agg_func}': {agg_e}. Using count instead.")
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
                    print(f"Warning: Line plot requires a numeric or datetime X-axis. '{x_col}' is {df[x_col].dtype}.")
                    return None
            else:
                print("Warning: Line plot requires both X and Y columns.")
                return None
        elif plot_type == "heatmap":
            # Calculate correlation matrix for numeric columns
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.shape[1] < 2:
                 print("Warning: Heatmap requires at least two numeric columns.")
                 return None
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title=f"{base_title}: Correlation Heatmap", **kwargs)
        elif plot_type == "pie":
            # Use value counts of the x_col
            counts = df[x_col].value_counts()
            fig = px.pie(values=counts.values, names=counts.index, title=f"{base_title}: Distribution of {x_col}", **kwargs)
        else:
            print(f"Error: Unsupported plot type '{plot_type}'. Supported types: histogram, scatter, bar, box, line, heatmap, pie.")
            return None

        # Update layout for clarity
        if fig:
            fig.update_layout(title_x=0.5) # Center title

        return fig

    except Exception as e:
        print(f"Error creating {plot_type} plot: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return None