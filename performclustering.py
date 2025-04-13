import pandas as pd
import streamlit as st
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
import time
import traceback


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
