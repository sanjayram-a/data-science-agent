# clustering.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

def perform_clustering(df: pd.DataFrame, n_clusters: int = 3, features: Optional[List[str]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """Performs K-Means clustering on the dataset."""
    if df is None:
        return None, "Error: No dataset loaded."
    if n_clusters <= 1:
        return None, "Error: Number of clusters must be greater than 1."

    df_processed = df.copy()

    # --- Feature Selection ---
    if features:
        # Check if all selected features exist
        missing_features = [f for f in features if f not in df_processed.columns]
        if missing_features:
            return None, f"Error: The following specified features are missing: {', '.join(missing_features)}"
        X = df_processed[features]
    else:
        # Use only numeric features by default if none are specified
        X = df_processed.select_dtypes(include=np.number)
        features = X.columns.tolist() # Update features list
        if not features:
             # If no numeric features, try using all (will require encoding)
             print("Warning: No numeric features found for clustering. Attempting to use all features with encoding.")
             X = df_processed
             features = X.columns.tolist()


    if X.empty:
        return None, "Error: No features available for clustering."
    if len(X) < n_clusters:
        return None, f"Error: Number of samples ({len(X)}) is less than the number of clusters ({n_clusters})."

    print(f"Performing K-Means clustering with {n_clusters} clusters on features: {', '.join(features)}")

    # --- Preprocessing (similar to modeling, handle numeric and categorical) ---
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # --- Clustering Pipeline ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Explicitly set n_init

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('kmeans', kmeans)])

    # --- Fit and Predict ---
    try:
        print("Fitting K-Means model...")
        # Fit the pipeline and get cluster labels
        pipeline.fit(X)
        cluster_labels = pipeline.predict(X)
        print("Clustering complete.")

        # Add cluster labels back to the original (copied) dataframe
        clustered_df = df.copy() # Use the original df structure
        clustered_df['cluster'] = cluster_labels

        # Calculate cluster centers (need to inverse transform or use transformed data)
        # Getting meaningful centers after one-hot encoding can be complex.
        # We'll report basic info for now.
        cluster_centers = None
        try:
             # Access the KMeans model within the pipeline
             kmeans_model = pipeline.named_steps['kmeans']
             # The centers are in the transformed space
             transformed_centers = kmeans_model.cluster_centers_
             # Getting back to original feature space is non-trivial with OneHotEncoder.
             # For simplicity, we won't try to inverse transform here.
             # We can report the size of each cluster instead.
             print("Cluster centers are in the transformed feature space.")
        except Exception as center_e:
             print(f"Warning: Could not retrieve cluster centers: {center_e}")


        # Analyze cluster sizes
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index().to_dict()
        print(f"Cluster sizes: {cluster_sizes}")

        results = {
            "n_clusters": n_clusters,
            "features_used": features,
            "cluster_labels": cluster_labels.tolist(), # Add labels as a list
            "cluster_sizes": cluster_sizes,
            "clustered_dataframe_sample": clustered_df.head().to_dict('records'), # Sample for verification
            "clustering_pipeline": pipeline # Return the pipeline
        }

        message = f"Successfully performed K-Means clustering with {n_clusters} clusters.\nCluster sizes: {cluster_sizes}"
        return results, message

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error during clustering: {e}"