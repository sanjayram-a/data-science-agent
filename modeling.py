# modeling.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Scikit-learn imports
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def train_model(df: pd.DataFrame, target_col: str, model_type: str = 'auto', features: Optional[List[str]] = None, test_size: float = 0.2, random_state: int = 42) -> Tuple[Optional[Dict[str, Any]], str]:
    """Trains a regression or classification model."""
    if df is None:
        return None, "Error: No dataset loaded."
    if target_col not in df.columns:
        return None, f"Error: Target column '{target_col}' not found in the dataframe."

    df_processed = df.copy()

    # --- Feature Selection ---
    if features:
        # Ensure target is not accidentally included in features
        features = [f for f in features if f != target_col]
        # Check if all selected features exist
        missing_features = [f for f in features if f not in df_processed.columns]
        if missing_features:
            return None, f"Error: The following specified features are missing: {', '.join(missing_features)}"
        # Check if features list is empty after removing target
        if not features:
             return None, "Error: No valid features selected (features list was empty or only contained the target column)."
        X = df_processed[features]
    else:
        # Use all columns except the target
        X = df_processed.drop(columns=[target_col])
        features = X.columns.tolist() # Update features list

    if X.empty:
        return None, "Error: No features available for training after selecting/excluding the target column."

    y = df_processed[target_col]

    # --- Determine Model Type ---
    task = None
    if model_type == 'auto':
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15: # Heuristic for regression
            task = 'regression'
            print(f"Auto-detected task: Regression (target '{target_col}' is numeric with >15 unique values)")
        elif pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or y.nunique() <= 15: # Heuristic for classification
            task = 'classification'
            print(f"Auto-detected task: Classification (target '{target_col}' is categorical or has <=15 unique values)")
        else:
            return None, f"Error: Could not automatically determine model type for target '{target_col}'. Please specify 'regression' or 'classification'."
    elif model_type.lower() in ['regression', 'classification']:
        task = model_type.lower()
        print(f"User-specified task: {task.capitalize()}")
    else:
        return None, f"Error: Invalid model_type '{model_type}'. Choose 'auto', 'regression', or 'classification'."

    # --- Preprocessing ---
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Create preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Use median for robustness
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Use most frequent for categorical
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Use sparse_output=False for easier handling downstream
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any) - though usually features are specified
    )

    # --- Model Selection ---
    if task == 'regression':
        # Try RandomForest first, fallback to LinearRegression
        try:
            model = RandomForestRegressor(random_state=random_state, n_estimators=100) # Default n_estimators
            model_name = "RandomForestRegressor"
        except:
            model = LinearRegression()
            model_name = "LinearRegression"
        metrics_func = {
            "R-squared": r2_score,
            "Mean Absolute Error (MAE)": mean_absolute_error,
            "Mean Squared Error (MSE)": mean_squared_error,
            "Root Mean Squared Error (RMSE)": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
        }
        # Label Encode target if it's object/categorical but represents numbers
        if not pd.api.types.is_numeric_dtype(y):
             try:
                 y = pd.to_numeric(y)
                 print(f"Converted target column '{target_col}' to numeric for regression.")
             except ValueError:
                 return None, f"Error: Target column '{target_col}' is not numeric and could not be converted for regression."

    else: # Classification
        # Label Encode the target variable if it's not already numeric/encoded
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            target_classes = le.classes_
            print(f"Encoded target column '{target_col}'. Classes: {target_classes}")
        else:
            target_classes = np.unique(y) # Get classes from numeric target

        # Try RandomForest first, fallback to LogisticRegression
        try:
            model = RandomForestClassifier(random_state=random_state, n_estimators=100)
            model_name = "RandomForestClassifier"
        except:
            model = LogisticRegression(random_state=random_state, max_iter=1000) # Increase max_iter
            model_name = "LogisticRegression"

        metrics_func = {
            "Accuracy": accuracy_score,
            "Precision": lambda yt, yp: precision_score(yt, yp, average='weighted', zero_division=0), # Use weighted avg, handle zero division
            "Recall": lambda yt, yp: recall_score(yt, yp, average='weighted', zero_division=0),
            "F1-Score": lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0)
        }

    # --- Train/Test Split ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if task == 'classification' and len(np.unique(y)) > 1 else None)
    except ValueError as e:
         # Handle potential stratify error if a class has only 1 member
         if "The least populated class" in str(e):
             print(f"Warning: Could not stratify train/test split due to small class size. Splitting without stratification. Error: {e}")
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
         else:
             return None, f"Error during train/test split: {e}"

    # --- Create and Train Pipeline ---
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    try:
        print(f"Training {model_name}...")
        pipeline.fit(X_train, y_train)
        print("Training complete.")
    except Exception as e:
        return None, f"Error during model training: {e}"

    # --- Evaluation ---
    try:
        print("Evaluating model...")
        y_pred = pipeline.predict(X_test)
        metrics = {name: func(y_test, y_pred) for name, func in metrics_func.items()}

        results = {
            "model_name": model_name,
            "task": task,
            "features_used": features,
            "target_column": target_col,
            "test_size": test_size,
            "metrics": metrics,
            "model_pipeline": pipeline # Return the trained pipeline
        }

        if task == 'classification':
            try:
                # Ensure labels are consistent if LabelEncoder was used
                labels = target_classes if 'target_classes' in locals() else np.unique(np.concatenate((y_test, y_pred)))
                results["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=labels).tolist() # Convert to list for JSON compatibility
                results["class_labels"] = labels.tolist() # Store labels used for confusion matrix
            except Exception as cm_e:
                print(f"Warning: Could not generate confusion matrix: {cm_e}")
                results["confusion_matrix"] = None
                results["class_labels"] = None


        # Convert metrics to native types
        results["metrics"] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in results["metrics"].items()}

        message = f"Successfully trained {model_name} for {task} on target '{target_col}'.\nMetrics: {results['metrics']}"
        print(message)
        return results, message

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error during model evaluation: {e}"