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
