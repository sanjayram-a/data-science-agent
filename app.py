# app.py
import os
import io
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for

# Import modules
import state
from data_utils import load_dataframe, generate_data_profile, convert_numpy_types
from llm_utils import configure_gemini, get_configured_models
# Agent setup
from agent_setup import initialize_agent, get_agent
import traceback # For detailed error logging

# Configure Flask app
app = Flask(__name__)
# It's good practice to set a secret key for session management (even if not heavily used yet)
app.secret_key = os.urandom(24)
# Configure upload folder (optional, if saving uploads)
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# --- Routes ---

# --- Page Routes ---

def _get_common_context():
    """Helper to get context needed for base template and pages."""
    summary = state.get_full_state_summary()
    genai_model, llama_llm = get_configured_models()
    api_configured = genai_model is not None and llama_llm is not None
    return {"summary": summary, "api_configured": api_configured}

@app.route('/')
def index():
    """Renders the overview/upload page."""
    context = _get_common_context()
    return render_template('overview.html', **context)

@app.route('/chat-page')
def chat_page():
    """Renders the agent chat page."""
    context = _get_common_context()
    return render_template('chat.html', **context)

@app.route('/cleaning')
def cleaning_page():
    """Renders the data cleaning page."""
    context = _get_common_context()
    return render_template('cleaning.html', **context)

@app.route('/modeling')
def modeling_page():
    """Renders the modeling page."""
    context = _get_common_context()
    return render_template('modeling.html', **context)

@app.route('/charts')
def charts_page():
    """Renders the chart analysis page."""
    context = _get_common_context()
    # Assuming state.py has a function to get the last plot filename
    last_plot_filename = state.get_last_plot_filename() # Need to implement this in state.py
    return render_template('charts.html', **context, last_plot_filename=last_plot_filename)

# --- Data Cleaning API Routes ---
@app.route('/api/clean', methods=['POST'])
def clean_data():
    """Handles data cleaning operations."""
    if not state.has_current_df():
        return jsonify({"error": "No data loaded."}), 400
    
    data = request.get_json()
    operation = data.get('operation')
    
    try:
        df = state.get_current_df()
        if operation == 'missing':
            # Handle missing values
            df = df.fillna(df.mean(numeric_only=True))
        elif operation == 'duplicates':
            # Remove duplicates
            df = df.drop_duplicates()
        elif operation == 'outliers':
            # Simple outlier removal using IQR
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        elif operation == 'categorical':
            # Encode categorical columns
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = pd.Categorical(df[col]).codes
        
        # Update state with cleaned dataframe
        state.set_current_df(df, name=state.get_current_df_name(), is_cleaned=True)
        
        return jsonify({
            "success": True,
            "message": f"Successfully applied {operation} cleaning operation",
            "rows": len(df),
            "columns": len(df.columns)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/column-operation', methods=['POST'])
def column_operation():
    """Handles column-specific operations."""
    if not state.has_current_df():
        return jsonify({"error": "No data loaded."}), 400
    
    data = request.get_json()
    column = data.get('column')
    operation = data.get('operation')
    params = data.get('params', {})
    
    try:
        df = state.get_current_df()
        
        if operation == 'fillna':
            method = params.get('method')
            if method == 'mean':
                df[column] = df[column].fillna(df[column].mean())
            elif method == 'median':
                df[column] = df[column].fillna(df[column].median())
            elif method == 'mode':
                df[column] = df[column].fillna(df[column].mode()[0])
            elif method == 'value':
                df[column] = df[column].fillna(params.get('value'))
        
        elif operation == 'drop':
            df = df.drop(columns=[column])
        
        elif operation == 'rename':
            new_name = params.get('new_name')
            df = df.rename(columns={column: new_name})
        
        elif operation == 'type':
            new_type = params.get('new_type')
            df[column] = df[column].astype(new_type)
        
        # Update state with modified dataframe
        state.set_current_df(df, name=state.get_current_df_name(), is_cleaned=True)
        
        return jsonify({
            "success": True,
            "message": f"Successfully applied {operation} to column {column}"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Modeling API Routes ---
@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Handles model training requests."""
    if not state.has_current_df():
        return jsonify({"error": "No data loaded."}), 400
    
    data = request.get_json()
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        df = state.get_current_df()
        target = data.get('target')
        split_ratio = data.get('split_ratio', 0.8)
        
        # Prepare data
        X = df.drop(columns=[target])
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_ratio)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model based on problem type
        problem_type = data.get('problem_type')
        model_type = data.get('model_type')
        params = data.get('parameters', {})
        
        if problem_type == 'regression':
            from sklearn.metrics import mean_squared_error, r2_score
            if model_type == 'linear':
                from sklearn.linear_model import LinearRegression
                model = LinearRegression(**params)
            elif model_type == 'ridge':
                from sklearn.linear_model import Ridge
                model = Ridge(**params)
            elif model_type == 'rf_regressor':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(**params)
        
        elif problem_type == 'classification':
            from sklearn.metrics import accuracy_score, classification_report
            if model_type == 'logistic':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(**params)
            elif model_type == 'rf_classifier':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {}
        if problem_type == 'regression':
            metrics['r2_score'] = r2_score(y_test, y_pred)
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
        else:
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics.update({
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score']
            })
        
        # Store model in state
        state.set_current_model({
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'feature_names': X.columns.tolist()
        })
        
        return jsonify({
            "success": True,
            "metrics": metrics
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- API/Action Routes ---

@app.route('/configure_api', methods=['POST'])
def configure_api_route():
    """Handles API key configuration."""
    api_key = request.form.get('gemini_api_key')
    if not api_key:
        flash('Gemini API Key is required.', 'error')
        return redirect(url_for('index'))

    genai_model, llama_llm = configure_gemini(api_key)

    if genai_model and llama_llm:
        flash('Gemini API Key configured successfully!', 'success')
        # Initialize agent after successful API configuration
        print("Attempting to initialize agent...")
        agent = initialize_agent()
        if agent:
            print("Agent initialized successfully after API config.")
            flash('Agent initialized successfully!', 'info')
        else:
            print("Agent initialization failed after API config.")
            flash('API Key configured, but agent initialization failed. Check logs.', 'warning')
    else:
        flash('Failed to configure Gemini API Key. Check the key and try again.', 'error')

    return redirect(url_for('index'))


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and initial processing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file:
        try:
            filename = file.filename
            print(f"Received file: {filename}")
            # Read file content into memory
            file_stream = io.BytesIO(file.read())

            # Load dataframe
            df = load_dataframe(file_stream, filename)
            file_stream.close() # Close the stream

            if df is None:
                 return jsonify({"error": f"Failed to load dataframe from {filename}. Unsupported format or file error."}), 400

            # Generate profile
            profile = generate_data_profile(df, filename)

            if profile is None:
                 # Set state even if profile fails, so user knows data is loaded
                 state.set_current_df(df, name=filename, is_cleaned=False, profile=None)
                 return jsonify({"error": "Dataframe loaded, but failed to generate data profile.", "file_loaded": True, "profile": None}), 500

            # Update application state
            state.set_current_df(df, name=filename, is_cleaned=False, profile=profile)
            print("State updated with new dataframe and profile.")

            # Return profile (ensure it's JSON serializable)
            # The profile should already be converted by generate_data_profile
            return jsonify({"success": True, "profile": profile, "filename": filename})

        except Exception as e:
            print(f"Error during file upload processing: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"An unexpected error occurred during file processing: {str(e)}"}), 500

    return jsonify({"error": "File upload failed."}), 400

# --- Chat API Route ---
@app.route('/chat', methods=['POST'])
def chat_handler():
    """Handles chat messages sent to the agent."""
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    agent = get_agent()
    if agent is None:
         # Try to initialize again if not already done
         agent = initialize_agent()
         if agent is None:
              return jsonify({"error": "Agent not initialized. Please configure API Key first."}), 503 # Service Unavailable

    print(f"Received chat message: {user_message}")
    try:
        # Use agent.chat() for conversation
        # Note: For complex state changes triggered by tools, the agent's response
        # string might need parsing, or we might need more sophisticated state checking.
        # The current tool wrappers return strings summarizing actions.
        response = agent.chat(user_message)

        # Extract the primary text response from the agent
        reply_text = "Error processing agent response." # Default error message
        if hasattr(response, 'response'):
             reply_text = response.response
        else:
             reply_text = str(response) # Fallback

        print(f"Agent raw response text: {reply_text}")

        # --- Check if a plot was generated and update state ---
        # This logic relies on the visualization tool saving a plot and potentially
        # mentioning the filename in the reply_text, or updating state directly.
        # We also need to ensure the plot is moved to the static folder.
        # Let's refine the check based on state potentially set by the tool wrapper.

        # Check if the visualization tool updated the state with a new plot figure/filename
        # We need a reliable way to know if a plot was just created.
        # Assuming the tool wrapper calls state.set_last_plot_fig(fig, filename)
        # And we have a way to get just the filename.

        # Refined plot check:
        # Check state for the latest plot filename potentially set by the tool
        potential_plot_filename = state.get_last_plot_filename() # Get the filename stored in state

        if potential_plot_filename:
            # Check if this filename is mentioned in the current reply (heuristic)
            # Or, better: Assume if state.get_last_plot_filename() returns a value,
            # it corresponds to the *latest* operation.
            # We need to ensure the file exists and move it to static.
            plot_src_path = potential_plot_filename # Path where the tool saved it
            plot_dest_folder = 'static'
            plot_dest_path = os.path.join(plot_dest_folder, os.path.basename(plot_src_path))

            if os.path.exists(plot_src_path):
                try:
                    # Ensure static directory exists
                    if not os.path.exists(plot_dest_folder):
                        os.makedirs(plot_dest_folder)
                    # Move the plot file
                    os.rename(plot_src_path, plot_dest_path)
                    print(f"Moved plot {os.path.basename(plot_src_path)} to static folder.")
                    # Update state with the *relative* path within static for URL generation
                    state.set_last_plot_filename(os.path.basename(plot_src_path))
                except Exception as move_e:
                    print(f"Error moving plot {os.path.basename(plot_src_path)} to static folder: {move_e}")
                    # Clear the plot filename from state if move failed? Or keep original? Clearing for now.
                    state.set_last_plot_filename(None)
            else:
                 print(f"Plot file {plot_src_path} mentioned or stored in state not found.")
                 # Clear the plot filename from state if file doesn't exist
                 state.set_last_plot_filename(None)

        # --- Check for other artifacts (like dataframes) if needed for future pages ---
        # (Code for checking last_exec can remain if needed for other state updates,
        # but exec_output_info is not returned in JSON anymore)
        last_exec = state.get_last_executed_code_output()
        if last_exec and "df_output" in last_exec and isinstance(last_exec["df_output"], pd.DataFrame):
             if "[DataFrame 'df_result' Captured]" in reply_text:
                 print("Detected captured DataFrame 'df_result'. State updated.")
                 # No need to generate HTML here, just acknowledge state update.


        # Return ONLY the agent's text reply
        # The frontend will direct users to other pages for artifacts
        return jsonify({
            "reply": reply_text
        })

    except Exception as e:
        print(f"Error during agent chat: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An error occurred while processing your request: {str(e)}"}), 500

# --- Data Analysis API Routes ---
@app.route('/api/dataset-stats')
def get_dataset_stats():
    """Get dataset statistics including missing values and duplicates."""
    if not state.has_current_df():
        return jsonify({"error": "No data loaded"}), 400
    
    try:
        df = state.get_current_df()
        stats = {
            "missing_values": {
                "total": int(df.isnull().sum().sum()),
                "by_column": df.isnull().sum().to_dict(),
                "percentage": float((df.isnull().sum().sum() / (df.size)) * 100)
            },
            "duplicates": {
                "total": int(df.duplicated().sum()),
                "percentage": float((df.duplicated().sum() / len(df)) * 100)
            },
            "rows": len(df),
            "columns": len(df.columns)
        }
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/preview')
def get_data_preview():
    """Get data preview with HTML table."""
    if not state.has_current_df():
        return jsonify({"error": "No data loaded"}), 400
    
    try:
        df = state.get_current_df()
        preview_html = df.head(10).to_html(classes='table table-striped table-hover')
        stats = {
            "missing_values": f"{df.isnull().sum().sum()} cells ({(df.isnull().sum().sum() / (df.size)) * 100:.1f}%)",
            "duplicates": f"{df.duplicated().sum()} rows ({(df.duplicated().sum() / len(df)) * 100:.1f}%)"
        }
        return jsonify({
            "success": True,
            "html": preview_html,
            "stats": stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/columns')
def get_columns():
    """Get column information for model configuration."""
    print("API: /api/columns endpoint called")
    
    if not state.has_current_df():
        print("API: No data found in state.has_current_df()")
        return jsonify({"error": "No data loaded"}), 400
    
    try:
        df = state.get_current_df()
        print(f"API: Successfully retrieved dataframe with shape {df.shape}")
        
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            missing_count = df[col].isnull().sum()
            columns.append({
                "name": col,
                "dtype": dtype,
                "unique_count": int(unique_count),
                "missing_count": int(missing_count),
                "is_numeric": pd.api.types.is_numeric_dtype(df[col])
            })
        print(f"API: Returning {len(columns)} columns")
        return jsonify({
            "success": True,
            "columns": columns
        })
    except Exception as e:
        print(f"API: Error in get_columns: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/model-info')
def get_model_info():
    """Get model training information and dataset splits."""
    if not state.has_current_df():
        return jsonify({"error": "No data loaded"}), 400
    
    try:
        df = state.get_current_df()
        total_rows = len(df)
        train_size = int(total_rows * 0.8)  # Default 80% train
        test_size = total_rows - train_size
        
        return jsonify({
            "success": True,
            "info": {
                "total_rows": total_rows,
                "train_size": train_size,
                "test_size": test_size,
                "feature_count": len(df.columns),
                "numeric_features": len(df.select_dtypes(include=['int64', 'float64']).columns),
                "categorical_features": len(df.select_dtypes(include=['object', 'category']).columns)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Set debug=True for development (auto-reloads, detailed errors)
    # Set use_reloader=False if it causes issues with background processes or state
    app.run(debug=True, use_reloader=True)