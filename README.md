# Data Science Agent Web Application

## Overview

This Data Science Agent is a powerful interactive web application built with Flask that provides an intuitive interface for comprehensive data analysis tasks. The application integrates AI capabilities (powered by Google Gemini and LlamaIndex) to enable natural language interactions for data exploration, visualization, cleaning, modeling, and advanced analytics. It's designed for data scientists, analysts, and non-technical users who want to perform data science tasks through a user-friendly interface.

## Key Features

### Data Management
* **Multi-format Support**: Upload datasets in CSV, Excel, and other tabular formats
* **Data Profiling**: Automatic generation of comprehensive data profiles with statistics and visualizations
* **State Management**: Persistent state tracking for datasets, models, and visualizations across the application

### Interactive UI Pages
* **Overview Dashboard**: Central hub for data upload and summary statistics
* **Data Cleaning Interface**: Specialized tools for handling missing values, duplicates, outliers, and data type conversions
* **Modeling Workspace**: Configure, train, and evaluate machine learning models 
* **Analytics Dashboard**: Create, customize, and manage data visualizations
* **AI Chat Interface**: Natural language interaction for complex data tasks

### AI Capabilities
* **Natural Language Processing**: Chat interface with support for data-related queries and commands
* **Intelligent Tools**: Agent-based architecture with specialized tools for:
  * Data exploration and profiling
  * Automated data cleaning and preprocessing
  * Statistical analysis and visualization
  * Machine learning model training and evaluation
  * Clustering and dimensionality reduction
  * Custom Python code execution for advanced operations

### Visualization & Reporting
* **Interactive Charts**: Dynamic chart creation with customization options
* **Model Evaluation**: Comprehensive metrics, confusion matrices, and performance visualizations
* **Results Export**: Download capabilities for analyses, models, and visualizations

## Setup & Installation

### Prerequisites
* Python 3.8+ installed
* Pip package manager
* Google Gemini API key (obtain from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   
   # Activation on Windows
   .\venv\Scripts\activate
   
   # Activation on macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**
   
   **Option 1: Environment File (Recommended)**
   - Create a `.env` file in the project root
   - Add your API key: `GEMINI_API_KEY=your_api_key_here`
   
   **Option 2: Web UI Configuration**
   - Start the application and configure through the UI

5. **Launch the Application**
   ```bash
   python app.py
   ```
   
   The application will be available at `http://127.0.0.1:5000` by default

## Application Architecture

### Core Components

1. **Flask Backend (`app.py`)**
   - Manages HTTP routes and API endpoints
   - Handles file uploads and state management
   - Renders UI templates and serves static assets

2. **State Management (`state.py`)**
   - Maintains application state between requests
   - Tracks current dataframe, models, and visualizations
   - Provides utility functions for state access and modification

3. **AI Agent System**
   - **Setup (`agent_setup.py`)**: Initializes and configures the ReActAgent
   - **Tools (`agent_tools.py`)**: Defines the specialized tools for data operations
   - **LLM Integration (`llm_utils.py`)**: Manages communication with Google Gemini

4. **Specialized Modules**
   - **Data Utils (`data_utils.py`)**: Functions for loading and profiling data
   - **Cleaning (`cleaning.py`)**: Data cleaning and preprocessing operations
   - **Visualization (`visualization.py`)**: Chart generation and plotting functions
   - **Modeling (`modeling.py`)**: ML model training and evaluation
   - **Clustering (`clustering.py`)**: Clustering and dimensionality reduction
   - **Code Execution (`code_executor.py`)**: Safe Python code execution environment

### UI Pages

1. **Overview Page (`overview.html`)**
   - Landing page with upload functionality
   - Dataset summary and initial statistics
   - Quick actions for common tasks

2. **Cleaning Page (`cleaning.html`)**
   - Tools for handling missing values
   - Duplicate detection and removal
   - Outlier detection and handling
   - Data type conversion and transformation

3. **Modeling Page (`modeling.html`)**
   - Feature selection interface
   - Model type selection (classification, regression, clustering)
   - Hyperparameter configuration
   - Training and evaluation workflow
   - Model comparison tools

4. **Analytics Dashboard (`charts.html`)**
   - Chart creation interface
   - Visualization gallery
   - Export and sharing options

5. **Chat Interface (`chat.html`)**
   - Natural language interaction
   - Command history and suggestions
   - Results display area for code, charts, and tables

## API Reference

### Data Management Endpoints

**`POST /upload`**
- **Purpose**: Upload and process data files
- **Request**: Multipart form with file attachment
- **Response**: JSON with profile summary and status

**`GET /api/dataset-stats`**
- **Purpose**: Retrieve dataset statistics
- **Response**: JSON with detailed statistics on missing values, duplicates, etc.

**`GET /api/preview`**
- **Purpose**: Get a preview of current data
- **Response**: JSON with HTML table and basic stats

**`GET /api/columns`**
- **Purpose**: Get column information for the current dataset
- **Response**: JSON with column names, types, and characteristics

### Cleaning Endpoints

**`POST /api/clean`**
- **Purpose**: Apply bulk cleaning operations
- **Request**: JSON with operation type
- **Response**: JSON with operation status and updated row/column counts

**`POST /api/column-operation`**
- **Purpose**: Apply operations to specific columns
- **Request**: JSON with column name, operation type, and parameters
- **Response**: JSON with operation status

### Modeling Endpoints

**`GET /api/model-info`**
- **Purpose**: Get information for model training
- **Response**: JSON with dataset splits and feature information

**`POST /api/train-model`**
- **Purpose**: Train a machine learning model
- **Request**: JSON with model configuration
- **Response**: JSON with metrics and results

### Chat and Agent Endpoints

**`POST /chat`**
- **Purpose**: Process user messages through the AI agent
- **Request**: JSON with message text
- **Response**: JSON with agent reply and any generated artifacts

**`POST /configure_api`**
- **Purpose**: Configure the Gemini API key
- **Request**: Form data with API key
- **Response**: Redirect with status message

## Usage Guide

### Step 1: Data Upload and Exploration

1. Navigate to the **Overview** page
2. Use the upload form to select and upload your dataset
3. Review the automatically generated data profile
4. Use the quick action buttons or navigate to specific pages for detailed operations

### Step 2: Data Cleaning

1. Navigate to the **Cleaning** page
2. Use the quick action buttons for common cleaning operations:
   - Handle missing values (using mean, median, mode, or custom values)
   - Remove duplicate rows
   - Handle outliers using IQR or z-score methods
   - Encode categorical variables
3. For column-specific operations:
   - Select a column from the dropdown
   - Choose an operation (fill missing values, drop, rename, change type)
   - Configure operation parameters
   - Click "Apply" to execute

### Step 3: Data Visualization

1. Navigate to the **Charts** page
2. Quick visualizations:
   - Distribution plots for numeric columns
   - Correlation matrix for numeric columns
   - Box plots for numeric columns
   - Scatter plot matrices
3. Custom chart creation:
   - Select chart type (bar, line, scatter, histogram, box, pie)
   - Select X and Y axis variables
   - Click "Generate Chart" to create the visualization
4. View, download, or delete generated charts in the gallery

### Step 4: Model Training and Evaluation

1. Navigate to the **Modeling** page
2. Configure your model:
   - Select problem type (regression, classification, clustering)
   - Select model algorithm (e.g., Linear Regression, Random Forest, etc.)
   - Choose target variable
   - Select features to include
   - Adjust train/test split percentage
3. Set model-specific parameters in the "Model Parameters" section
4. Click "Train Model" to start the training process
5. Review results:
   - Performance metrics
   - Feature importance
   - Predictions preview
6. Compare models or download trained models using the buttons provided

### Step 5: Using the AI Chat Interface

1. Navigate to the **Chat** page
2. Enter natural language queries or commands, such as:
   - "Show me the first 10 rows of the dataset"
   - "Create a histogram of the age column"
   - "What's the correlation between income and education?"
   - "Train a random forest classifier to predict customer churn"
   - "Identify clusters in my data using DBSCAN"
   - "Clean my dataset by removing rows with missing values"
3. View the agent's response and any generated outputs

## Advanced Usage

### Custom Python Code Execution

The agent can execute custom Python code for specialized analysis:

```
Execute this code:
import pandas as pd
import matplotlib.pyplot as plt

# Get the current dataframe
df = get_current_df()

# Calculate aggregate statistics
agg_stats = df.groupby('category').agg({
    'numeric_col1': ['mean', 'std'],
    'numeric_col2': ['min', 'max']
})

print(agg_stats)
```

### Batch Operations via Chat

Complex sequences of operations can be requested through chat:

```
Please do the following:
1. Clean the data by removing rows with missing values
2. Create a correlation matrix for all numeric columns
3. Identify the top 3 features most correlated with the target column
4. Train a random forest model using those features
5. Show me the feature importance plot
```

## File Structure

```
.
├── app.py                  # Main Flask application
├── agent_setup.py          # AI agent initialization
├── agent_tools.py          # Tool definitions for the agent
├── data_utils.py           # Data loading and utility functions
├── llm_utils.py            # LLM configuration and integration
├── state.py                # Application state management
├── cleaning.py             # Data cleaning implementations
├── visualization.py        # Data visualization functions
├── modeling.py             # Model training implementations
├── clustering.py           # Clustering algorithms implementation
├── code_executor.py        # Python code execution environment
├── requirements.txt        # Project dependencies
├── static/                 # Static assets
│   ├── style.css           # Application styling
│   └── *.html              # Generated plots
└── templates/              # Flask HTML templates
    ├── base.html           # Base template with common structure
    ├── overview.html       # Overview/upload page
    ├── cleaning.html       # Data cleaning interface
    ├── modeling.html       # Model training interface
    ├── charts.html         # Data visualization interface
    └── chat.html           # Agent chat interface
```

## Troubleshooting

### Common Issues

1. **API Key Configuration**
   - Check that your Gemini API key is valid and has sufficient quota
   - Ensure proper formatting in .env file (no quotes around the key)

2. **File Upload Problems**
   - Check that file format is supported (CSV, Excel)
   - Verify file size is under the maximum limit
   - Check for encoding issues in the data file

3. **Column Loading Issues**
   - Ensure a dataset has been uploaded before accessing other pages
   - Check browser console for specific error messages
   - Verify the state functions are properly implemented

4. **Model Training Errors**
   - Ensure target variable is appropriate for selected model type
   - Verify selected features are compatible with the model
   - Check for missing values or invalid data types

### Getting Support

For issues or feature requests, please:
- Check the browser console for detailed error messages
- Verify API key configuration and permissions
- Review setup instructions and requirements

## Dependencies

- **Flask**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn/Plotly**: Data visualization
- **LlamaIndex**: Agent framework
- **Google Generative AI**: LLM integration

## License

[Your License Information]

## Contributors

[Your Contributor Information]

---

**Note**: For production deployment, it is recommended to:
- Disable debug mode in app.py
- Use a production WSGI server like Gunicorn
- Implement proper security measures for API keys
- Consider containerization with Docker for consistent deployment