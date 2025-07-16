# Insurance Fraud Detection ML Pipeline

## Overview

The `InsuranceFraudDetector` is a comprehensive machine learning pipeline designed to detect fraudulent insurance claims. It supports multiple data sources, implements advanced preprocessing techniques, trains multiple ML models, and provides detailed performance analysis with visualizations.

## Features

- **Multi-source data ingestion**: BigQuery, local files (CSV, Excel, Parquet), or pandas DataFrames
- **Automated preprocessing**: Missing value imputation, categorical encoding, feature engineering
- **Multiple ML algorithms**: Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Class imbalance handling**: SMOTE oversampling for imbalanced datasets
- **Comprehensive evaluation**: Multiple metrics, ROC curves, confusion matrices
- **Feature importance analysis**: For tree-based models
- **Automated model selection**: Based on AUC score

## Installation

```python
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn google-cloud-bigquery
```

## Quick Start

```
from insurance_fraud_detector import InsuranceFraudDetector

# Initialize
## with local CSV file or Google Cloud Storage File
detector = InsuranceFraudDetector(
    data_filepath='fraud_data.csv' #or 'gs://gc_bucket_id/data/fraud_data.csv',
    target_column='is_fraud'
)

## Or with a SQL query to collect data from Google BigQuery
insurance_claims_df = pd.read_parquet('fraud_data.parquet')
detector = InsuranceFraudDetector(
            service_account_filepath = 'windy-smoke-420803-05e83bb28a9b.json' #if required
            , project_id = 'bigquery-public-data', dataset_name = 'fhir_synthea'
            , data_query = '''SELECT * FROM claims'''
            , target_column='is_fraud')
)

## Or with a dataframe
insurance_claims_df = pd.read_parquet('fraud_data.parquet')
detector = InsuranceFraudDetector(
    data_frame=insurance_claims_df,
    target_column='is_fraud'
)
```

```
# Run complete pipeline
results, best_model, report, plots = detector.run_complete_pipeline()
```

## Class Initialization

### Constructor Parameters

```python
InsuranceFraudDetector(
    target_column: str = None,
    service_account_filepath: str = None,
    project_id: str = None,
    dataset_name: str = None,
    data_query: str = None,
    data_filepath: str = None,
    data_frame: pd.DataFrame = pd.DataFrame()
)
```

### Parameters

- **target_column** (str, optional): Name of the target column. If None, automatically searches for columns containing 'fraud' or 'target'
- **service_account_filepath** (str, optional): Path to Google Cloud service account JSON file
- **project_id** (str, optional): Google Cloud project ID
- **dataset_name** (str, optional): BigQuery dataset name
- **data_query** (str, optional): SQL query for BigQuery data extraction
- **data_filepath** (str, optional): Path to local data file (CSV, Excel, or Parquet)
- **data_frame** (pd.DataFrame, optional): Pre-loaded pandas DataFrame

## Data Source Options

### Option 1: BigQuery
```python
detector = InsuranceFraudDetector(
    service_account_filepath='path/to/service-account.json',
    project_id='your-project-id',
    dataset_name='your-dataset',
    data_query='SELECT * FROM insurance_claims WHERE date > "2023-01-01"',
    target_column='fraud_flag'
)
```

### Option 2: Local File
```python
detector = InsuranceFraudDetector(
    data_filepath='data/insurance_claims.csv',
    target_column='is_fraudulent'
)
```

### Option 3: DataFrame
```python
detector = InsuranceFraudDetector(
    data_frame=your_dataframe,
    target_column='fraud_indicator'
)
```

## Core Methods

### 1. `load_and_explore_data()`

Loads data from the specified source and performs initial exploration.

**Returns**: pandas DataFrame

**Features**:
- Supports multiple file formats (CSV, Excel, Parquet)
- Displays dataset shape, column names, first few rows
- Shows data types and missing value summary
- Handles BigQuery authentication and querying

### 2. `preprocess_data(df, target_column_keywords=['fraud', 'target'])`

Comprehensive data preprocessing pipeline.

**Parameters**:
- **df**: Input DataFrame
- **target_column_keywords**: Keywords to search for target column

**Returns**: Tuple of (X, y) where X is features and y is target

**Processing Steps**:
1. Removes columns with all NULL values
2. Identifies target column automatically if not specified
3. Imputes missing values (median for numeric, mode for categorical)
4. Encodes categorical variables using LabelEncoder
5. Applies feature engineering
6. Stores preprocessing artifacts for future use

### 3. `engineer_features(X, amount_keywords=['amt', 'amount', 'cost'])`

Creates additional features for improved fraud detection.

**Parameters**:
- **X**: Feature DataFrame
- **amount_keywords**: Keywords to identify amount columns

**Feature Engineering**:
- **Amount features**: Percentile ranking, log transformation, high amount flags
- **Age features**: Age group binning (0-25, 25-45, 45-65, 65+)
- **Frequency features**: High frequency flags for count columns
- **Risk features**: High risk flags for risk/score columns

### 4. `train_models(X, y, test_size=0.2, random_state=42)`

Trains multiple ML models and evaluates their performance.

**Parameters**:
- **X**: Feature matrix
- **y**: Target vector
- **test_size**: Proportion of data for testing (default: 0.2)
- **random_state**: Random seed for reproducibility

**Models Trained**:
- **Random Forest**: 100 estimators, handles imbalanced data well
- **Gradient Boosting**: 100 estimators, sequential learning
- **Logistic Regression**: Linear model with L2 regularization
- **SVM**: Support Vector Machine with RBF kernel

**Returns**: Tuple of (results_dict, best_model_name)

**Features**:
- Automatic train/test split with stratification
- Feature scaling for linear models
- SMOTE oversampling for imbalanced datasets
- Cross-validation scoring
- Comprehensive metrics calculation

### 5. `plot_results()`

Creates comprehensive visualizations of model performance.

**Returns**: Tuple of (classification_report, matplotlib_plots)

**Visualizations**:
1. **Model Performance Comparison**: Bar chart of all metrics
2. **ROC Curves**: Receiver Operating Characteristic curves for all models
3. **Confusion Matrix**: Detailed confusion matrix for best model
4. **Feature Importance**: Top 10 most important features (tree-based models)

### 6. `predict_fraud_risk(new_data)`

Predicts fraud risk for new insurance claims.

**Parameters**:
- **new_data**: DataFrame with same structure as training data

**Returns**: Tuple of (predictions, probabilities)

**Features**:
- Applies same preprocessing as training data
- Uses best performing model
- Returns both binary predictions and probability scores

### 7. `run_complete_pipeline()`

Executes the entire fraud detection pipeline.

**Returns**: Tuple of (results, best_model_name, classification_report, plots)

**Pipeline Steps**:
1. Data loading and exploration
2. Data preprocessing and feature engineering
3. Model training and evaluation
4. Results visualization
5. Performance summary

## Evaluation Metrics

The pipeline calculates comprehensive metrics for model evaluation:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of predicted fraud cases that are actually fraud
- **Recall**: Proportion of actual fraud cases correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **AUC Score**: Area Under the ROC Curve

## Model Selection

The pipeline automatically selects the best model based on AUC score, which is particularly important for fraud detection as it:
- Handles class imbalance well
- Focuses on ranking quality
- Provides threshold-independent evaluation

## Best Practices

### Data Preparation
- Ensure consistent column naming
- Handle missing values appropriately
- Verify target column format (binary: 0/1 or boolean)

### Feature Engineering
- Include domain-specific features (transaction amounts, frequencies)
- Consider temporal features if timestamps available
- Normalize or scale features for linear models

### Model Tuning
- The pipeline uses default hyperparameters for quick prototyping
- Consider implementing GridSearchCV for production use
- Monitor for overfitting with cross-validation

### Deployment Considerations
- Save trained models using joblib or pickle
- Implement data drift monitoring
- Set up regular model retraining schedules

## Example Usage

```python
# Complete example with error handling
try:
    # Initialize detector
    detector = InsuranceFraudDetector(
        data_filepath='insurance_claims.csv',
        target_column='fraud_flag'
    )
    
    # Run pipeline
    results, best_model, report, plots = detector.run_complete_pipeline()
    
    # Make predictions on new data
    new_claims = pd.read_csv('new_claims.csv')
    predictions, probabilities = detector.predict_fraud_risk(new_claims)
    
    # Flag high-risk claims
    high_risk_claims = new_claims[probabilities > 0.7]
    
except Exception as e:
    print(f"Pipeline error: {e}")
```

## Performance Optimization

### Memory Management
- Use chunked processing for large datasets
- Consider feature selection for high-dimensional data
- Implement lazy loading for BigQuery

### Computational Efficiency
- Leverage parallel processing (n_jobs=-1 for Random Forest)
- Use appropriate data types (category for categorical variables)
- Consider dimensionality reduction for large feature sets

## Troubleshooting

### Common Issues

1. **Target column not found**: Verify column name or let auto-detection work
2. **Memory errors**: Reduce dataset size or use chunked processing
3. **Poor model performance**: Check class balance and feature quality
4. **BigQuery authentication errors**: Verify service account permissions

### Performance Issues

- **Low precision**: Consider adjusting classification threshold
- **Low recall**: Review feature engineering and class balancing
- **Overfitting**: Implement regularization or reduce model complexity

## Future Enhancements

- **Deep learning models**: Neural networks for complex patterns
- **Ensemble methods**: Combine multiple models for better performance
- **Online learning**: Real-time model updates
- **Explainable AI**: SHAP values for prediction explanations
- **Automated hyperparameter tuning**: Grid search and Bayesian optimization

## License

This pipeline is designed for educational and research purposes. Ensure compliance with data privacy regulations when working with sensitive insurance data.
