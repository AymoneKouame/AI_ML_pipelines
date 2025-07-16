import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from google.oauth2.service_account import Credentials
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

class InsuranceFraudDetector:
    """
    A comprehensive insurance fraud detection system using multiple ML algorithms
    """
    
    def __init__(self, target_column:str = None
                 ## when input data needs to be queried from GBQ
                 , service_account_filepath:str = None, project_id:str = None, dataset_name:str = None, data_query:str = None
                 ## when input data is a local or GC Storage file
                 , data_filepath:str = None
                 ## when input data is a dataframe
                 , data_frame = pd.DataFrame()):

        # Set up variables in 'self'
        self.service_account_filepath = service_account_filepath
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.target_column = target_column
        self.data_filepath = data_filepath
        self.data_query = data_query
        self.data_frame = data_frame
        
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.best_model = None
        self.best_score = 0


    def query_gbq(self):
    
        dataset = f'{self.project_id}.{self.dataset_name}'
        credentials = Credentials.from_service_account_file(self.service_account_filepath)
        
        config = bigquery.QueryJobConfig(default_dataset = dataset) 
        client = bigquery.Client(default_query_job_config= config, credentials = credentials)
        
        query_job = client.query(self.data_query)  # API request
        df = query_job.result().to_dataframe()
        
        return df            

        
    def load_and_explore_data(self):
        """
        Load and perform initial exploration of the dataset
        """        
        try:
            # Try different file formats
            if self.data_filepath != None:
                print(f'Reading {self.data_filepath} ...')
                data_inputs_dd = {'.csv': pd.read_csv, '.xlsx': pd.read_excel, '.parquet': pd.read_parquet}
                file_ext = '.'+self.data_filepath.split('.')[1].lower()
                df = data_inputs_dd[file_ext](self.data_filepath)
                
            elif self.data_query !=None:
                print('Querying GBQ ...')
                df = self.query_gbq()

            elif self.data_frame.empty == False:
                print('Loading dataframe ...')
                df = self.data_frame.copy()
                
            print(f"Dataset shape: {df.shape}")
            print("\n~~~~~~~~~~~~~~~~ Column names ~~~~~~~~~~~~~~~~")
            print(df.columns.tolist())
            print("\n~~~~~~~~~~~~~~~~First few rows ~~~~~~~~~~~~~~~~")
            display(df.head())
            print("\n~~~~~~~~~~~~~~~~Data info~~~~~~~~~~~~~~~~")
            display(df.info())
            print("\n~~~~~~~~~~~~~~~~Missing values~~~~~~~~~~~~~~~~")
            display(df.isnull().sum())
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")

        
    def preprocess_data(self, df, target_column_keywords = ['fraud', 'target']):
        """
        Comprehensive data preprocessing pipeline
        """
        print("Starting data preprocessing...")
        
        ## Drop columns with ALL NULL values, if any
        print('  Drop column with all NULLs...')        
        all_null_cols = []
        for c in df.columns:   
            if (df[c].isnull().unique()).all():
                all_null_cols = all_null_cols+[c]
        df = df.drop(all_null_cols, axis =1)
        
        
        # Identify target column
        print('  Define or identify target variable...')
        if (self.target_column not in df.columns) or (self.target_column == None):
            # Try to find fraud indicator column
            fraud_cols = []
            for fraud_kw in target_column_keywords:
                fraud_cols = fraud_cols+[col for col in df.columns if fraud_kw in col.lower()]
                if fraud_cols:
                    target_column = fraud_cols[0]
                else:
                    raise ValueError("No fraud/target column found. Please specify target_column parameter.")
       
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        print(f"Target variable: {target_column}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Handle missing values
        print('''  Impute missing values. 
    Inpute numeric colums wuth median and categorical values with the most frequent category...''')
        # Impute missing values for the remaining columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        categorical_columns = X.select_dtypes(exclude=[np.number]).columns        
        
        if len(numeric_columns) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])
        
        if len(categorical_columns) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])
        
        # Encode categorical variables
        print('''  Encode Categorical variables...''')
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Feature engineering
        X = self.engineer_features(X)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Preprocessed data shape: {X.shape}")
        print("Preprocessing completed!")
        
        return X, y
    
    def engineer_features(self, X, amount_keywords = ['amt', 'amount', 'cost']):
        """
        Create additional features that might be relevant for fraud detection
        """
        print("Engineering features...")
        print(f" Features before engineering: {X.shape[1]}")        
        # Create ratio features if amount columns exist
        amount_cols = []
        for amount_keyword in amount_keywords:
            amount_cols = amount_cols+[col for col in X.columns if amount_keyword in col.lower()]
        print(f' Create percentile, high amount flag, and perform log transformatopm for cost/amount columns...')
        if len(amount_cols) > 0:
            for amount_col in amount_cols:
            
                # Amount percentile ranking
                X[f'{amount_col}_percentile'] = X[amount_col].rank(pct=True)
                
                # Log transformation for amount (handle zeros)
                X[f'{amount_col}_log'] = np.log1p(X[amount_col])
                
                # High amount flag
                X[f'{amount_col}_high'] = (X[amount_col] > X[amount_col].quantile(0.9)).astype(int)
        
        # Age-related features if age column exists
        print(f' Create groups age columns, if any...')       
        age_cols = [col for col in X.columns if 'age' in col.lower()]
        if len(age_cols) > 0:
            for age_col in age_cols:
                X[f'{age_col}_group'] = pd.cut(X[age_col], bins=[0, 25, 45, 65, 100], labels=[0, 1, 2, 3])
                X[f'{age_col}_group'] = X[f'{age_col}_group'].astype(int)
        
        # Frequency-based features
        print(f' Create frequency groups for counts/frequency columns, if any...')
        freq_cols = [col for col in X.columns if 'frequency' in col.lower() or ('count' in col.lower() and col.lower() != 'County')]
        if len(freq_cols) > 0:
            for freq_col in freq_cols:
                X[f'{freq_col}_high'] = (X[freq_col] > X[freq_col].quantile(0.8)).astype(int)
        
        # Risk score features
        print(f' Create risk groups for risk columns, if any...')        
        risk_cols = [col for col in X.columns if 'risk' in col.lower() or 'score' in col.lower()]
        if len(risk_cols) > 0:
            for risk_col in risk_cols:
                X[f'{risk_col}_high'] = (X[risk_col] > X[risk_col].quantile(0.8)).astype(int)
        
        print(f" Features after engineering: {X.shape[1]}")
        
        return X
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """
        Train multiple ML models and compare their performance
        """
        print("Splitting data and training models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        models_config = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000),
            'SVM': SVC(random_state=random_state, probability=True)
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Handle class imbalance with SMOTE for tree-based models
            if name in ['RandomForest', 'GradientBoosting']:
                # Use original features for tree-based models
                X_train_model = X_train
                X_test_model = X_test
            else:
                # Use scaled features for linear models
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            
            # Apply SMOTE for imbalanced datasets
            if y_train.value_counts().min() / y_train.value_counts().max() < 0.1:
                smote = SMOTE(random_state=random_state)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_model, y_train)
            else:
                X_train_balanced, y_train_balanced = X_train_model, y_train
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions
            y_pred = model.predict(X_test_model)
            y_pred_proba = model.predict_proba(X_test_model)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc,
                'predictions': y_pred,
                'predictions_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC: {auc:.4f}")
            
            # Track best model
            if auc > self.best_score:
                self.best_score = auc
                self.best_model = model
        
        # Store results
        self.models = results
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"\nBest model: {self.get_best_model_name()} with AUC: {self.best_score:.4f}")
        
        return results, self.get_best_model_name()
    
    def get_best_model_name(self):
        """Get the name of the best performing model"""
        for name, result in self.models.items():
            if result['model'] == self.best_model:
                return name
        return "Unknown"
    
    def plot_results(self):
        """
        Visualize model performance and create comprehensive plots
        """
        print("Creating visualization plots...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Performance Comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        model_names = list(self.models.keys())
        
        metric_data = {}
        for metric in metrics:
            metric_data[metric] = [self.models[name][metric] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            axes[0, 0].bar(x + i * width, metric_data[metric], width, label=metric.replace('_', ' ').title())
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x + width * 2)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC Curves
        for name, result in self.models.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['predictions_proba'])
            axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC = {result['auc_score']:.3f})")
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confusion Matrix for Best Model
        best_model_name = self.get_best_model_name()
        cm = confusion_matrix(self.y_test, self.models[best_model_name]['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 4. Feature Importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            axes[1, 1].barh(range(len(feature_importance)), feature_importance['importance'])
            axes[1, 1].set_yticks(range(len(feature_importance)))
            axes[1, 1].set_yticklabels(feature_importance['feature'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title(f'Top 10 Feature Importances - {best_model_name}')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed classification report for best model
        print(f"\nDetailed Classification Report for {best_model_name}:")
        classification_report_df = classification_report(self.y_test, self.models[best_model_name]['predictions'])
        print(classification_report_df)

        return classification_report_df, plt
    
    def predict_fraud_risk(self, new_data):
        """
        Predict fraud risk for new insurance claims
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Please train models first.")
        
        # Preprocess new data (you may need to adapt this based on your specific preprocessing)
        # For now, assuming new_data is already in the correct format
        
        # Scale data if needed
        if self.get_best_model_name() not in ['RandomForest', 'GradientBoosting']:
            new_data_scaled = self.scaler.transform(new_data)
            predictions = self.best_model.predict(new_data_scaled)
            probabilities = self.best_model.predict_proba(new_data_scaled)[:, 1]
        else:
            predictions = self.best_model.predict(new_data)
            probabilities = self.best_model.predict_proba(new_data)[:, 1]
        
        return predictions, probabilities
    
    def run_complete_pipeline(self):
        """
        Run the complete fraud detection pipeline
        """
        print("=== Insurance Fraud Detection Pipeline ===\n")
        
        # Load data
        df = self.load_and_explore_data()
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Train models
        results_dd, best_model_name = self.train_models(X, y)
        
        # Plot results
        classification_report_tb, plots = self.plot_results()
        
        # Summary
        print("\n=== Pipeline Summary ===")
        print(f"Dataset size: {len(df)} samples")
        print(f"Number of features: {X.shape[1]}")
        print(f"Fraud rate: {y.mean():.2%}")
        print(f"Best model: {self.get_best_model_name()}")
        print(f"Best AUC score: {self.best_score:.4f}")

        print('Returning results, best_model , classification_report_df, and plots.')
        return results_dd, best_model_name, classification_report_tb, plots
