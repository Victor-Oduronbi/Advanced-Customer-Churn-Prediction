import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

class ChurnProductionPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.preprocessor = None
        self.model = None
        
    def load_and_clean(self):
        """Loads dataset, handles structural anomalies, and fixes data types."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        df.drop(columns=['customerID'], inplace=True, errors='ignore')
        
        # Resolve blank string spaces in TotalCharges
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        df['TotalCharges'] = df['TotalCharges'].astype(float).fillna(0.0)
        
        # Standardize target format safely
        df['Churn'] = df['Churn'].astype(str).str.lower().map({'yes': 1, 'true': 1, 'no': 0, 'false': 0})
        return df

    def build_and_train(self):
        """Preprocesses data, handles imbalance with SMOTE, and trains champion model."""
        df = self.load_and_clean()
        
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        
        # Establish processing sub-pipelines
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
            ]
        )
        
        # Transform features and handle structural class imbalance
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        # Train our champion high-recall business model
        print("Training production-grade Logistic Regression Champion...")
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train_resampled, y_train_resampled)
        
        # Validate baseline generalization
        X_test_scaled = self.preprocessor.transform(X_test)
        test_acc = self.model.score(X_test_scaled, y_test)
        print(f"Pipeline training complete. Test Accuracy Baseline: {test_acc:.4f}")

    def save_artifacts(self, model_pkl='champion_model.pkl', preprocessor_pkl='preprocessor.pkl'):
        """Serializes and saves training pipeline pieces to disk."""
        if self.model is None or self.preprocessor is None:
            raise ValueError("Pipeline must be trained before artifacts can be exported.")
        joblib.dump(self.model, model_pkl)
        joblib.dump(self.preprocessor, preprocessor_pkl)
        print(f"Artifacts successfully saved to disk: {model_pkl}, {preprocessor_pkl}")

if __name__ == "__main__":
    # Execute production compilation pipeline
    pipeline_runner = ChurnProductionPipeline(data_path='WA_Fn-UseC_-Telco-Customer-Churn.csv')
    pipeline_runner.build_and_train()
    pipeline_runner.save_artifacts()