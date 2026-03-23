
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AutoMLPipeline:
    def __init__(self, data_path, target_column, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.best_model_name = None
        self.metrics = {}
        logging.info(f"AutoMLPipeline initialized for target: {target_column}")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            logging.info(f"Data loaded successfully from {self.data_path}. Shape: {self.df.shape}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        categorical_features = X.select_dtypes(include=["object", "category"]).columns
        numerical_features = X.select_dtypes(include=np.number).columns

        numerical_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features)
            ])
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        logging.info("Data preprocessed and split into training and testing sets.")

        pass

if __name__ == "__main__":
    print("AutoML pipeline with data loading and preprocessing.")
