
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

    def train_and_evaluate(self):
        models = {
            "Logistic Regression": LogisticRegression(solver='liblinear', random_state=self.random_state),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
            "SVC": SVC(random_state=self.random_state, probability=True)
        }

        param_grids = {
            "Logistic Regression": {
                "classifier__C": [0.1, 1.0, 10.0]
            },
            "Random Forest": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [None, 10, 20]
            },
            "Gradient Boosting": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__learning_rate": [0.01, 0.1, 0.2]
            },
            "SVC": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["linear", "rbf"]
            }
        }

        best_score = 0
        for name, model in models.items():
            logging.info(f"Training and evaluating {name}...")
            pipeline = Pipeline(steps=[("preprocessor", self.preprocessor), ("classifier", model)])
            grid_search = GridSearchCV(
                pipeline, param_grids[name], cv=KFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring="accuracy", n_jobs=-1, verbose=0
            )
            grid_search.fit(self.X_train, self.y_train)

            predictions = grid_search.best_estimator_.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, predictions, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, predictions, average='weighted', zero_division=0)

            logging.info(f"{name} - Best Params: {grid_search.best_params_}")
            logging.info(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            if accuracy > best_score:
                best_score = accuracy
                self.model = grid_search.best_estimator_
                self.best_model_name = name
                self.metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                }
        logging.info(f"Best model found: {self.best_model_name} with Accuracy: {self.metrics["accuracy"]:.4f}")

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.train_and_evaluate()
        return self.model, self.metrics

if __name__ == "__main__":
    # Example usage (requires a dummy dataset.csv in the same directory)
    # Create a dummy CSV for demonstration if it doesn't exist
    if not os.path.exists("dataset.csv"):
        dummy_data = {
            "feature1": np.random.rand(100),
            "feature2": np.random.randint(0, 100, 100),
            "feature3": [random.choice(['A', 'B', 'C']) for _ in range(100)],
            "target": [random.choice([0, 1]) for _ in range(100)]
        }
        pd.DataFrame(dummy_data).to_csv("dataset.csv", index=False)
        logging.info("Created dummy dataset.csv for example.")

    try:
        pipeline = AutoMLPipeline(data_path="dataset.csv", target_column="target")
        best_model, metrics = pipeline.run()
        logging.info("AutoML pipeline execution complete.")
        logging.info(f"Final Best Model: {pipeline.best_model_name}")
        logging.info(f"Final Metrics: {metrics}")
    except Exception as e:
        logging.error(f"AutoML pipeline failed: {e}")
