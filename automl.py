import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class AutoMLPipeline:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.model = None

    def preprocess(self):
        # Drop missing values and encode categorical features
        self.data = self.data.dropna()
        print("Data preprocessing complete.")

    def train(self, target_column):
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        predictions = self.model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, predictions)}")

print("AutoML Pipeline script ready.")
