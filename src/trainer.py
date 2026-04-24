import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class ModelTrainer:
    def __init__(self, config: dict):
        # Initialize the model trainer with the given configuration
        model_config = config["model"]
        model_type = model_config["type"]
        model_params = model_config.get("params", {})

        if model_type in {"RandomForest", "RandomForestClassifier"}:
            self.model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
    def predict(self, X_test: np.ndarray):
        test_df = pd.read_csv("data/test.csv")
        X_test = test_df.drop(columns=["species"])
        #y_test = test_df["species"]

        return X_test, #y_test
        
        #return y_pred

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # Train the model using the training data and return the trained model
        #self.model.fit(X_train, y_train)
        train_df = pd.read_csv("data/train.csv")
        X_train = train_df.drop(columns=["species"])
        y_train = train_df["species"]

        #return X_train, y_train
        self.model.fit(X_train,y_train)
        

    def evaluate(self, y_pred: np.ndarray, y_test: np.ndarray) -> dict:
        # Evaluate the model using the test data and return the accuracy
        test_df = pd.read_csv("data/test.csv")
        X_test = test_df.drop(columns=["species"])
        y_test = test_df["species"]
        
        
        y_pred = self.model.predict(X_test)


        
        
        
        
        
        #y_pred = self.model.predict(X_test)
        print("y_test:", y_test.head())
        #print(y_test.dtype())
        print("y_pred:", y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    