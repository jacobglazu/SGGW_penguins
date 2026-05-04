import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import mlflow
import mlflow.data
from mlflow.models import infer_signature


class ModelTrainer:
    y_test = None
    y_pred = None
    test_dataset = None
    train_dataset = None
    metrics = None

    def __init__(self, config: dict):
        # Initialize the model trainer with the given configuration
        model_config = config["model"]
        model_type = model_config["type"]
        model_params = model_config.get("params", {})
        self.predict = self.predict
        self.train = self.train
        self.predict = self.predict
        self.evaluate = self.evaluate
        global y_test 
        global y_pred 
        cm = None
        signature = None
        global test_dataset
        global train_dataset
        global metrics 

        if model_type in {"RandomForest", "RandomForestClassifier"}:
            self.model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
    def predict(self, X_test: np.ndarray, y_test: np.ndarray):
        test_df = pd.read_csv("data/test.csv")
        X_test = test_df.drop(columns=["species"])
        y_test = test_df["species"]

        #return X_test, #y_test
        test_data = pd.concat([X_test, y_test], axis=1)
        test_dataset = mlflow.data.from_pandas(test_data, name="penguins_test")
        return test_dataset

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # Train the model using the training data and return the trained model
        #self.model.fit(X_train, y_train)
        train_df = pd.read_csv("data/train.csv")
        X_train = train_df.drop(columns=["species"])
        y_train = train_df["species"]

        #return X_train, y_train
        model_train = self.model.fit(X_train,y_train)

        #with open("models/model.pkl", "wb") as f:
         #   pickle.dump(model_train, f)
        train_data = pd.concat([X_train, y_train], axis=1)
        train_dataset = mlflow.data.from_pandas(train_data, name="penguins_train")

        return train_dataset

    def evaluate(self, y_pred: np.ndarray, y_test: np.ndarray) -> dict:
        # Evaluate the model using the test data and return the accuracy
        test_df = pd.read_csv("data/test.csv")
        X_test = test_df.drop(columns=["species"])
        y_test = test_df["species"]
        
        
        y_pred = self.model.predict(X_test)
        print(y_pred)
        cm = confusion_matrix(y_test, y_pred)
        signature = infer_signature(X_test, y_pred)
        
        
        
        
        
        #y_pred = self.model.predict(X_test)
        print("y_test:", y_test.head())
        #print(y_test.dtype())
        print("y_pred:", y_pred)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        metrics = {
            "accuracy" : round(accuracy, 4),
            "f1" : round(f1, 4),
            "precision" : round(precision, 4),
            "recall" : round(recall, 4)
        }
        print(metrics)
        # Przekierownie zmiennych
        __all__ = ['y_test', 'y_pred', 'test_dataset', 'train_dataset', 'cm', 'signature', 'evaluate()', 'metrics']

        return metrics, cm , signature

        """return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            #"cm": cm,
            #"siganture": signature,
            #"y_pred": y_pred,
            #"y_test": y_test
        }"""
    