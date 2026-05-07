import os
import sys
import numpy as np
import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
os.environ["MLFLOW_PYTHON_BIN"] = sys.executable
import argparse

def main():
    
    # Przykładowa funkcjonalność
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    X, y, X_train, X_test, y_train, y_test = load_process()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": 42,
        "imputation": args.imputation,
    }

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if "MLFLOW_RUN_ID" not in os.environ:
       mlflow.set_experiment("penguins-classification")

    #run_id="52fc31821d2f4918a0c563684c53c9cf"
    #run = mlflow.get_run()
    #model_uri =  f"runs:/{run_id}/{run.data.tags.get('mlflow.model.log-model.history')}"
    #mlflow.sklearn.load_model(model_uri)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature
        )
        
        print(f"Parametry: n_estimators={args.n_estimators}, max_depth={args.max_depth}, imputation={args.imputation}")
        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
    
    

def load_process():
    # load model
    df = pd.read_csv("data/pemguins.csv")
    X = df.drop(columns=["species"])
    y = df["species"]
    
    
    # load model train
    train_df = pd.read_csv("data/train.csv")
    X_train = train_df.drop(columns=["species"])
    y_train = train_df["species"]
    
    # load model test
    test_df = pd.read_csv("data/test.csv")
    X_test = test_df.drop(columns=["species"])
    y_test = test_df["species"]

    return X, y, X_train, X_test, y_test, y_train


if __name__ == "__main__":
    main()
    
    
    