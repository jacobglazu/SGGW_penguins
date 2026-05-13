import os
import sys
import yaml
import json
from joblib import dump
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
from sklearn.preprocessing import OneHotEncoder
os.environ["MLFLOW_PYTHON_BIN"] = sys.executable
import argparse



def main():
    
    # Przykładowa funkcjonalność
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--imputation", type=str, default="median")
    args = parser.parse_args()


    X_final, y, X_train, X_test, y_train, y_test = load_process()
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )
    
    """params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": 42,
        "imputation": args.imputation,
    }"""

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    metric = {"accuracy" : accuracy, "f1" : f1}
    with open("metric.json", "w") as f:
        json.dump(metric, f, indent=2)


    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("penguins-classification")

    #run_id="52fc31821d2f4918a0c563684c53c9cf"
    #run = mlflow.get_run()
    #model_uri =  f"runs:/{run_id}/{run.data.tags.get('mlflow.model.log-model.history')}"
    #mlflow.sklearn.load_model(model_uri)
    #model_path = r"d:\\Kuba\\SGGW_penguins\\models\\model.pkl"
    #with open(model_path, "rb") as file:
     #   model_2 = pickle.load(file)

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
        #dump(model, "model.pkl")
        #mlflow.log_artifact("model.pkl")
        #mlflow.log_artifact("d:\\Kuba\\SGGW_penguins\\models\\model.pkl")
        #mlflow.log_artifacts("metrics/")
        
        print(f"Parametry: n_estimators={args.n_estimators}, max_depth={args.max_depth}, imputation={args.imputation}")
        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
    
    

def load_process():
    # load model
    df = pd.read_csv("./data/penguins.csv")
    df_2 = df.copy()
    #df_2["sex"] = df_2["sex"].map({"male": 0, "female": 1}).astype(int)

    X = df_2.drop(columns=["species"])
    y = df_2["species"]

    cat_features = ["island", "sex"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_encoded = encoder.fit_transform(X[cat_features])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(cat_features), index=X.index)
            
    #training_config = self.config["training"]
    X_final = pd.concat([X.drop(columns = cat_features), X_encoded_df], axis=1)
    
    
    
    # load model train
    train_df = pd.read_csv("./data/train.csv")
    X_train = train_df.drop(columns=["species"])
    y_train = train_df["species"]
    
    # load model test
    test_df = pd.read_csv("./data/test.csv")
    X_test = test_df.drop(columns=["species"])
    y_test = test_df["species"]

    return X_final, y, X_train, X_test, y_test, y_train


if __name__ == "__main__":
    main()
    
    
    