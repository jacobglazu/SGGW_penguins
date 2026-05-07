import os
import argparse
import mlflow
from sklearn.ensemble import RandomForestClassifier

def main(n_estimators, max_depth):
    # Przykładowa funkcjonalność
    mlflow.start_run()

    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("penguins-classification")
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    # Tutaj dodaj kod do trenowania modelu i rejestracji w MLflow
    print(f"Model trained with n_estimators={n_estimators} and max_depth={max_depth}")
    
    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, required=True)
    parser.add_argument("--max_depth", type=int, required=True)
    args = parser.parse_args()

    """if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("titanic-mlflow-project")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
        )"""
    
    main(args.n_estimators, args.max_depth)
