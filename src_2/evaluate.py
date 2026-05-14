import json
import pickle
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models import infer_signature

def main():
    test_df = pd.read_csv("data/test.csv")
    X_test = test_df.drop(columns=["species"])
    y_test = test_df["species"]

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
                        
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    metrics = {"accuracy": accuracy, "f1_score": f1}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    with mlflow.start_run(run_name="best-model"):
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)

    print(f"Metryki: accuracy={accuracy:.4f}, f1_score={f1:.4f}")
    print("metrics.json was saved.")
    

if __name__ == "__main__":
    main()