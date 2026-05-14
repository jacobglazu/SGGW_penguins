import pickle
import yaml
import pandas as pd
import optuna
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from optuna.integration.mlflow import MLflowCallback

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    train_df = pd.read_csv("data/train.csv")
    X_train = train_df.drop(columns=["species"])
    y_train = train_df["species"]

    mlflow.set_experiment("penguins-optuna")
    experiment = mlflow.get_experiment_by_name("penguins-optuna")
    exp_id = experiment.experiment_id

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=params["model"]["random_state"]
        )
        return cross_val_score(model, X_train, y_train, n_jobs=-1, cv=5).mean()
    
    mlflow_cb = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), 
        metric_name="accuracy",
        mlflow_kwargs={"experiment_id": exp_id}
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, callbacks=[mlflow_cb])

    # Best model training
    best_model = RandomForestClassifier(**study.best_params, random_state=params["model"]["random_state"])
    best_model.fit(X_train, y_train)
    
    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_model, f)

if __name__ == "__main__":
    main()