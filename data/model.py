import os
import tempfile
import shutil
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.data
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

def load_data():
    # Load the dataset
    data = fetch_openml(data_id=42585, as_frame=True)
    df = data.frame

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Separate features and target
    X = df.drop(columns=['species'])
    y = df['species']
    
   # One-hot encode categorical features
    X = pd.get_dummies(X, columns=['island', 'sex'], drop_first=True)
    
    print(X.head(120))
    print(y.head(120))
    return X, y

def main():
    X, y = load_data()
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    # Log the model and metrics to MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

if __name__ == "__main__":
    #main()
    load_data()
    #print(X.head())
    #print(y.head())