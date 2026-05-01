#from bentoml import metrics
import os
import numpy as np
import tempfile
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
import pickle
from .data_loader import DataLoader
from .preprocessor import Preprocessor
from .trainer import ModelTrainer
#from ..data import model
import pandas as pd
import mlflow
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
from .trainer import *

class PenguinPipeline:
    def __init__(self, config: dict, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.config = config
        self.data_loader = DataLoader(config)
        self.preprocessor = Preprocessor(config)
        self.trainer = ModelTrainer(config)
        self.model = config["model"]
        #self.trainer.train()
        self.y_test = self.trainer.y_test
        self.y_pred = self.trainer.y_test      
        

    def run(self):
        # Load the data
        df = self.data_loader.load_data()
        df = df.copy()
        #self.preprocessor.fit(X_train)
        # Preprocess the data"
        X = df.drop(columns=["species"])
        y = df["species"]
        
        


        cat_features = ["island", "sex"]
        #encode_config = self.config["preprocessing"]["encode"]
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_encoded = encoder.fit_transform(X[cat_features])
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(cat_features), index=X.index)
            
        training_config = self.config["training"]
        X_final = pd.concat([X.drop(columns = cat_features), X_encoded_df], axis=1)
        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(X_final, y, 
            test_size=training_config["test_size"], 
            random_state=training_config["random_state"])

        #self.preprocessor.fit(X_train)
        
        

        #X_train = self.preprocessor.transform(X_train)
        #X_test = self.preprocessor.transform(X_test)

        #model = self.trainer.train(X_train, y_train)

        os.makedirs("models", exist_ok=True)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
    
        train_df.to_csv("data/train.csv", index=False)
        test_df.to_csv("data/test.csv", index=False)
        with open("models/encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
        print("Preprocessing done. Encoder saved.")
        

        trainer = ModelTrainer(self.config)
        model = trainer.train(X_train, y_train)
        y_pred= trainer.predict(X_test, y_test)
        # Evaluate the model
        metrics = trainer.evaluate(y_pred, y_test)
        cm =  trainer.evaluate(y_pred, y_test)
        signature = trainer.evaluate(y_pred, y_test)
        """for metric_name, metric_value in metrics.items():
            print(f"{metric_name.capitalize()}: {metric_value:.4f}")"""
        
        #X_train = self.preprocessor.transform(X_train_trans)
        #X_test = self.preprocessor.transform(X_test)
        #with open ("models/model.pkl" , "rb") as f:
            #model = pickle.load(f)

        
        model = self.config["model"]
        model_type = model["type"]
        model_par = model["parameters"]
        print(f"Model", self.model)
        model_path = "d:\\Kuba\\SGGW_penguins\\data\\models\\model.pkl"
        #return metrics
        mlflow.set_experiment("penguins-classification")

        with mlflow.start_run(run_name="Model_One"):
           mlflow.log_input(trainer.train_dataset, context="train")
           mlflow.log_input(trainer.test_dataset, context= "test")

           params = {
            "C": 1.0,
            "penalty": "l2",
            }
           mlflow.log_params(params)
           model_ml = LogisticRegression(**params)
           # to do cos nie dzila metrics
           #eval_result = trainer.evaluate()
           #dict_metrics = dict(metrics)
           for metric_name, metric_value in enumerate(metrics):
            mlflow.log_metrics(metrics)
            print(metrics)

           tmpdir = tempfile.mkdtemp()
           try:
                #cm = trainer.cm
                disp = ConfusionMatrixDisplay(confusion_matrix= cm, )
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                disp.plot(ax=ax_cm, cmap="Blues")
                ax_cm.set_title("Macierz pomyłek — Random Forest")
                cm_path = os.path.join(tmpdir, "confusion_matrix.png")
                fig_cm.savefig(cm_path, bbox_inches="tight", dpi=150)
                plt.close(fig_cm)
                mlflow.log_artifact(cm_path, artifact_path="plots")
           finally:
                shutil.rmtree(tmpdir)

           mlflow.log_artifact(__file__, artifact_path="code")

           #signature = trainer.signature
           mlflow.sklearn.log_model(model_ml, "model", signature=signature)

           """ mlflow.log_params({"model_type" : model_type})
            mlflow.log_params({"model_par": model_par})
            mlflow.log_metrics(metrics)
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.load_model(model_path, "model")"""
            

