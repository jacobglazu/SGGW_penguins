#from bentoml import metrics
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
from .data_loader import DataLoader
from .preprocessor import Preprocessor
from .trainer import ModelTrainer
import pandas as pd


class PenguinPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.data_loader = DataLoader(config)
        self.preprocessor = Preprocessor(config)
        self.trainer = ModelTrainer(config)
        

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
        
        model = self.trainer.train(X_train, y_train)
        y_pred= self.trainer.predict(X_test, y_test)
        # Evaluate the model
        metrics = self.trainer.evaluate(y_test,y_pred)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.capitalize()}: {metric_value:.4f}")
        
        #X_train = self.preprocessor.transform(X_train_trans)
        #X_test = self.preprocessor.transform(X_test)

        


        return metrics

