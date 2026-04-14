#from bentoml import metrics
from sklearn.model_selection import train_test_split

from .data_loader import DataLoader
from .preprocessor import Preprocessor
from .trainer import ModelTrainer

class PenguinPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.data_loader = DataLoader(config)
        self.preprocessor = Preprocessor(config)
        self.trainer = ModelTrainer(config)

    def run(self):
        # Load the data
        df = self.data_loader.load_data()

        #self.preprocessor.fit(X_train)
        # Preprocess the data"
        X = df.drop(columns=["species"])
        y = df["species"]
        
        training_config = self.config["training"]
        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
            test_size=training_config["test_size"], 
            random_state=training_config["random_state"])

        self.preprocessor.fit(X_train)
        
        X_train = self.preprocessor.transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        model = self.trainer.train(X_train, y_train)
        # Evaluate the model
        metrics = self.trainer.evaluate( X_test, y_test)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.capitalize()}: {metric_value:.4f}")
        

        return metrics