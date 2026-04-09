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

        # Preprocess the data
        X, y = self.preprocessor.preprocess(df)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = self.trainer.train(X_train, y_train)

        # Evaluate the model
        metrics = self.trainer.evaluate(model, X_test, y_test)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.capitalize()}: {metric_value:.4f}")
        

        return metrics