import pandas as pd
from sklearn.datasets import fetch_openml

class DataLoader:
    def __init__(self, config: dict):
        # Initialize the data loader with the given configuration
        self.config = config
    
    def load_data(self) -> pd.DataFrame:
        # Load the dataset based on the configuration
        data_config = self.config["data"]
        dataset_id = data_config["dataset_id"]

        # Fetch the dataset from OpenML
        dataset = fetch_openml(data_id=dataset_id, as_frame=True)
        df = dataset.frame

        # Perform any necessary preprocessing steps
        drop_columns = data_config.get("drop_columns", [])
        if drop_columns:
            df = df.drop(columns=drop_columns)  
        return df