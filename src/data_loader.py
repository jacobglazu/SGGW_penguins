import os
import yaml
import pandas as pd
from sklearn.datasets import fetch_openml

class DataLoader:
    def __init__(self, config: dict):
        # Initialize the data loader with the given configuration
        self.config = config
       
        
    
    def load_data(self) -> pd.DataFrame:
        
        # Load the dataset based on the configuration
        #data_config = self.config["data"]
        #dataset_id = data_config["dataset_id"]
        

        # Fetch the dataset from OpenML
        #dataset = fetch_openml(data_id=dataset_id, as_frame=True)

        with open("params.yaml") as f:
            params = yaml.safe_load(f)
    
        dataset_id = params["data"]["dataset_id"]
        print(f"Pobieranie danych Palmer Penguins (id={dataset_id})...")

        dataset = fetch_openml(data_id=dataset_id, as_frame=True, parser="pandas")

        df = dataset.frame
        df = df.dropna()
        output_data = "data"
        output_file = os.path.join(output_data, "penguins.csv")

        if not os.path.exists(output_data):
            os.makedirs(output_data)

        df.to_csv(output_file, index=False)
        print(f"Dataset saved to {output_file}")

       


        # Perform any necessary preprocessing steps
        
        drop_columns = dataset.get("drop_columns", [])
        if drop_columns:
            df = df.drop(columns=drop_columns)  
        return df