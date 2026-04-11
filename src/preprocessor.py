import numpy as np
import pandas as pd

class Preprocessor:
    def __init__(self, config: dict):
        # Initialize the preprocessor with the given configuration
        self.config = config
        self._median_values = {}  # To store median values for imputation
        self._modes = {}  # To store mode values for imputation
        self._most_frequent = {}  # To store most frequent values for imputation
        self.isFitted = False  # Flag to check if the preprocessor has been fitted

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        # Fit the preprocessor to the data (calculate median and mode for imputation)
        fill_strategy = self.config["preprocessing"]["fill_strategy"]

        for column, strategy in fill_strategy.items():
            if strategy == "median":
                self._median_values[column] = df[column].fillna(df[column].median())
                self._median_values[column] = self._median_values[column].astype(np.int64)
            elif strategy == "mode":
                self._modes[column] = df[column].dropna().mode()[0]
                self._modes[column] = self._modes[column].astype(np.int64)
            elif strategy == "most_frequent":
                self._most_frequent[column] = df[column].dropna()
                
        
                

        self.isFitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Transform the data (impute missing values and perform any other transformations)
        if not self.isFitted:
            raise ValueError("Preprocessor must be fitted before calling transform.")
        
        df = df.copy()  # Avoid modifying the original DataFrame
        

        for column, median_val in self._median_values.items():
            df[column] = df[column].fillna(median_val)
        for column, mode_val in self._modes.items():
            df[column] = df[column].fillna(mode_val)
        
        
        for column, median_val in self._median_values.items():
            df[column] = df[column].fillna(median_val)
        for column, mode_val in self._modes.items():
            df[column] = df[column].fillna(mode_val)    
        encode_config = self.config["preprocessing"]["encode"]
        if "sex" in encode_config:
            df["sex"] = df["sex"].map(encode_config["sex"]).astype(int)
             
        if encode_config.get("embarked") == "onehot":
            df = pd.get_dummies(df, columns=["embarked"], drop_first=True)
        return df
                   