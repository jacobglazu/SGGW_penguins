import pandas as pd
from statistics import median
from typing import Self

class Preprocessor:
    def __init__(self, config: dict):
        # Initialize the preprocessor with the given configuration
        self.config = config
        self._median_values = {}  # To store median values for imputation
        self._modes = {}  # To store mode values for imputation
        self._most_frequent = {}  # To store most frequent values for imputation
        self.isFitted = False  # Flag to check if the preprocessor has been fitted

    def fit(self, df: pd.DataFrame) -> Self:
        # Fit the preprocessor to the data (calculate median and mode for imputation)
        fill_strategy = self.config["preprocessing"]["fill_strategy"]

        """for column, strategy in fill_strategy.items():
            if strategy == "median":
                median_val = df[column].mean()  # compute from the DataFrame
                if pd.isna(median_val):
                    self._median_values[column] = 0.0  # fallback if all NaN
                else:
                    self._median_values[column] = median_val
                df[column] = df[column].fillna(self._median_values[column].mean())
               
            elif strategy == "mode":
                mode = self._modes.df[column].mode()
                if mode.empty:
                    self._modes[column] = 0  # Handle case where all values are NaN
                else:
                    self._modes[column] = mode[0]
                df[column] = df[column].fillna(self._modes[column])
                
            elif strategy == "most_frequent":
                #most_frequent = pd.to_numeric(self._most_frequent[column], errors='coerce')
                if column in self._most_frequent:
                    most_frequent = pd.to_numeric(self._most_frequent[column], errors='coerce')
                    if pd.isna(most_frequent):
                        
                        self._most_frequent[column] = "MALE"  # Handle case where all values are NaN
                    else:
                        self._most_frequent[column] = "FEMALE"  
                    df[column] = df[column].fillna(self._most_frequent[column])
            
                      
                

        self.isFitted = True
        return self"""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Transform the data (impute missing values and perform any other transformations)
        if not self.isFitted:
            raise ValueError("Preprocessor must be fitted before calling transform.")
        
        df = df.copy()  # Avoid modifying the original DataFrame
        

        """for column, median_val in self._median_values.items():
            df[column] = df[column].fillna(median_val)
        for column, mode_val in self._modes.items():
            df[column] = df[column].fillna(mode_val)"""
        
        
        """for column, median_val in self._median_values.items():
            df[column] = df[column].fillna(median_val)
        for column, mode_val in self._modes.items():
            df[column] = df[column].fillna(mode_val)"""   
        
        """if "sex" in encode_config:
            df["sex"] = df["sex"].map(encode_config["sex"]).astype(int)
             
        if encode_config.get("embarked") == "onehot":
            df = pd.get_dummies(df, columns=["embarked"], drop_first=True)
        return df"""
        encode_config = self.config["preprocessing"]["encode"]           