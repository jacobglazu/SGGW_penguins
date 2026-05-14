import os
import pickle
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    df = pd.read_csv("data/penguins.csv")
    
   
    df = df.dropna()

    X = df.drop(columns=["species"])
    y = df["species"]

    cat_features = ["island", "sex"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_encoded = encoder.fit_transform(X[cat_features])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(cat_features), index=X.index)
    
    X_final = pd.concat([X.drop(columns=cat_features), X_encoded_df], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=params["prepare"]["test_size"], random_state=params["prepare"]["random_state"]
    )

    os.makedirs("models", exist_ok=True)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    with open("models/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    
    print("Preprocessing done. Encoder save.")

if __name__ == "__main__":
    main()