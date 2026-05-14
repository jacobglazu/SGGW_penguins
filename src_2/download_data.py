import os

import yaml
from sklearn.datasets import fetch_openml

def main():

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    dataset_id = params["data"]["dataset_id"]
    print(f"Download data from Openml (id={dataset_id})...")

    data = fetch_openml(data_id=dataset_id, as_frame=True)
    df = data.frame

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/penguins.csv", index=False)
    print(f"Save data/penguins.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()



