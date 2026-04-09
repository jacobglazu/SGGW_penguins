from pathlib import Path

import yaml
from src.pipeline import Pipeline

def main():
    config_path = Path(__file__).parent / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pipeline = Pipeline(config)
    metrics = pipeline.run()
    print("Evaluation Metrics:")
    pipeline.run()

if __name__ == "__main__":
    main()