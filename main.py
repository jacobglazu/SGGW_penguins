from pathlib import Path

from sklearn.pipeline import Pipeline
import yaml
from src.pipeline import PenguinPipeline

def main():
    config_path = Path(__file__).parent / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pipeline = PenguinPipeline(config)
    metrics = pipeline.run()
    print("Evaluation Metrics:")
    

if __name__ == "__main__":
    main()