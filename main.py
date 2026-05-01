from pathlib import Path
import yaml
from src.pipeline import PenguinPipeline
from src.trainer import ModelTrainer


def main():
    config_path = Path(__file__).parent / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    

    pipeline = PenguinPipeline(config)
    print("Evaluation Metrics:")
    metrics = pipeline.run()
    
    #return metrics
    

if __name__ == "__main__":
    main()
    