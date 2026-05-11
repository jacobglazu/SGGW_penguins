from pathlib import Path
import yaml
from src.pipeline import PenguinPipeline
from src.trainer import ModelTrainer
#import sys
#sys.path.append("D:/Kuba/SGGW_penguins")
#from notebooks.train import train


def main():
    config_path = Path(__file__).parent / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    

    pipeline = PenguinPipeline(config)
    print("Evaluation Metrics:")
    metrics = pipeline.run()
    #train(100,6)
    
    #return metrics
    

if __name__ == "__main__":
    main()
    