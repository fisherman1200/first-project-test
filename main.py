from trainers.train import train_model
from utils.config import load_config

if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    train_model(cfg)