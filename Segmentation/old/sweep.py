import wandb
from train import main
from sweep_config import sweep_config

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="segmentation_MBH")
    wandb.agent(sweep_id, function=main, count=50)