import wandb
from train import train_model  # wrapper qui lance le training avec cfg

def sweep_train():
    # Chaque run reçoit un cfg mis à jour par W&B
    cfg = wandb.config
    train_model(cfg)

if __name__ == "__main__":
    import sweep_config

    sweep_id = wandb.sweep(sweep_config.sweep_config, project="segmentation_sweep")
    wandb.agent(sweep_id, function=sweep_train, count=20)  # 20 essais max