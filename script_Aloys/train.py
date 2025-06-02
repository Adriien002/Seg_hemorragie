import argparse
import os
import pprint
import warnings
import torch
import yaml
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger, NeptuneLogger
from src.modules.classification.classifier_module import ClassifierModule
from src.utils.dataset import make_rsna_slice_dataloaders

warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40960, rlimit[1]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the stage_2_train folder')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to the train_foldx.csv file')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to the val_foldx.csv file')
    parser.add_argument('--neptune_config', type=str, required=False, help='Path to neptune configuration file')
    parser.add_argument('--with_id', type=str, required=False, help='Neptune run ID')
    parser.add_argument('--resume', type=str, required=False, help='Path to checkpoint for resuming training')
    return parser.parse_args()


def train(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['data']['root_path'] = args.data_path
    config['data']['train_csv'] = args.train_csv
    config['data']['val_csv'] = args.val_csv

    print("#########")
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(config)
    print("#########")

    pl.seed_everything(config.get("seed", 42))
    torch.set_float32_matmul_precision("high")

    trn_loader, val_loader = make_rsna_slice_dataloaders(config)
    pl_module = ClassifierModule(config)

    # Setup logger
    if 'SLURM_JOB_ID' in os.environ or not args.neptune_config:
        logger = TensorBoardLogger(save_dir='log_dir')
    else:
        if args.neptune_config:
            with open(args.neptune_config) as f:
                neptune_config = yaml.safe_load(f)

        if args.with_id is not None:
            neptune_config['with_id'] = args.with_id

        logger = NeptuneLogger(**neptune_config)

    # Create trainer
    trainer = pl.Trainer(
        benchmark=True,
        precision=config["training"].get("precision", "16-mixed"),
        logger=logger,
        max_epochs=config["training"].get("max_epochs", 50),
        num_sanity_val_steps=0 if torch.cuda.device_count() < 1 else None,
        strategy='auto',
    )

    trainer.fit(pl_module, train_dataloaders=trn_loader, val_dataloaders=val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    args = parse_args()
    train(args)