import argparse
import logging
import os
import yaml

import pytorch_lightning as pl
from easydict import EasyDict
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from .dataset import StrDataset
from .model import LQAE
from .util import seed_all


def train(config):
    seed_all(config.seed)
    
    dataset_config = config.dataset
    train_config = config.train
    logger_config = config.logging
    
    #create and parse dataset
    all_dataset = StrDataset(dataset_config)
    train_size = int(dataset_config.train_ratio * len(all_dataset))
    val_size = len(all_dataset) - train_size
    train_dataset, val_dataset = random_split(all_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=train_config.shuffle,
        num_workers=train_config.num_workers,
        collate_fn=StrDataset.featurize
    )

    validation_dataloader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=StrDataset.featurize
    )

    model = LQAE(config)
    
    logger = (
        pl.loggers.WandbLogger(project=config.logging.wandb_project)
        if config.logging.wandb_project is not None
        else True
    )

    logging.info(f"Using LQAE model for training")
    lr_logger = pl.callbacks.LearningRateMonitor()

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.train.save_path,
        mode='min',
        monitor=train_config.save_monitor,
        save_top_k=train_config.save_top_k,
        every_n_train_steps=train_config.save_every_n_step,
        save_last=True,
    )
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_logger],
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        strategy=train_config.strategy,
        fast_dev_run=train_config.fast_dev_run,
        accelerator=train_config.accelerator,
        devices=train_config.gpu_device if train_config.accelerator == 'gpu' else train_config.cpu_device,
        gradient_clip_val=train_config.gradient_clip_val,
        log_every_n_steps=logger_config.log_every_n_steps,
        max_epochs=train_config.epochs,
        num_nodes=train_config.num_nodes,
        precision=train_config.precision,
        resume_from_checkpoint = None if not train_config.resume else train_config.resume_from_checkpoint,
        track_grad_norm=logger_config.track_grad_norm,
        val_check_interval=train_config.val_check_interval,
    )

    trainer.fit(model, train_dataloader, validation_dataloader)


def get_args():
    parser = argparse.ArgumentParser(description="EnTex project training script")
    parser.add_argument("--config_path", default='./entex/configs/entex_test.yml', help="path to load yaml-like configurations")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_DIR'] = './wandb/'
    
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
        
    train(config=config)