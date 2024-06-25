import argparse
import logging
import os
import yaml

import pytorch_lightning as pl
from easydict import EasyDict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split

from .dataset import StructureDataset
#from .models.model_llama2 import LQAE as LqaeLlama2
from .models.model_opt import LQAE as LqaeOpt
from .utils.util import seed_all


def train(config):
    seed_all(config.seed)
    
    train_config = config.train
    logger_config = config.logging
    
    model = LqaeOpt(config.model, config.optimizer)
    
    dataset_config = config.dataset
    dataset = StructureDataset(dataset_config)
    
    train_size = int(dataset_config.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_set, valid_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        collate_fn=StructureDataset.featurize,
        shuffle=train_config.shuffle,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        collate_fn=StructureDataset.featurize,
        shuffle=False,
        drop_last=True,
    )
    
    logger = (
        pl.loggers.WandbLogger(project=''.join(config.logging.wandb_project))
        if config.logging.wandb_project is not None
        else True
    )

    logging.info(f"Using LQAE model for training")    
    lr_logger = pl.callbacks.LearningRateMonitor()

    #early_stopping_callback = EarlyStopping(monitor=train_config.stop_monitor, mode='min')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=''.join(config.train.save_path),
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
        resume_from_checkpoint = None if not train_config.resume else ''.join(train_config.resume_from_checkpoint),
        track_grad_norm=logger_config.track_grad_norm,
        val_check_interval=train_config.val_check_interval,
        num_sanity_val_steps=train_config.num_sanity_val_steps,
    )
    trainer.fit(model, train_loader, valid_loader)


def get_args():
    parser = argparse.ArgumentParser(description="EnTex project training script")
    parser.add_argument("--config", default='entex/configs/test.yml', help="path to load yaml-like configurations")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #os.environ['WANDB_MODE'] = 'dryrun'
    #os.environ['WANDB_DIR'] = 'wandb/optexp1'
    
    args = get_args()
    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
        
    train(config=config)