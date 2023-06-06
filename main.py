import pandas as pd
import torch
import numpy as np
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_regressors import get_model
from os.path import join
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import OmegaConf

import lightning as pl
from pytorch_lightning.strategies import DDPStrategy


@hydra.main(version_base=None, config_path="config", config_name="cambridge_train")
def main(cfg) -> None:

    # Initiate logger and output folder for the experiment
    log_path = utils.init_logger(outpath=cfg.inputs.output_path)
    utils.save_config_to_output_dir(log_path, cfg)

    # Record execution details
    logging.info("Start {} with {}".format(cfg.inputs.model_name, cfg.inputs.mode))
    if cfg.inputs.experiment is not None:
        logging.info("Experiment details: {}".format(cfg.inputs.experiment))
    logging.info("Using dataset: {}".format(cfg.inputs.dataset_path))
    logging.info("Using labels file: {}".format(cfg.inputs.trainset_path))

    # --------------------
    # Data
    # --------------------
    transform = utils.train_transforms.get('baseline')
    dataset = CameraPoseDataset(cfg.inputs.dataset_path, cfg.inputs.trainset_path, transform)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.general.batch_size, shuffle=True)
    random_sampler = torch.utils.data.RandomSampler(dataset, num_samples=train_dataloader.dataset.dataset_size // 10)
    val_dataloader = torch.utils.data.DataLoader(dataset,batch_size=cfg.general.batch_size, sampler=random_sampler)

    transform = utils.test_transforms.get('baseline')
    dataset = CameraPoseDataset(cfg.inputs.dataset_path, cfg.inputs.testset_path, transform)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.general.batch_size, shuffle=False)

    #--------------------
    # Model
    # --------------------
    model_config = OmegaConf.to_container(cfg[cfg.inputs.model_name])
    model = get_model(cfg.inputs.model_name, model_config)

    # Load the checkpoint if needed
    # if cfg.inputs.checkpoint_path:
    #     model.load_state_dict(torch.load(cfg.inputs.checkpoint_path))
    #     logging.info("Initializing from checkpoint: {}".format(cfg.inputs.checkpoint_path))

    # --------------------
    # Trainer
    # --------------------
    # Set the seeds and the device
    pl.seed_everything()

    trainer = pl.Trainer(max_epochs=cfg.general.n_epochs,
                         accelerator='gpu',
                         devices=3,
                         strategy='ddp_find_unused_parameters_true')
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
