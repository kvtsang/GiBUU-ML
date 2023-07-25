#!/usr/bin/env python3

import yaml
import torch
import fire
import os
import time
import sys
import shutil
import lightning.pytorch as pl

from gibuu_ml.utils import import_from
from gibuu_ml.io import dataloader_factory

def train(
    cfg_file, 
    lr=None, load=None, resume=None, 
    max_epochs=10000, uid=False, log=None,
):
    # check input arguments
    assert (not load) or (not resume), \
        '--load and --resume cannot be used together'

    # prepare config dict
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)

    if lr is not None:
        opt_cfg = cfg.setdefault('optimizer', {})
        opt_cfg['lr'] = lr

    # dataloader
    dataloader, val_dataloader = dataloader_factory(cfg)

    # uid
    pid = os.getpid()
    ts = int(time.time())

    # -------------------------------------------------------------------------
    # logger
    # -------------------------------------------------------------------------
    logger_cfg = cfg.setdefault('logger', {})

    if log is None:
        log_dir = logger_cfg.get('save_dir', 'logs')
        log_name = logger_cfg.get('name', None)
    else:
        log_dir, log_name = os.path.split(log)

    if log_name is None:
        log_name = 'gibuu'
        uid = True

    if uid:
        log_name = f'{log_name}_{ts:x}_{pid}'

    logger_cfg['save_dir'] = log_dir
    logger_cfg['name'] = log_name
    logger = pl.loggers.CSVLogger(**logger_cfg)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    Model = import_from(cfg['class']['model'])
    runtime_cfg = cfg.setdefault('runtime', {})

    if load is None:
        model = Model(cfg)
    else:
        print(f'[INFO] load {load}')
        model = Model.load_from_checkpoint(load, strict=False, cfg=cfg)
        runtime_cfg['load'] = load

    if resume is not None:
        runtime_cfg['resume'] = resume

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    trainer_cfg = cfg.get('trainer', {}).copy()
    if trainer_cfg.get('precision') == 'bf16-mixed':
        torch.set_float32_matmul_precision('medium')

    callbacks_cfg = trainer_cfg.pop('callbacks',  {})
    callbacks = []
    for class_name, callback_cfg in callbacks_cfg.items():
        Cls = getattr(pl.callbacks, class_name)
        callbacks.append(Cls(**callback_cfg))

    # -------------------------------------------------------------------------
    # Trainer 
    # -------------------------------------------------------------------------
    trainer_cfg.update(dict(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
    ))
    trainer = pl.Trainer(**trainer_cfg)

   # -------------------------------------------------------------------------
    # save cfg file
    # -------------------------------------------------------------------------
    cfg_dir = logger.root_dir
    if not os.path.isdir(cfg_dir):
        os.makedirs(cfg_dir)
    cfg_file = os.path.join(cfg_dir, 'config.yaml')
    with open(cfg_file, 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    # -------------------------------------------------------------------------
    # start training
    # -------------------------------------------------------------------------
    if resume is None:
        trainer.fit(model, dataloader, val_dataloader)
    else:
        print(f'[INFO] resume {resume}')
        trainer.fit(model, dataloader, val_dataloader, ckpt_path=resume)

    # -------------------------------------------------------------------------
    # mv cfg file to log_dir
    # -------------------------------------------------------------------------
    shutil.copy(cfg_file, logger.log_dir)

if __name__ == '__main__':
    fire.Fire(train)
