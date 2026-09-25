import time
import hashlib
from pathlib import Path

import mlflow

import torch
from torch.utils.data import DataLoader

from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from hydra import initialize,compose
from hydra.utils import instantiate

from fuelex.utils import fix_seed,init_logger
from .preprocessing import build_transform

def get_exp_info(hydra_config: HydraConf) -> dict[str, str]:
    """Create a unique experiment name based on the choices made in the config.

    Args:
        hydra_config (HydraConf): hydra config.

    Returns:
        str: experiment information.
    """
    choices = OmegaConf.to_container(hydra_config.runtime.choices)
    cfg_hash = hashlib.sha1(
        OmegaConf.to_yaml(hydra_config).encode(), usedforsecurity=False
    ).hexdigest()[:6]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model = choices['model']
    ds = choices["dataset"]
    groups = hydra_config.dataset.groups
    
    task = choices["task"]
    exp_info = {
        "timestamp": timestamp,
        "model":model,
        "ds": ds,
        "task": task,
        "exp_name": f"{timestamp}_{cfg_hash}_{model}_{ds}_{'_'.join(groups)}",
    }
    return exp_info

def model_factory(cfg):
    model_cfg = cfg.model

    model = instantiate(model_cfg)

    return model

def dataset_factory(cfg,preprocessors):
    dataset_cfg = cfg.dataset

    train_transform, test_transform, val_transform = preprocessors

    train_dataset = instantiate(
        cfg.dataset,
        split="train",
        transform=train_transform
    )
    test_dataset = instantiate(
        cfg.dataset,
        split="test",
        transform=test_transform
    )
    val_dataset = instantiate(
        cfg.dataset,
        split="val",
        transform=val_transform
    )

    return train_dataset, test_dataset, val_dataset

def criterion_factory(cfg):
    criterion_cfg = cfg.criterion

    criterion = instantiate(criterion_cfg)

    return criterion

def optimizer_factory(cfg,params):
    optimizer_cfg = cfg.optimizer

    optimizer = instantiate(optimizer_cfg,params)

    return optimizer

def lr_scheduler_factory(cfg,optimizer,total_iters):
    lr_scheduler_cfg = cfg.lr_scheduler
    lr_scheduler = instantiate(lr_scheduler_cfg,optimizer=optimizer,total_iters=total_iters)

    return lr_scheduler

def preprocessor_factory(cfg):
    preprocessor_cfg = cfg.preprocessing

    train_transform = build_transform(preprocessor_cfg.train)
    test_transform = build_transform(preprocessor_cfg.test)
    val_transform = build_transform(preprocessor_cfg.val)

    return train_transform, test_transform, val_transform

    

def run_train(args):
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    epochs = cfg.task.n_epochs
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    test_num_workers = cfg.test_num_workers
    test_batch_size = cfg.test_batch_size

    with initialize(config_path=config_path,version_base='1.4'):
        cfg = compose(
            config_name=config_name,
            overrides=[
                overrides
            ]
        )

    fix_seed(cfg.seed)

    exp_info = get_exp_info(cfg)
    exp_name = exp_info['exp_name']
    task_name = exp_info['task']
    exp_dir = Path(cfg.work_dir) / exp_name
    exp_dir.mkdir(parents=True,exist_ok=True)
    logger_path = exp_dir / 'train.log'
    config_log_dir = exp_dir / 'configs'
    config_log_dir.mkdir(exist_ok=True)

    use_mlflow = cfg.mlflow.use_mlflow

    if use_mlflow:
        mlflow_project = cfg.mlflow.project
        mlflow.set_tracking_uri(f"http://localhost:{cfg.mlflow.port}")
        active_run = mlflow.start_run(
            run_id=exp_name,
            experiment_id=mlflow_project
        )

    device = cfg.device

    train_transform, test_transform, val_transform = preprocessor_factory(cfg)
    train_dataset, test_dataset, val_dataset = dataset_factory(cfg,(train_transform,test_transform,val_transform))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        datset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=test_num_workers
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    val_evaluator = instantiate(cfg.task.evaluator,val_loader=val_loader)
    test_evaluator = instantiate(cfg.task.evaluator,val_loader=test_loader)

    cfg.model.n_classes = cfg.model.encoder.n_classes = train_dataset.n_classes
    model = model_factory(cfg)

    criterion = criterion_factory(cfg)
    optimizer = optimizer_factory(cfg,model.parameters())

    total_iters = len(train_loader) * cfg.n_epochs
    lr_scheduler = lr_scheduler_factory(cfg,optimizer=optimizer,total_iters=total_iters)

    trainer = instantiate(
        cfg.task.trainer,
        model=model,
        train_loader=train_loader,
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        criterion=criterion,
        exp_dir = exp_dir,
        device=device 
    )

    logger = init_logger(logger_path)

    trainer.fit()

    if use_mlflow:
        mlflow.end_run()


    