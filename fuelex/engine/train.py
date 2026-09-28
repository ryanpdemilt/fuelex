import time
import hashlib
import pprint
from pathlib import Path

from yaml import dump

import mlflow

import torch
from torch.utils.data import DataLoader

from hydra.conf import HydraConf
from omegaconf import OmegaConf, DictConfig
from hydra import initialize,compose
from hydra.utils import instantiate, call

from fuelex.utils import fix_seed,init_logger, get_best_model_ckpt_path, get_final_model_ckpt_path
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
    
    task = choices["task"]
    exp_info = {
        "timestamp": timestamp,
        "model":model,
        "ds": ds,
        "task": task,
        "exp_name": f"{timestamp}_{cfg_hash}_{model}_{ds}",
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

    optimizer = instantiate(optimizer_cfg,params=params)

    return optimizer

def lr_scheduler_factory(cfg,optimizer,total_iters):
    lr_scheduler_cfg = cfg.lr_scheduler
    lr_scheduler = call(lr_scheduler_cfg,optimizer=optimizer,total_iters=total_iters)

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

    with initialize(config_path=config_path,version_base='1.4'):
        cfg = compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=True
        )

    epochs = cfg.task.trainer.n_epochs
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    test_num_workers = cfg.test_num_workers
    test_batch_size = cfg.test_batch_size

    fix_seed(cfg.seed)

    exp_info = get_exp_info(cfg.hydra)
    exp_name = exp_info['exp_name']
    task_name = exp_info['task']
    exp_dir = Path(cfg.work_dir) / exp_name
    exp_dir.mkdir(parents=True,exist_ok=True)
    logger_path = exp_dir / 'train.log'
    config_log_dir = exp_dir / 'configs'
    config_log_dir.mkdir(exist_ok=True)

    use_mlflow = cfg.mlflow.use_mlflow

    if use_mlflow:
        # mlflow_project = cfg.mlflow.project
        exp_name = f"{exp_info['model']}_{exp_info['ds']}"
        mlflow.set_tracking_uri(f"http://localhost:{cfg.mlflow.port}")
        mlflow.set_experiment(exp_name)
        active_run = mlflow.start_run()

    device = cfg.device

    logger = init_logger(logger_path)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    train_transform, test_transform, val_transform = preprocessor_factory(cfg)
    train_dataset, test_dataset, val_dataset = dataset_factory(cfg,(train_transform,test_transform,val_transform))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        dataset=test_dataset,
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

    logger.info("Built {} dataset.".format(cfg.dataset.dataset_name))

    logger.info(
        f"Total number of train images: {len(train_dataset)}\n"
        f"Total number of validation images: {len(val_dataset)}\n"
    )

    val_evaluator = instantiate(cfg.task.evaluator,exp_dir=exp_dir,device=device,val_loader=val_loader,use_mlflow=use_mlflow)
    test_evaluator = instantiate(cfg.task.evaluator,exp_dir=exp_dir,device=device,val_loader=test_loader,use_mlflow=use_mlflow)

    
    cfg.model.n_classes = train_dataset.n_classes
    cfg.model.encoder.classes = train_dataset.n_classes
    model = model_factory(cfg)
    logger.info("Built {}.".format(model.model_name))

    criterion = criterion_factory(cfg)
    optimizer = optimizer_factory(cfg,model.parameters())

    total_iters = len(train_loader) * epochs
    lr_scheduler = lr_scheduler_factory(cfg,optimizer=optimizer,total_iters=total_iters)

    trainer = instantiate(
        cfg.task.trainer,
        model=model,
        evaluator=val_evaluator,
        train_loader=train_loader,
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        criterion=criterion,
        exp_dir = exp_dir,
        device=device,
        use_mlflow=use_mlflow
    )

    

    trainer.fit()

    if cfg.use_final_ckpt:
        model_ckpt_path = get_final_model_ckpt_path(exp_dir)
    else:
        model_ckpt_path = get_best_model_ckpt_path(exp_dir)
        
    if model_ckpt_path is None and not cfg.task.trainer.model_name == "knn_probe":
        raise ValueError(f"No model checkpoint found in {exp_dir}")
    
    test_evaluator.evaluate(model, "test_model", model_ckpt_path)

    if use_mlflow:
        mlflow.end_run()


    