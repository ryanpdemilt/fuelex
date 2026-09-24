import torch

from hydra import initialize,compose
from hydra.utils import instantiate

def get_exp_info(hydra_config):
    pass

def run_train(self,args):
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    with initialize(config_path=config_path,version_base='1.4'):
        cfg = compose(
            config_name=config_name,
            overrides=[
                overrides
            ]
        )