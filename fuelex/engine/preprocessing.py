import torch

from torchvision.transforms.v2 import Compose, RandomResizedCrop, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip,Identity

from hydra.utils import instantiate

def build_transform(cfg):
    ops = []
    for transform_name in cfg.keys():
        t = instantiate(cfg[transform_name])
        ops.append(t)

    transform = Compose(ops)

    return transform