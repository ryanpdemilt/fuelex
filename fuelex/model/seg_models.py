import torch
import segmentation_models_pytorch as smp

def unet_resnet34(in_channels=64,classes=40):
    return smp.UNet(
        encoder_name='resnet34',
        in_channels=in_channels,
        classes=classes
    )

def unet_plus_plus_resnet34(in_channels=64,classes=40):
    return smp.UNet(
        encoder_name='resnet34',
        in_channels=in_channels,
        classes=classes
    )