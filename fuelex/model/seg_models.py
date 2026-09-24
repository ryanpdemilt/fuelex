import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SingleSourceSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder_name = 'unet_resnet34',
        in_channels = 64,
        n_classes = 40,
        source='aef'
    ):
        super().__init__()

        if encoder_name == 'unet_resnet34':
            self.encoder = unet_resnet34(
                in_channels=in_channels,
                n_classes=n_classes
            )

def unet_resnet34(in_channels=64,n_classes=40):
    return smp.UNet(
        encoder_name='resnet34',
        in_channels=in_channels,
        classes=n_classes
    )

def unet_plus_plus_resnet34(in_channels=64,n_classes=40):
    return smp.UnetPlusPlus(
        encoder_name='resnet34',
        in_channels=in_channels,
        classes=n_classes
    )