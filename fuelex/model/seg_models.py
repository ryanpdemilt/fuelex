import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet, UnetPlusPlus

from hydra.utils import instantiate

class SingleSourceSegmentationModel(nn.Module):
    def __init__(
        self,
        in_modality: str,
        in_channels: int,
        n_classes: int,
        encoder: nn.Module
    ):
        super().__init__()

        self.in_modality = in_modality
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        self.encoder = encoder
        print(encoder)

    def forward(self,inputs):
        image = inputs[self.in_modality]

        logits = self.encoder(image)

        return logits
    
# def unet_resnet34(in_channels=64,n_classes=40):

#     return smp.Unet(
#         encoder_name='resnet34',
#         in_channels=in_channels,
#         classes=n_classes
#     )

# def unet_plus_plus_resnet34(in_channels=64,n_classes=40):
#     return smp.UnetPlusPlus(
#         encoder_name='resnet34',
#         in_channels=in_channels,
#         classes=n_classes
#     )