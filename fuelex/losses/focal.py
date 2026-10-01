import torch
import torch.nn as nn

from segmentation_models_pytorch.losses.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from segmentation_models_pytorch.losses import FocalLoss

class FocalLoss(nn.Module):
    def __init__(
            self,
            mode,
            from_logits,
            eps,
            alpha,
            gamma,
            ignore_index,
            normalized,
            reduced_threshold,
            class_weights,
            reduction
    ):
            self.mode = mode
            self.from_logits = from_logits
            self.eps = eps
            self.alpha = alpha
            self.gamma = gamma
            self.ignore_index = ignore_index
            self.normalized = normalized
            self.reduced_threshold = reduced_threshold
            self.class_weights = class_weights
            self.reduction = reduction

            self.loss_fn = FocalLoss(
                mode = mode,
                eps = eps,
                alpha = alpha,
                gamma = gamma,
                ignore_index = ignore_index,
                normalized = normalized,
                reduced_threshold = reduced_threshold,
                class_weights = class_weights,
                reduction = reduction
            )

    def forward(self,y_pred,y_true):
        if not self.from_logits:
            y_pred = torch.clamp(y_pred, self.eps, 1 - self.eps)

            if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
                # inverse sigmoid
                y_pred = torch.log(y_pred / (1 - y_pred))

            elif self.mode == MULTICLASS_MODE:
                # convert softmax probabilities to log-space
                y_pred = torch.log(y_pred)

        loss = self.loss_fn(y_pred,y_true)

        return loss