import torch

def MultiStepLR(optimizer,total_iters,lr_milestones,**kwargs):

    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[int(total_iters * r) for r in lr_milestones]
        **kwargs
)