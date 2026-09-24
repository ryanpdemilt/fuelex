class SegmentationTrainer:
    def __init__(
            self,
            model,
            train_loader,
            criterion,
            optimizer,
            lr_scheduler,
            evaluator,
            n_epochs,
            exp_dir,
            device,
            ckpt_interval,
            eval_interval
    ):
        pass

    def fit(self):
        pass

    def train(self):
        pass

    def train_one_epoch(self):
        pass

    def save_model(self):
        pass
    def load_model(self):
        pass
    def log(self):
        pass
    def get_checkpoint(self):
        pass

    def compute_loss(self):
        pass

    def compute_logging_metrics(self):
        pass

    def save_best_checkpoint(self):
        pass

class ImageSegmentationTrainer(SegmentationTrainer):
    def __init__(
            self
    ):
        pass

    def fit(self):
        pass

    def train(self):
        pass

    def train_one_epoch(self):
        pass

class PointwiseSegmentationTrainer(SegmentationTrainer):
    def __init__(
            self
    ):
        pass

    def fit(self):
        pass

    def train(self):
        pass

    def train_one_epoch(self):
        pass