import sys
sys.path.append("../")

import torch
import torch.nn as nn
import numpy as np

class BaseTrainer:

    def __init__(self, cfg, model, train_loader, val_loader, test_loader, model_name):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_name = model_name

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            print("Training with GPU")
        else:
            self.device = "cpu"
            print("Warning: Training with CPU, might take a long time to finish")

        # variables that are tracked during model backprop
        self.log_variables = {
            "y0_trues": [],
            "y0_preds": [],
            "y1_trues": [],
            "y1_preds": [],
            "t_trues": [],
            "t_preds": [],
        }

        # names of the variables that we will keep track of
        # during training and validation
        self.track_metric_names = [
            "loss", "absATE",  "pehe", "factual_mae", "ite_true", "ite_pred"
        ]

        self.training_metrics = {k: [] for k in self.track_metric_names}
        self.validation_metrics = {k: [] for k in self.track_metric_names}

        def _init_log_variables(self):
            for k, _ in self.log_variables.items():
                    self.log_variables[k] = []

        def _init_training_metrics(self):
            for k, _ in self.training_metrics.items():
                self.training_metrics[k] = []

        def _init_validation_metrics(self):
            for k, _ in self.validation_metrics.items():
                self.validation_metrics[k] = []

        def get_score_checkpoint(self, epoch, val_loss):
            """Retrieves the path to a checkpoint file."""
            name = f"model_{epoch}_score={val_loss:4f}.pth"
            return os.path.join(self.cfg.PATHS.MODEL_OUT_DIR, name)

    def save_checkpoint(self, epoch, val_loss):
        """Saves a checkpoint."""

        sd = self.model.state_dict()
        # Record the state
        checkpoint = {
            "epoch": epoch,
            "model_state": sd,
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": self.cfg.dump(),
        }
        # Write the checkpoint
        checkpoint_file = self.get_score_checkpoint(epoch, val_loss)
        # print(f"saving to {checkpoint_file}")
        torch.save(checkpoint, checkpoint_file)
        return checkpoint_file

    def get_best_score_checkpoint(self):
        """Retrieves the checkpoint with lowest loss score."""
        checkpoint_dir = self.cfg.PATHS.MODEL_OUT_DIR
        # Checkpoint file names are in lexicographic order
        checkpoints = [f for f in os.listdir(checkpoint_dir) if ".pth" in f]
        best_checkpoint_val_loss = [
            float(
                ".".join(x.split("=")[1].split(".")[0:2])
            )
            for x in checkpoints
        ]
        best_idx = np.array(best_checkpoint_val_loss).argmin()
        name = checkpoints[best_idx]
        return os.path.join(checkpoint_dir, name)

    def load_best_score_checkpoint(self, load_optimizer=False):

        """Loads the checkpoint from the given file."""
        checkpoint_file = self.get_best_score_checkpoint()
        # Load the checkpoint on CPU to avoid GPU mem spike
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        # Account for the DDP wrapper in the multi-gpu setting
        self.model.load_state_dict(checkpoint["model_state"])
        # Load the optimizer state (commonly not done when fine-tuning)
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        return checkpoint["epoch"]

    def prepare_batch(self, batch_data):
        return {}

    def run_epoch(self, split, epoch_count=0):
       return {}

    def fit(self):
        model, cfg = self.model, self.cfg
        raw_model = model.module if hasattr(self.model, "module") else model
        self.optimizer, self.scheduler = raw_model.configure_optimizers(cfg)

        best_loss = float("inf")
        self._init_training_metrics()
        self._init_validation_metrics()
        for epoch_count in range(cfg.OPTIM.MAX_EPOCHS):
            train_epoch_metrics = self.run_epoch("train", epoch_count)
            self.scheduler.step()
            val_epoch_metrics = self.run_epoch("validation", epoch_count)
            # append the epoch metric into
            for k, v in train_epoch_metrics.items():
                self.training_metrics[k].append(v)
            for k, v in val_epoch_metrics.items():
                self.validation_metrics[k].append(v)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            val_loss = val_epoch_metrics["mse_loss"]
            is_good_model = val_loss < best_loss
            if is_good_model:
                best_loss = val_loss
                self.save_checkpoint(epoch_count, val_loss)

        self.plot_training_curves()
        print(f"Experiment logs stored at: {cfg.PATHS.OUT_DIR}")

class TrainerMIMIC(BaseTrainer):

    def __init__(self, cfg, model, train_dataloader, val_dataloader,  test_dataloader):
        super().__init__(cfg, model, train_dataloader, val_dataloader, test_dataloader)

    def prepare_batch(self, batch_data):
        return super().prepare_batch(batch_data)

    def run_epoch(self, split, epoch_count=0):
        return super().run_epoch(split, epoch_count=epoch_count)

    