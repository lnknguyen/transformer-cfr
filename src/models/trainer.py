import sys
sys.path.append("../")

import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim
import os
import matplotlib.pyplot as plt
import seaborn as sns

class BaseTrainer:

    def __init__(self, cfg, model, train_loader, val_loader, test_loader, model_name):

        self.cfg = cfg
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
            "loss", "absATE", "absATT", "pehe", "factual_mae", "ite_true", "ite_pred"
        ]

        self.training_metrics = {k: [] for k in self.track_metric_names}
        self.validation_metrics = {k: [] for k in self.track_metric_names}

    def _init_save_path(self):

        self.out_name = os.path.join(self.cfg.PATH.MODEL_OUT_DIR, self.cfg.UTILS.TIMESTAMP)
        if not os.path.exists(self.out_name):
            os.makedirs(self.out_name)
            
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
        return os.path.join(self.cfg.PATH.MODEL_OUT_DIR, name)

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
        checkpoint_dir = self.cfg.PATH.MODEL_OUT_DIR
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

    def plot_training_curve(self):
        
        # Get save path
        loss_fig_save_path = os.path.join(self.cfg.PATH.MODEL_OUT_DIR, "training_losses.png")
        comp_fig_save_path = os.path.join(self.cfg.PATH.MODEL_OUT_DIR, "comparison.png")
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        min_loss_at = np.argmin(np.array(self.validation_metrics["factual_mae"]))
        min_abs_ate_at = np.argmin(np.array(self.validation_metrics["absATE"]))
        min_pehe_at = np.argmin(np.array(self.validation_metrics["pehe"]))
        min_abs_att_at = np.argmin(np.array(self.validation_metrics["absATT"]))
        
        # 1. Print train/val loss
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), constrained_layout=True)
        x_grid = np.arange(len(self.training_metrics["loss"])) + 1
        sns.lineplot(x=x_grid, y=self.validation_metrics["loss"], ax=ax, marker="o", label='val loss')
        sns.lineplot(x=x_grid, y=self.training_metrics["loss"], ax=ax, marker="o", label='train loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses vs Epoch')
        ax.legend()
        fig.savefig(loss_fig_save_path)

        # 2. PEHE + ATE + ATT vs MAE loss
        fig1, ax1 = plt.subplots(nrows=3, ncols=1, figsize=(15, 8), constrained_layout=True)
        x_grid = np.arange(len(self.validation_metrics["absATE"]))

        # 1. plot factual mae vs absATE
        lns1 = sns.lineplot(x_grid, self.validation_metrics["factual_mae"], ax=ax1.flat[0], marker="o", color=cycle[0], label='Factual MAE')
        ax1.flat[0].axvline(min_loss_at, color=cycle[0], linestyle="--")
        ax1.flat[0].set_xlabel('Epochs')
        ax1.flat[0].set_ylabel('Factual MAE')

        ax1_0 = ax1.flat[0].twinx()
        lns2 = sns.lineplot(x_grid, self.validation_metrics["absATE"], ax=ax1_0, marker="o", color=cycle[1], label='Abs. ATE')
        ax1_0.axvline(min_abs_ate_at, color=cycle[1], linestyle="--")
        ax1_0.set_ylabel('Abs ATE')

        ax1.flat[0].set_title('Validation Losses: Factual MAE and Abs. ATE')
        leg = lns1.get_lines() + lns2.get_lines()
        labs = [l.get_label() for l in leg]
        ax1.flat[0].legend(leg, labs, loc=7)
        ax1_0.get_legend().remove()


        # 2. plot factual mae vs pehe
        lns1 = sns.lineplot(x=x_grid, y=self.validation_metrics["factual_mae"], ax=ax1.flat[1], color=cycle[0], marker="o", label='Factual MAE')
        ax1.flat[1].axvline(min_loss_at, color=cycle[0], linestyle="--")
        ax1.flat[1].set_xlabel('Epochs')
        ax1.flat[1].set_ylabel('Factual MAE')

        ax1_1 = ax1.flat[1].twinx()
        lns2 = sns.lineplot(x=x_grid, y=self.validation_metrics["pehe"], ax=ax1_1, marker="o", color=cycle[1], label='PEHE')
        ax1_1.axvline(min_pehe_at, color=cycle[1], linestyle="--")
        ax1_1.set_ylabel('PEHE')

        ax1.flat[1].set_title('Validation Losses: Factual MAE and PEHE')
        leg = lns1.get_lines() + lns2.get_lines()
        labs = [l.get_label() for l in leg]
        ax1.flat[1].legend(leg, labs, loc=7)
        ax1_1.get_legend().remove()
        
        # 3. plot factual mae vs ATT
        lns1 = sns.lineplot(x_grid, self.validation_metrics["factual_mae"], ax=ax1.flat[2], marker="o", color=cycle[0], label='Factual MAE')
        ax1.flat[2].axvline(min_loss_at, color=cycle[0], linestyle="--")
        ax1.flat[2].set_xlabel('Epochs')
        ax1.flat[2].set_ylabel('Factual MAE')

        ax1_2 = ax1.flat[2].twinx()
        lns2 = sns.lineplot(x_grid, self.validation_metrics["absATT"], ax=ax1_2, marker="o", color=cycle[1], label='Abs. ATT')
        ax1_2.axvline(min_abs_att_at, color=cycle[1], linestyle="--")
        ax1_2.set_ylabel('ATT')

        ax1.flat[2].set_title('Validation Losses: Factual MAE and Abs. ATT')
        leg = lns1.get_lines() + lns2.get_lines()
        labs = [l.get_label() for l in leg]
        ax1.flat[2].legend(leg, labs, loc=7)
        ax1_2.get_legend().remove()
        
        fig1.savefig(comp_fig_save_path)
        
    def fit(self):
       
        #TODO: move to config

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.TRAIN.LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, eta_min=1e-5)
        best_loss = float("inf")

        self._init_training_metrics()
        self._init_validation_metrics()
        self._init_save_path()

        for epoch in range(1, self.cfg.TRAIN.EPOCHS + 1):
            print(f"running epoch {epoch}")
            self.scheduler.step()
            train_epoch_metrics = self.run_epoch("train", epoch)
            val_epoch_metrics = self.run_epoch("val", epoch)

            # append the epoch metric into
            for k, v in train_epoch_metrics.items():
                self.training_metrics[k].append(v)
            for k, v in val_epoch_metrics.items():
                self.validation_metrics[k].append(v)

            val_loss = val_epoch_metrics["loss"]

            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                
        self.plot_training_curve()
#        self.save_val_metrics()
        print(self.validation_metrics)
        

'''
Trainer class for MNIST dataset
'''
class TrainerMNIST(BaseTrainer):

    def __init__(self, cfg, model, train_dataloader, val_dataloader,  test_dataloader, model_name):
        super().__init__(cfg, model, train_dataloader, val_dataloader, test_dataloader, model_name)

    def prepare_batch(self, batch_data):
        
        xs_1 = batch_data[0].to(self.device)
        xs_2 = batch_data[1].to(self.device)
        xs_3 = batch_data[2].to(self.device)
        t = batch_data[3].to(self.device)
        #z = batch_data[4].to(self.device)
        yf = batch_data[5].to(self.device)
        y0 = batch_data[6].to(self.device)
        y1 = batch_data[7].to(self.device)
        
        return xs_1, xs_2, xs_3, t, yf, y0, y1

    def run_epoch(self, split, epoch_count=0):
        self._init_log_variables()

        epoch_metrics = {}
        if split.lower() == "train":
            loader = self.train_loader
        elif split.lower() == "val":
            loader = self.val_loader
        elif split.lower() == "test":
            loader = self.test_loader
        
        is_train = True if split.lower() == "train" else False
        losses = []
        self.model.train(is_train)

        for batch_idx, batch_data in enumerate(loader):
            
            xs_1, xs_2, xs_3, t, yf, y0, y1 = self.prepare_batch(batch_data)

            with torch.set_grad_enabled(is_train):
                
                loss, yf_pred, y0_pred, y1_pred, t_logits = self.model(xs_1, xs_2, xs_3, t, yf)
            
                t_pred = torch.round(torch.sigmoid(t_logits))

                loss = (
                    loss.mean()
                )  
                losses.append(loss.item())

                self.log_variables["y0_trues"].extend(y0.cpu().detach().numpy().tolist())
                self.log_variables["y0_preds"].extend(y0_pred.cpu().detach().numpy().tolist())
                self.log_variables["y1_trues"].extend(y1.cpu().detach().numpy().tolist())
                self.log_variables["y1_preds"].extend(y1_pred.cpu().detach().numpy().tolist())
                self.log_variables["t_trues"].extend(t.cpu().detach().numpy().tolist())
                self.log_variables["t_preds"].extend(t_pred.cpu().detach().numpy().tolist())

                if is_train:
                    # backprop and update the parameters
                    self.model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 0.5
                    )

                    self.optimizer.step()
            
         #Compute metrics
        if not is_train:

            # compute metrics
            for k, v in self.log_variables.items():
                self.log_variables[k] = np.array(v)

            yf_true = np.where(
                self.log_variables["t_trues"] == 0,
                self.log_variables["y0_trues"],
                self.log_variables["y1_trues"]
            )
            yf_pred = np.where(
                self.log_variables["t_trues"] == 0,
                self.log_variables["y0_preds"],
                self.log_variables["y1_preds"]
            )

            ycf_true = np.where(
                self.log_variables["t_trues"] == 0,
                self.log_variables["y1_trues"],
                self.log_variables["y0_trues"]
            )
            ycf_pred = np.where(
                self.log_variables["t_trues"] == 0,
                self.log_variables["y1_preds"],
                self.log_variables["y0_preds"]
            )

            ## Computer ATT
            treated_idx = (self.log_variables["t_trues"] == 0)

            y1_true_treated = self.log_variables["y1_trues"][treated_idx]
            y0_true_treated = self.log_variables["y0_trues"][treated_idx]

            y1_pred_treated = self.log_variables["y1_preds"][treated_idx]
            y0_pred_treated = self.log_variables["y0_preds"][treated_idx]

            pred_att = (y1_pred_treated - y0_pred_treated)
            true_att = (y1_true_treated - y0_true_treated)

            epoch_metrics["factual_mae"] = (yf_true - yf_pred).mean()

            epoch_metrics["loss"] = float(np.mean(losses))
            ite_true = self.log_variables["y1_trues"] - self.log_variables["y0_trues"]
            ite_pred = self.log_variables["y1_preds"] - self.log_variables["y0_preds"]

            del_fcf_true = yf_true - ycf_true
            del_fcf_pred = yf_pred - ycf_pred

            epoch_metrics["ite_true"] = ite_true.mean()
            epoch_metrics["ite_pred"] = ite_pred.mean()

            epoch_metrics["pehe"] = np.sqrt(np.mean(np.square((ite_true) - (ite_pred))))
            epoch_metrics["absATE"] = np.abs(np.mean(ite_true) - np.mean(ite_pred))
            epoch_metrics["absATT"] = np.abs(np.mean(pred_att - true_att))
            
            epoch_metrics["loss"] = float(np.mean(losses))            
            print_str = f"{split} epoch: {epoch_count}\t "
            for k, v in epoch_metrics.items():
                print_str += f"{k}: {v:.5f} "
        
            print(print_str)
        else:
            epoch_metrics["loss"] = float(np.mean(losses))

        return epoch_metrics  
    
'''
Trainer class for MIMIC dataset
'''
class TrainerMIMIC(BaseTrainer):

    def __init__(self, cfg, model, train_dataloader, val_dataloader,  test_dataloader, model_name):
        super().__init__(cfg, model, train_dataloader, val_dataloader, test_dataloader, model_name)
    
    def prepare_batch(self, batch_data):
        
        x_diag = batch_data[0].to(self.device)
        x_age = batch_data[1].to(self.device)
        x_gender = batch_data[2].to(self.device)
        t = batch_data[3].to(self.device)
        yf = batch_data[4].to(self.device)
        y0 = batch_data[5].to(self.device)
        y1 = batch_data[6].to(self.device)
        
        return x_diag, x_age, x_gender, t, yf, y0, y1

    def run_epoch(self, split, epoch_count=0):
        self._init_log_variables()

        epoch_metrics = {}
        if split.lower() == "train":
            loader = self.train_loader
        elif split.lower() == "val":
            loader = self.val_loader
        elif split.lower() == "test":
            loader = self.test_loader
        
        is_train = True if split.lower() == "train" else False
        losses = []
        self.model.train(is_train)

        for batch_idx, batch_data in enumerate(loader):
            
            x_diag, x_age, x_gender, t, yf, y0, y1 = self.prepare_batch(batch_data)

            with torch.set_grad_enabled(is_train):
                
                loss, yf_pred, y0_pred, y1_pred, t_logits = self.model(x_diag, x_age, x_gender, t, yf, y0, y1)
            
                t_pred = torch.round(torch.sigmoid(t_logits))

                loss = (
                    loss.mean()
                )  
                losses.append(loss.item())

                self.log_variables["y0_trues"].extend(y0.cpu().detach().numpy().tolist())
                self.log_variables["y0_preds"].extend(y0_pred.cpu().detach().numpy().tolist())
                self.log_variables["y1_trues"].extend(y1.cpu().detach().numpy().tolist())
                self.log_variables["y1_preds"].extend(y1_pred.cpu().detach().numpy().tolist())
                self.log_variables["t_trues"].extend(t.cpu().detach().numpy().tolist())
                self.log_variables["t_preds"].extend(t_pred.cpu().detach().numpy().tolist())

                if is_train:
                    # backprop and update the parameters
                    self.model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 0.5
                    )

                    self.optimizer.step()
            
         #Compute metrics
        if not is_train:

            # compute metrics
            for k, v in self.log_variables.items():
                self.log_variables[k] = np.array(v)

            yf_true = np.where(
                self.log_variables["t_trues"] == 0,
                self.log_variables["y0_trues"],
                self.log_variables["y1_trues"]
            )
            yf_pred = np.where(
                self.log_variables["t_trues"] == 0,
                self.log_variables["y0_preds"],
                self.log_variables["y1_preds"]
            )

            ycf_true = np.where(
                self.log_variables["t_trues"] == 0,
                self.log_variables["y1_trues"],
                self.log_variables["y0_trues"]
            )
            ycf_pred = np.where(
                self.log_variables["t_trues"] == 0,
                self.log_variables["y1_preds"],
                self.log_variables["y0_preds"]
            )

            ## Computer ATT
            treated_idx = (self.log_variables["t_trues"] == 0)

            y1_true_treated = self.log_variables["y1_trues"][treated_idx]
            y0_true_treated = self.log_variables["y0_trues"][treated_idx]

            y1_pred_treated = self.log_variables["y1_preds"][treated_idx]
            y0_pred_treated = self.log_variables["y0_preds"][treated_idx]

            pred_att = (y1_pred_treated - y0_pred_treated)
            true_att = (y1_true_treated - y0_true_treated)

            epoch_metrics["factual_mae"] = (yf_true - yf_pred).mean()

            epoch_metrics["loss"] = float(np.mean(losses))
            ite_true = self.log_variables["y1_trues"] - self.log_variables["y0_trues"]
            ite_pred = self.log_variables["y1_preds"] - self.log_variables["y0_preds"]

            del_fcf_true = yf_true - ycf_true
            del_fcf_pred = yf_pred - ycf_pred

            epoch_metrics["ite_true"] = ite_true.mean()
            epoch_metrics["ite_pred"] = ite_pred.mean()

            epoch_metrics["pehe"] = np.sqrt(np.mean(np.square((ite_true) - (ite_pred))))
            epoch_metrics["absATE"] = np.abs(np.mean(ite_true) - np.mean(ite_pred))
            epoch_metrics["absATT"] = np.abs(np.mean(pred_att - true_att))
            
            epoch_metrics["loss"] = float(np.mean(losses))            
            print_str = f"{split} epoch: {epoch_count}\t "
            for k, v in epoch_metrics.items():
                print_str += f"{k}: {v:.5f} "
        
            print(print_str)
        else:
            epoch_metrics["loss"] = float(np.mean(losses))

        return epoch_metrics  