import sys
sys.path.append("../../")

from config.config import cfg
from transformer_mnist_cfr import MNIST_Transformer
from lstm_mnist_cfr import MNIST_LSTM
from dataset.loaders import get_mnist_cfr_loaders, get_mnist_dataset
from utils.trainer import TrainerMNIST
import time
import argparse
import optuna
import torch
import numpy as np

from sklearn.model_selection import KFold

# Initial script
def run_experiment(cfg, model_name):
    train_dataloader, val_dataloader, test_dataloader = get_mnist_cfr_loaders(cfg, batch_size = cfg.TRAIN.BATCH_SIZE)
    if model_name == "lstm":
        model = MNIST_LSTM(cfg, vocab_size = cfg.MODEL.VOCAB_SIZE)
    elif model_name == "trans":
        model = MNIST_Transformer(cfg, vocab_size = cfg.MODEL.VOCAB_SIZE)
    trainer = TrainerMNIST(cfg, model, train_dataloader, val_dataloader, test_dataloader, cfg.MODEL.NAME)
    trainer.fit()
    trainer.predict()

def run_inference(cfg, model_name):

    # Load the best model
    if model_name == "lstm":
        
        model = MNIST_LSTM(cfg, vocab_size = cfg.MODEL.VOCAB_SIZE)
    elif model_name == "trans":
        
        model = MNIST_Transformer(cfg, vocab_size = cfg.MODEL.VOCAB_SIZE)
    
    pehe_arr, ate_arr = [], []
    
    # Define the K-fold Cross Validator
    k_folds = 20
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    # Start print
    print('--------------------------------')

    dataset = get_mnist_dataset(cfg, cfg.TRAIN.BATCH_SIZE)
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        val_split = int(np.floor(len(train_ids) * 0.15))
        val_ids = train_ids[:val_split]
        train_ids = train_ids[val_split:]

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_dataloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=cfg.TRAIN.BATCH_SIZE, sampler=train_subsampler)
        val_dataloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=cfg.TRAIN.BATCH_SIZE, sampler=val_subsampler)
        test_dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=cfg.TRAIN.BATCH_SIZE, sampler=test_subsampler)

        
        trainer = TrainerMNIST(cfg, model, train_dataloader, val_dataloader, test_dataloader, cfg.MODEL.NAME)    
        trainer.fit()
        metrics = trainer.predict()
        pehe_arr.append(metrics["pehe"])
        ate_arr.append(metrics["absATE"])
        print("Fold {}. Metrics: {}".format(fold, metrics))
        
    pehe_arr, ate_arr = np.array(pehe_arr), np.array(ate_arr)
    print("PEHE mean and std:", np.mean(pehe_arr), np.std(pehe_arr))
    print("ATE mean and std:", np.mean(ate_arr), np.std(ate_arr))

def tune_experiment(cfg, model_name):

    def objective(trial):

        if model_name == "lstm":
            
            lr = trial.suggest_loguniform("lr", 1e-6, 1e-4)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 64, 256)
            embedding_dim = trial.suggest_int('embedding_dim', 64, 256)
            
            cfg.TRAIN.LR = lr
            cfg.MODEL.DROPOUT_P = dropout_rate
            cfg.MODEL.LSTM_NUM_LAYER = num_layers
            cfg.TRAIN.BATCH_SIZE = batch_size
            cfg.MODEL.LSTM_HIDDEN_SIZE = lstm_hidden_size
            cfg.MODEL.EMBEDDING_DIM = embedding_dim
            
            train_dataloader, val_dataloader, test_dataloader = get_mnist_cfr_loaders(cfg, 
                                                                                      batch_size = cfg.TRAIN.BATCH_SIZE)
            
            model = MNIST_LSTM(cfg, vocab_size = cfg.MODEL.VOCAB_SIZE)
        elif model_name == "trans":

            lr = trial.suggest_loguniform("lr", 1e-6, 1e-3)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            att_head = trial.suggest_categorical("attention_head", [4, 8, 16])
            embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
            
            cfg.TRAIN.LR = lr
            cfg.MODEL.DROPOUT_P = dropout_rate
            cfg.MODEL.TRANS_DEPTH = num_layers
            cfg.TRAIN.BATCH_SIZE = batch_size
            cfg.MODEL.EMBEDDING_DIM = embedding_dim
            
            train_dataloader, val_dataloader, test_dataloader = get_mnist_cfr_loaders(cfg, 
                                                                                      batch_size = cfg.TRAIN.BATCH_SIZE)
            model = MNIST_Transformer(cfg, vocab_size = cfg.MODEL.VOCAB_SIZE)
            
        # Run trainer
        trainer = TrainerMNIST(cfg, model, train_dataloader, val_dataloader, test_dataloader, cfg.MODEL.NAME)
        trainer.fit()
        test_metrics = trainer.predict()
        return test_metrics["loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials= 20)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE
    ]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def main():
    
    starttime = time.time()
   
    parser = argparse.ArgumentParser(description='Running experiments on MIMIC dataset.')
    parser.add_argument('name',
                       help='specify a model name')
    parser.add_argument('--tune',
                       help='runs tune experiment',
                       action="store_true")
    parser.add_argument('--inference',
                       help='runs inference with the best model',
                       action="store_true")

    # Execute parse_args()
    args = parser.parse_args()
    
    exp_dir = "../experiments/"
    if args.name == "lstm":
        exp_name = "exp1_mnist_config.yaml"
    else:
        exp_name = "exp0_mnist_config.yaml"
    cfg.merge_from_file(exp_dir + exp_name)
    
    if args.tune:
        print(cfg)
        tune_experiment(cfg, args.name)
    elif args.inference:
        cfg.freeze()
        print(cfg)
        run_inference(cfg, args.name)
    else:
        cfg.freeze()
        print(cfg)
        run_experiment(cfg, args.name)

    print(f"Done in {(time.time() - starttime)/60} minutes.")
    
if __name__ == "__main__":
    main()