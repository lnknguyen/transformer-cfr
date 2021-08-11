import sys
sys.path.append("../../")

from config.config import cfg
from transformer_cfr_mimic import MIMIC_Transformer
from lstm_mimic_cfr import MIMIC_LSTM
from dataset.loaders import get_mimic_cfr_loaders
from utils.trainer import TrainerMIMIC
import time
import argparse
import optuna

def run_experiment(cfg, model_name):
    train_dataloader, val_dataloader, test_dataloader = get_mimic_cfr_loaders(cfg, 
                                                                              batch_size = cfg.TRAIN.BATCH_SIZE)
    if model_name == "lstm":
        model = MIMIC_LSTM(cfg, vocab_size = cfg.MODEL.VOCAB_SIZE)
    elif model_name == "trans":
        model = MIMIC_Transformer(cfg, vocab_size = cfg.MODEL.VOCAB_SIZE)
    
    trainer = TrainerMIMIC(cfg, model, train_dataloader, val_dataloader, test_dataloader, cfg.MODEL.NAME)
    trainer.fit()
    trainer.predict()

def tune_experiment(cfg, model_name):

    def objective(trial):

        if model_name == "lstm":
            
            lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)
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
            
            train_dataloader, val_dataloader, test_dataloader = get_mimic_cfr_loaders(cfg, 
                                                                                      batch_size = cfg.TRAIN.BATCH_SIZE)
            
            model = MIMIC_LSTM(cfg, vocab_size = cfg.MODEL.VOCAB_SIZE)
        elif model_name == "trans":

            lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            att_head = trial.suggest_categorical("attention_head", [4, 8, 16])
            embedding_dim = trial.suggest_categorical('embedding_dim', [62, 126, 254])
            
            cfg.TRAIN.LR = lr
            cfg.MODEL.DROPOUT_P = dropout_rate
            cfg.MODEL.TRANS_DEPTH = num_layers
            cfg.TRAIN.BATCH_SIZE = batch_size
            cfg.MODEL.EMBEDDING_DIM = embedding_dim
            
            train_dataloader, val_dataloader, test_dataloader = get_mimic_cfr_loaders(cfg, 
                                                                                      batch_size = cfg.TRAIN.BATCH_SIZE)
            model = MIMIC_Transformer(cfg, vocab_size = cfg.MODEL.VOCAB_SIZE)
            
        # Run trainer
        trainer = TrainerMIMIC(cfg, model, train_dataloader, val_dataloader, test_dataloader, cfg.MODEL.NAME)
        trainer.fit()
        test_metrics = trainer.predict()
        return test_metrics["loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials= 50)

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

    # Execute parse_args()
    args = parser.parse_args()
    
    exp_dir = "../experiments/"
    if args.name == "lstm":
        exp_name = "exp1_mimic_config.yaml"
    else:
        exp_name = "exp0_mimic_config.yaml"
    cfg.merge_from_file(exp_dir + exp_name)
    
    if args.tune:
        tune_experiment(cfg, args.name)
    else:
        cfg.freeze()
        print(cfg)
        run_experiment(cfg, args.name)

    print(f"Done in {(time.time() - starttime)/60} minutes.")

if __name__ == "__main__":
    main()