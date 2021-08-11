import sys
sys.path.append("../../")

from config.config import cfg
from transformer_mnist_cfr import MNIST_Transformer
from lstm_mnist_cfr import MNIST_LSTM
from dataset.loaders import get_mnist_cfr_loaders
from utils.trainer import TrainerMNIST
import time
import argparse

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
    
def main():
    
    parser = argparse.ArgumentParser(description='Running experiments on MNIST dataset.')
    parser.add_argument('name',
                       help='specify a model name')
    # Execute parse_args()
    args = parser.parse_args()

    starttime = time.time()
   
    exp_dir = "../experiments/"
    if args.name == "lstm":
        exp_name = "exp1_mnist_config.yaml"
    else:
        exp_name = "exp0_mnist_config.yaml"
    cfg.merge_from_file(exp_dir + exp_name)
    cfg.freeze()
    print(cfg)
    run_experiment(cfg, args.name)
    print(f"Done in {(time.time() - starttime)/60} minutes.")

if __name__ == "__main__":
    main()