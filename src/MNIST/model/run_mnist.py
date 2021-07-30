import sys
sys.path.append("../../")

from config.config import cfg
from transformer_mnist_cfr import MNIST_Transformer
from loaders import get_mnist_cfr_loaders
from trainer import TrainerMNIST
import time

# Initial script
def run_experiment(cfg):
    train_dataloader, val_dataloader, test_dataloader = get_mnist_cfr_loaders(cfg, batch_size = cfg.TRAIN.BATCH_SIZE)
    model = MNIST_Transformer(cfg, vocab_size = cfg.MODEL.VOCAB_SIZE)
    trainer = TrainerMNIST(cfg, model, train_dataloader, val_dataloader, test_dataloader, cfg.MODEL.NAME)
    trainer.fit()

def main():
    starttime = time.time()
   
    exp_dir = "../../config/experiments/"
    exp_name = "exp0_mnist_config.yaml"
    cfg.merge_from_file(exp_dir + exp_name)
    cfg.freeze()
    print(cfg)
    run_experiment(cfg)
    print(f"Done in {(time.time() - starttime)/60} minutes.")

if __name__ == "__main__":
    main()