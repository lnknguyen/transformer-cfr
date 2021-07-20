import sys
sys.path.append("../../")

from config.config import cfg
from transformer_mnist_cfr import MNIST_Transformer
from loaders import get_mnist_cfr_loaders
from trainer import TrainerMNIST
import time

# Initial script
def run_experiment():
    train_dataloader, val_dataloader, test_dataloader = get_mnist_cfr_loaders(batch_size = cfg.TRAIN.BATCH_SIZE)
    model = MNIST_Transformer(cfg, vocab_size = 258)
    trainer = TrainerMNIST(cfg, model, train_dataloader, val_dataloader, test_dataloader, "trans_mnist")
    trainer.fit()

def main():
    starttime = time.time()
   
    run_experiment()
    print(f"Done in {(time.time() - starttime)/60} minutes.")

if __name__ == "__main__":
    main()