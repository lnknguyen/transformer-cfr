import sys
sys.path.append("../../")

from config.config import cfg
from transformer_cfr_mimic import MIMIC_Transformer
from loaders import get_mimic_cfr_loaders
from trainer import TrainerMIMIC
import time

# Initial script
def run_experiment():
    train_dataloader, val_dataloader, test_dataloader = get_mimic_cfr_loaders(batch_size = cfg.TRAIN.BATCH_SIZE)
    model = MIMIC_Transformer(cfg, vocab_size = 258)
    trainer = TrainerMIMIC(cfg, model, train_dataloader, val_dataloader, test_dataloader, "trans_mimic")
    trainer.fit()

def main():
    starttime = time.time()
   
    run_experiment()
    print(f"Done in {(time.time() - starttime)/60} minutes.")

if __name__ == "__main__":
    main()