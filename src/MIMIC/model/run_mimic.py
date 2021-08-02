import sys
sys.path.append("../../")

from config.config import cfg
from transformer_cfr_mimic import MIMIC_Transformer
from dataset.loaders import get_mimic_cfr_loaders
from utils.trainer import TrainerMIMIC
import time

def run_experiment(cfg):
    train_dataloader, val_dataloader, test_dataloader = get_mimic_cfr_loaders(cfg, 
                                                                              batch_size = cfg.TRAIN.BATCH_SIZE)
    model = MIMIC_Transformer(cfg, vocab_size = cfg.MODEL.DIAG_VOCAB_SIZE)
    trainer = TrainerMIMIC(cfg, model, train_dataloader, val_dataloader, test_dataloader, "trans_mimic")
    trainer.fit()

def main():
    starttime = time.time()
   
    exp_dir = "../experiments/"
    exp_name = "exp0_mimic_config.yaml"
    cfg.merge_from_file(exp_dir + exp_name)
    cfg.freeze()
    print(cfg)
    run_experiment(cfg)
    print(f"Done in {(time.time() - starttime)/60} minutes.")

if __name__ == "__main__":
    main()