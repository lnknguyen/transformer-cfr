import sys, os
sys.path.append("../../")

import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch

from dataset.prepare_cfr_mimic import make_cfr_mimic

torch.manual_seed(1)
device = "cpu"
# take over whatever gpus are on the system
if torch.cuda.is_available():
    device = torch.cuda.current_device()

def get_mimic_cfr_loaders(cfg, batch_size, num_workers = 1, device = device):

    '''
    Return context X, treatment T, factual outcome yf and potential outcomes y_0 ... y_n from context with noise
    '''
    x_diag, x_age, gender, t, y0, y1, yf = make_cfr_mimic(cfg.PATH.MIMIC_DATA_DIR,
                                                        cfg.SIM.TREATMENT_STRENGTH,
                                                        cfg.SIM.CONFOUNDER_STRENGTH,
                                                        cfg.SIM.OUTPUT_NOISE,
                                                        cfg.SIM.OUTPUT_TYPE
                                                        )

    print("1s:", sum(t) / len(t))
    print("y0s:", sum(y0) / len(y0))
    print("y1s:", sum(y1) / len(y1))
    
    x_diag = torch.LongTensor(x_diag)
    x_age = torch.LongTensor(x_age)
    gender = torch.LongTensor(gender)
    t = torch.FloatTensor(t)
    yf = torch.FloatTensor(yf)
    y0 = torch.FloatTensor(y0)
    y1 = torch.FloatTensor(y1)

    dataset_size = int(x_diag.shape[0])

    indices = list(range(dataset_size))
    ts_split = int(np.floor(0.2 * dataset_size))
    val_split = int(np.floor((dataset_size - ts_split) * 0.15))
    np.random.shuffle(indices)

    tr_indices, test_indices = indices[ts_split:], indices[:ts_split]
    train_indices, valid_indices = tr_indices[val_split:], tr_indices[:val_split]
    
    train_dataset = TensorDataset(
        x_diag[train_indices],
        x_age[train_indices],
        gender[train_indices],
        t[train_indices],
        yf[train_indices],
        y0[train_indices],
        y1[train_indices]
    )

    val_dataset = TensorDataset(
        x_diag[valid_indices],
        x_age[valid_indices],
        gender[valid_indices],
        t[valid_indices],
        yf[valid_indices],
        y0[valid_indices],
        y1[valid_indices]
    )

    test_dataset = TensorDataset(
        x_diag[test_indices],
        x_age[test_indices],
        gender[test_indices],
        t[test_indices],
        yf[test_indices],
        y0[test_indices],
        y1[test_indices]
    )

    return get_dataloaders_from_datasets(train_dataset, val_dataset, test_dataset, batch_size, num_workers)

def get_dataloaders_from_datasets(train_dataset, val_dataset, test_dataset, batch_size, num_workers):

    train_dataloader = DataLoader(
        train_dataset,
        batch_size= batch_size,
        num_workers= num_workers,
        drop_last=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size= batch_size,
        num_workers= num_workers,
        drop_last=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size= batch_size,
        num_workers= num_workers,
        drop_last=False,
    )

    return train_dataloader, val_dataloader, test_dataloader