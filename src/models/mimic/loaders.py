import sys, os
sys.path.append("../")

import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch

from data.prepare_cfr_dataset import make_cfr_mimic

torch.manual_seed(1)
device = "cpu"
# take over whatever gpus are on the system
if torch.cuda.is_available():
    device = torch.cuda.current_device()


def get_mimic_cfr_loaders(batch_size = 64, num_workers = 1, device = device):

    '''
    Return context X, treatment T, factual outcome yf and potential outcomes y_0 ... y_n from context with noise
    '''
    x_diag, x_age, x_gender, z, t, y0, y1, yf = make_cfr_mimic(0,0,0,'binary')

    # MNIST
    x_diag = torch.LongTensor(x_diag)
    x_age = torch.FloatTensor(x_age)
    x_gender = torch.LongTensor(x_gender)
    t = torch.FloatTensor(t)
    z = torch.FloatTensor(z)
    yf = torch.FloatTensor(yf)
    y0 = torch.FloatTensor(y0)
    y1 = torch.FloatTensor(y1)

    dataset_size = int(xs_1.shape[0])

    indices = list(range(dataset_size))
    ts_split = int(np.floor(0.2 * dataset_size))
    val_split = int(np.floor((dataset_size - ts_split) * 0.15))
    np.random.shuffle(indices)

    tr_indices, test_indices = indices[ts_split:], indices[:ts_split]
    train_indices, valid_indices = tr_indices[val_split:], tr_indices[:val_split]
    
    train_dataset = TensorDataset(
        x_diag[train_indices],
        x_age[train_indices],
        x_gender[train_indices],
        t[train_indices],
        z[train_indices],
        yf[train_indices],
        y0[train_indices],
        y1[train_indices]
    )

    val_dataset = TensorDataset(
        x_diag[valid_indices],
        x_age[valid_indices],
        x_gender[valid_indices],
        t[valid_indices],
        z[valid_indices],
        yf[valid_indices],
        y0[valid_indices],
        y1[valid_indices]
    )

    test_dataset = TensorDataset(
        x_diag[test_indices],
        x_age[test_indices],
        x_gender[test_indices],
        t[test_indices],
        z[test_indices],
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