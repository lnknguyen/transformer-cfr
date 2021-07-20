import sys, os
sys.path.append("../../")

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
    xs_1, xs_2, xs_3, t, z, y0, y1, yf = make_cfr_mimic(0,0,0,'binary')

    # MNIST
    xs_1 = torch.LongTensor(xs_1)
    xs_2 = torch.LongTensor(xs_2)
    xs_3 = torch.LongTensor(xs_3)
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
        xs_1[train_indices],
        xs_2[train_indices],
        xs_3[train_indices],
        t[train_indices],
        z[train_indices],
        yf[train_indices],
        y0[train_indices],
        y1[train_indices]
    )

    val_dataset = TensorDataset(
        xs_1[valid_indices],
        xs_2[valid_indices],
        xs_3[valid_indices],
        t[valid_indices],
        z[valid_indices],
        yf[valid_indices],
        y0[valid_indices],
        y1[valid_indices]
    )

    test_dataset = TensorDataset(
        xs_1[test_indices],
        xs_2[test_indices],
        xs_3[test_indices],
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