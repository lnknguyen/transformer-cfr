import sys, os
sys.path.append("../")

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import log_loss
import random

from dataset.make_cfr_mnist import make_cfr_mnist

bias = 50.0
noise = 1.0
print("Baseline training: bias {} noise {}".format(bias, noise))
xs_1, xs_2, xs_3, t, y0, y1, yf = make_cfr_mnist(beta0 = 1.0,
                                                 beta1 = bias,
                                                 gamma = noise,
                                                 setting = "continuous")

X = np.hstack((xs_1, xs_2, xs_3))

def shuffle_data(X, t, yf = None, y0 = None, y1 = None):
    dataset_size = X.shape[0]
    indices = list(range(dataset_size))
    ts_split = int(np.floor(0.2 * dataset_size))
    val_split = int(np.floor((dataset_size - ts_split) * 0.15))
    np.random.shuffle(indices)

    tr_indices, test_indices = indices[ts_split:], indices[:ts_split]
    train_indices, valid_indices = tr_indices[val_split:], tr_indices[:val_split]
    
    X_train, yf_train, y0_train, y1_train, t_train = X[train_indices], yf[train_indices], y0[train_indices], y1[train_indices], t[train_indices]
    X_val, yf_val, y0_val, y1_val, t_val = X[valid_indices], yf[valid_indices], y0[valid_indices], y1[valid_indices], t[valid_indices]
    X_test, yf_test, y0_test, y1_test, t_test = X[test_indices], yf[test_indices], y0[test_indices], y1[test_indices], t[test_indices]
    return X_train, yf_train, y0_train, y1_train, t_train, X_test, yf_test, y0_test, y1_test, t_test

# OLS
factual_mae_arr, pehe_arr, absATE_arr, ite_arr, factual_ce_arr = [], [], [], [], []
iters = 20

for i in range(iters):
    (unique, counts) = np.unique(t, return_counts=True)
    frequencies = np.asarray((unique, counts)).T

    X_train, yf_train, y0_train, y1_train, t_train, X_test, yf_test, y0_test, y1_test, t_test = shuffle_data(X, t, yf, y0, y1)
    control_idx = list(np.where(t_train == 0)[0])
    treated_idx = list(np.where(t_train == 1)[0])

    print(y0)
    print(y1)
    dt = t_test
    dx = X_test
    d0y = y0_test
    d1y = y1_test

    clf = Lasso(max_iter = 5000, tol = 1e-4)
    ols0 = clf.fit(X_train[control_idx], yf_train[control_idx])
    y0_preds = ols0.predict(dx)

    ols1 = clf.fit(X_train[treated_idx], yf_train[treated_idx])
    y1_preds = ols1.predict(dx)

    yf_true = np.where(
        dt == 0,
        d0y,
        d1y
    )

    yf_pred = np.where(
        dt == 0,
        y0_preds,
        y1_preds
    )

    ycf_true = np.where(
        dt == 0,
        d1y,
        d0y
    )
    ycf_pred = np.where(
        dt == 0,
        y1_preds,
        y0_preds
    )

    ite_true = d1y - d0y
    ite_pred = y1_preds - y0_preds
    factual_mae = mae(yf_pred, yf_true)
    ite = ite_true - ite_pred
    pehe = np.sqrt(np.mean(np.square((ite_true) - (ite_pred))))
    absATE = np.abs(np.mean(ite_true) - np.mean(ite_pred))

    factual_mae_arr.append(factual_mae)
    pehe_arr.append(pehe)
    absATE_arr.append(absATE)
    ite_arr.append(ite)

#factual_ce = log_loss(yf_true, yf_pred)
#factual_ce_arr.append(factual_ce) 

print("Factual mae")
print(np.mean(np.array(factual_mae_arr)))
print(np.std(np.array(factual_mae_arr)))
print("ITE")
print(np.mean(np.array(ite_arr)))
print(np.std(np.array(ite_arr)))
print("PEHE")
print(np.mean(np.array(pehe_arr))) 
print(np.std(np.array(pehe_arr)))
print("Abs ate")
print(np.mean(np.array(absATE_arr)))
print(np.std(np.array(absATE_arr)))
#print("Factual ce")
#print(np.mean(np.array(factual_ce_arr)))
#print(np.std(np.array(factual_ce_arr)))
