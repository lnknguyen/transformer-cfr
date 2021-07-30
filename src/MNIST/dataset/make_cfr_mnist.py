import sys
sys.path.append("../")

import numpy as np
import pandas as pd
from vocab.vectorizer import MIMICVectorizer
from scipy.special import expit, logit
from scipy.stats import bernoulli
import torch
import torch.nn as nn
from ast import literal_eval

def simulate_outcome(beta0, beta1, gamma, confounder, treatment, setting):
    
    no = confounder.shape[0]
    p0 = expit(beta1*confounder)
    p1 = expit(beta0 + beta1*confounder)
    print(no)
    # Simulate potential outcomes
    if setting == 'binary':
        y0 = bernoulli.rvs(expit(beta1*confounder))
        y1 = bernoulli.rvs(expit(beta0 + beta1*confounder))
        print(p0.reshape(no).shape)
        y0 = np.random.binomial(1, p0, (no,))
        y1 = np.random.binomial(1, p1, (no,))
        print(y0.shape)
    elif setting == 'continuous':
        y0 = beta1*confounder + random.normal(0, gamma)
        y1 = beta0 + y0
    else:
        raise Exception("Unrecognized setting")
    
    no = treatment.shape[0]
    # Define factual outcomes
    yf = np.zeros([no,1])
    yf = np.where(treatment == 0, y0, y1)
    yf = np.reshape(yf.T, [no, ])
    
    return y0, y1, yf

def make_cfr_mnist(beta0, beta1, gamma, setting):
    '''
    Creates loaders for MNIST dataset.
    '''
    
    # Load data
    data_dir =  '../../../data/mnist/processed/mnist-cfr.npz'
    data = np.load(data_dir)
    
    xs_1 = data['xs_1']
    xs_2 = data['xs_2']
    xs_3 = data['xs_3']
    t = data['t']
    z = data['confounder']
    
    y0, y1, yf = simulate_outcome(beta0, beta1, gamma, z, t, setting)
    return xs_1, xs_2, xs_3, t, z, y0, y1, yf