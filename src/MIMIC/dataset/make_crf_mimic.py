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

def pad_x_sequence_along_diagnoses(seq: list, max_diagnosis_length: int, padding_value: int = 0
    ) -> torch.LongTensor:
    """
    Args: 
      seq: list of lists [num_visits, max_diag_length_per_visit]
      max_seq_length: max length of sequence across the batches
      ax_diagnosis_length: max_diag_ln_across_batches
    Returns:
      padded_vector: torch Tensor [num_visits, max_diagnosis_length] padded by 0
    """
    padded_vector_along_diagnoses = np.zeros((len(seq), max_diagnosis_length))
    for i, visit in enumerate(seq):
        padded_vector_along_diagnoses[
            i, : len(visit)
        ] = visit  # pads zeros to the front

    return torch.LongTensor(padded_vector_along_diagnoses)

def simulate_outcome(beta0, beta1, gamma, confounder, treatment, outcome, setting):
    """
    Args: 
      beta0: treatment strength
      beta1: confounder strength
      gamma: noise level
      outcome: factual outcome. For binary setting, this is the mortality flag
    Returns:
      y0, y1, yf
    """
    
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

def make_cfr_mimic(data_dir, beta0, beta1, gamma, setting, max_diag_length_per_visit = 66):
    '''
    Creates loaders for MIMIC dataset.
    '''
    
    # Load data
    data_dir =  data_dir + 'sequential_mimic.csv'
    data = pd.read_csv(data_dir)
    
    # Define treatment and confounders
    age = data["admission_age"].apply(literal_eval)
    los = data["los"].apply(literal_eval)
    treatment = data["treated"]
    gender = data["gender"]
    
    # Vectorizer
    vectorizer = MIMICVectorizer.from_dataframe(data, "sequential_code")
    
    x_diag, x_age = [], []
    code_len_per_visit = []
    
    # Vectorize everything
    for code in data["sequential_code"]:

        diag, seq_length = vectorizer.vectorize(code)

        # Pad all vectors so that they have the length of the longest sequence
        padded_x_seq_along_diagnoses = pad_x_sequence_along_diagnoses(
                diag, max_diag_length_per_visit
        )
        x_diag.append(padded_x_seq_along_diagnoses)

        padded_x_age_along_diagnoses = pad_x_sequence_along_diagnoses(
                age, max_diag_length_per_visit
        )
        x_age.append(padded_x_age_along_diagnoses)

    x_diag = nn.utils.rnn.pad_sequence(
            x_diag, batch_first=True
    )
    x_age = nn.utils.rnn.pad_sequence(
            x_age, batch_first=True
    )    
    
    x_seq_length = torch.LongTensor(x_seq_length)
    
    #  set average age as confounder
    avg_age =  np.array([np.mean(sublist) for sublist in np.array(age)])
    y0, y1, yf = simulate_outcome(beta0, beta1, gamma, avg_age, treatment, setting)
    
    return x_diag, x_age, gender, treatment, y0, y1, yf