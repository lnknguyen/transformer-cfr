
from utils import get_project_root
root = get_project_root()


import numpy as np
import pandas as pd
from vocab.vectorizer import MIMICVectorizer
from scipy.special import expit, logit
from scipy.stats import bernoulli
import torch

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

def simulate_outcome(beta0, beta1, gamma, confounder, treatment, setting):
    
    # Simulate potential outcomes
    if setting == 'binary':
        y0 = bernoulli.rvs(expit(beta1*confounder))
        y1 = bernoulli.rvs(expit(beta0 + beta1*confounder))
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

def make_cfr_mimic(beta0, beta1, gamma, setting):
    
    # Load data
    data_dir =  f'../../data/mimic/processed/sequential_mimic.csv'
    data = pd.read_csv(data_dir)

    # Vectorizer
    vectorizer = MIMICVectorizer.from_dataframe(data, "sequential_code")
    
    x_diag, x_seq_length = [], []
    code_len_per_visit = []
    
    # Vectorize everything
    for code in data["sequential_code"]:

        diag, seq_length = vectorizer.vectorize(code)

        # Pad all vectors so that they have the length of the longest sequence
        padded_x_seq_along_diagnoses = pad_x_sequence_along_diagnoses(
                diag, max_diag_length_per_visit
        )
        x_diag.append(padded_x_seq_along_diagnoses)
        x_seq_length.append(seq_length)

    x_diag = nn.utils.rnn.pad_sequence(
            x_diag, batch_first=True
    )
    
    x_seq_length = torch.LongTensor(x_seq_length)

    ages, timedelta_means = [], []
    for x in data["sequential_days_delta"].values:
        timedelta_means.append(np.mean([float(i) for i in x.split(";")]))
    
    x_age = data["age"].to_numpy()
    x_age = torch.FloatTensor(data["age"].to_numpy()).unsqueeze(1)
    
    x_gender = torch.LongTensor(data["gender"].to_numpy())
    
    # Define treatment and confounders
    t = data["treated"]
    z = data['long_los']
    
    y0, y1, yf = simulate_outcome(beta0, beta1, gamma, z, t, setting)
    return x_diag, t, y0, y1, yf

def make_cfr_mnist(beta0, beta1, gamma, setting):

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
