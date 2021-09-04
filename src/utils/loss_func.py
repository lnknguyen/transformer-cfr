import numpy as np
from  scipy.stats import wasserstein_distance

def lindisc(X,p,t):
    ''' Linear MMD '''

    control_idx = list(np.where(t == 0)[0])
    treated_idx = list(np.where(t == 1)[0])

    Xc = X[control_idx]
    Xt = X[treated_idx]

    mean_control = np.mean(Xc)
    mean_treated = np.mean(Xt)
    
    c = np.square(2*p-1)*0.25
    f = np.sign(p-0.5)

    mmd = np.sum(np.square(p*mean_treated - (1-p)*mean_control))
    mmd = f*(p-0.5) + np.sqrt(c + mmd)

    return mmd

def wass(X, t):

    control_idx = list(np.where(t == 0)[0])
    treated_idx = list(np.where(t == 1)[0])

    Xc = X[control_idx]
    Xt = X[treated_idx]

    mean_control = np.mean(Xc)
    mean_treated = np.mean(Xt)

    return wasserstein_distance(mean_control, mean_treated)