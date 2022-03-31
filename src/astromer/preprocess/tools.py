import pandas as pd
import numpy as np
import os

from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed
np.seterr(divide='ignore', invalid='ignore')

def var_index(magnitudes, times):
    '''
    Variability index (Kim et al. 2014)
    Section 2.14 from FATS paper
    https://arxiv.org/pdf/1506.00010.pdf
    '''
    magnitudes = magnitudes.values
    times = times.values
    with np.errstate(divide='ignore'):
        N = len(magnitudes)
        diff_magns = np.diff(magnitudes)
        w = 1./np.power(np.diff(times), 2)
        mean_w = np.mean(w)
        sigma = np.var(magnitudes)
        zero_day = times[-1] - times[0]
        numerator = np.sum(w*np.power(diff_magns, 2))
        denominator = np.power(sigma, 2)*np.sum(w)*N**2
        return (mean_w*np.power(zero_day, 2))*(numerator/denominator)

def eta_e(magnitude, time):
    '''
    Variability index (Kim et al. 2014)
    Section 2.14 from FATS author's implementation
    '''

    w = 1.0 / np.power(np.subtract(time[1:], time[:-1]), 2)
    w_mean = np.mean(w)

    N = len(time)
    sigma2 = np.var(magnitude)

    S1 = sum(w * (magnitude[1:] - magnitude[:-1]) ** 2)
    S2 = sum(w)

    eta_e = (w_mean * np.power(time[N - 1] -
            time[0], 2) * S1 / (sigma2 * S2 * N ** 2))
    return eta_e

def run(sample_row, lcs_dir='.'):
    lc = pd.read_csv(os.path.join(lcs_dir, sample_row['Path']))
    skew_value = skew(lc.iloc[:, 1])
    kurt_value = kurtosis(lc.iloc[:, 1])
    sdev_value = lc.iloc[:, 1].std()
    # vind_value = var_index(lc.iloc[:, 1], lc.iloc[:, 0])
    etae_value = eta_e(lc.iloc[:, 1].values, lc.iloc[:, 0].values)
    return [sample_row['ID'], sdev_value, etae_value, skew_value, kurt_value]

def get_moments(metadata, n_jobs=1, lcs_dir='./data/raw_data/alcock/LCs'):
    var = Parallel(n_jobs=n_jobs)(delayed(run)(row, lcs_dir) \
                                for k, row in metadata.iterrows())
    df = pd.DataFrame(np.array(var), columns=['ID', 'std', 'eta_e', 'skew', 'kurt'])
    return df
