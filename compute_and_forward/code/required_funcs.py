import numpy as np
from numpy import exp as exp
from math import log



def IRS_C_and_F_rate(theta, h, G, h_s, a, SNR):
    K = h.shape[0]
    Theta = np.diagflat(exp(1j*np.array(theta)))

    h_eff = h + G @ Theta @ h_s #np.matmul(np.matmul(G, Theta), h_s)

    tmp_mat = np.identity(K) - (SNR / (1 + SNR * np.linalg.norm(h_eff) ** 2)) \
                                              * np.matmul(np.matrix(h_eff), np.matrix(h_eff).getH())
    rate = log(1 / np.matmul(np.matmul(np.matrix(a).getH(), tmp_mat), a).real, 2)

    return rate
