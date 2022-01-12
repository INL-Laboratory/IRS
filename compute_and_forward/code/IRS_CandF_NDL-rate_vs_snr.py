# This file implements 'alternating optimization' approaches.
# Here, the achievable rate vs. SNR is derived.
# But we assume there is no direct link between the users and the base station.
# We will use MATLAB to find the maximum over 'Initial Points' and then find the average
#   over the 'Channel Realizations.'

import numpy as np
from multiprocessing import Pool, cpu_count
from math import log, pi
#import matplotlib.pyplot as plt
from AO_method import AO_on_Theta_and_alpha
from required_funcs import IRS_C_and_F_rate
import scipy.io as sio

# Initial parameteres
mu = 0
sigma2 = 1

SNRdb = 5 # SNR in db
SNR = 10 ** (SNRdb/float(10))

K = 2 # Number of transmitters
M = 20 # Dimension of IRS

NumChnlRealization = 1 # Number of channel realization
NumInitialPoints = 1 # Number of run for each channel realization with different initial points for thetas

a = np.array([1,1]).reshape((2,1))
SNRdb_list = [0, 5, 10, 15, 20, 25, 30]
SNR_list = [10 ** (snr/float(10)) for snr in SNRdb_list]

delta = 0.001 # For gradient computation
step_size = 0.05 # The step size in the gradient descent algorithm.
max_ao_itr=500



def realize_channels(K, M, mu, sigma2, direct_link=True):
    # Realize the channel transfer vectors and matrices
    if direct_link:
        h = np.random.normal(mu, sigma2, (K, 1)) + 1j * np.random.normal(mu, sigma2, (K, 1))
    else:
        h = np.zeros((K, 1))
    h_s = np.random.normal(mu, sigma2, (M, 1)) + 1j * np.random.normal(mu, sigma2, (M, 1))
    G = np.random.normal(mu, sigma2, (K,M)) + 1j * np.random.normal(mu, sigma2, (K,M)) # is a K x M complex matrix

    return h, h_s, G



if __name__ == "__main__":
    # Create a number of workers for parallel processing
    pool_size = cpu_count()
#    pool_size = 37
    pool = Pool(processes=pool_size)

    AO_achv_rates = np.zeros( (len(SNR_list), NumChnlRealization, NumInitialPoints) )
    init_achv_rate = np.zeros( (len(SNR_list), NumChnlRealization, NumInitialPoints) )

    for snr_itr, snr in enumerate(SNR_list):
        print("SNR =", snr)
        for ChnlRl_itr in range(NumChnlRealization):
            h, h_s, G = realize_channels(K, M, mu, sigma2, False) # No direct link between users and the BS
            AO_params = []
            for init_itr in range(NumInitialPoints):
#                print("M=", M, "Channel Realization Itr=", ChnlRl_itr)
                theta_init = np.random.uniform(0, 2 * pi, M)

#                print(theta_init.shape, h.shape, G.shape, h_s.shape, a.shape)
#                init_achv_rate[M_itr, ChnlRl_itr, init_itr] = \
#                    IRS_C_and_F_rate(theta_init, h, G, h_s, a, SNR)

                AO_params.append( (theta_init, h, G, h_s, a, step_size, delta, snr, max_ao_itr) )

#                AO_rate, AO_rate_vec, AO_theta = AO_on_Theta_and_alpha()
            AO_rslts = pool.map(AO_on_Theta_and_alpha, AO_params)
            for i, rslt in enumerate(AO_rslts):
                AO_rate = rslt['rate']
                AO_achv_rates[snr_itr, ChnlRl_itr, i] = AO_rate
                print("SNR =", snr, ", AO achieved rate =", AO_rate)

            print("----------")

    sio.savemat('NDL_CandF_rslt-K={}_M={}_SNR=0_30.mat'.format(K, M), {'K': K, 'M': M, 'SNR_list': SNR_list, 'a': a,\
                                                                    'SNRdb_list': SNRdb_list,\
                                                                    'AO_achv_rates': AO_achv_rates})
