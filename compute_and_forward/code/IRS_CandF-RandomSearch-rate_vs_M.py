# This file only implements the 'random search' to be used as a baseline to compare with the proposed AO method.
# Here, the achievable rate vs. M is derived.
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
M = 3 # Dimension of IRS

NumChnlRealization = 2000 # Number of channel realization
NumInitialPoints = 35 # Number of run for each channel realization with different initial points for thetas

a = np.array([1,-1]).reshape((2,1))
M_list = range(0,51)

delta = 0.001 # For gradient computation
step_size = 0.05 # The step size in the gradient descent algorithm.
max_ao_itr=500



def realize_channels(K, M, mu, sigma2):
    # Realize the channel transfer vectors and matrices
    h = np.random.normal(mu, sigma2, (K, 1)) + 1j * np.random.normal(mu, sigma2, (K, 1))
    h_s = np.random.normal(mu, sigma2, (M, 1)) + 1j * np.random.normal(mu, sigma2, (M, 1))
    G = np.random.normal(mu, sigma2, (K,M)) + 1j * np.random.normal(mu, sigma2, (K,M)) # is a K x M complex matrix

    return h, h_s, G



if __name__ == "__main__":
    # Create a number of workers for parallel processing
    pool_size = cpu_count()
#    pool_size = 37
    pool = Pool(processes=pool_size)

#    AO_achv_rates = np.zeros( (len(M_list), NumChnlRealization, NumInitialPoints) )
    init_achv_rate = np.zeros( (len(M_list), NumChnlRealization, NumInitialPoints) )

    for M_itr, M in enumerate(M_list):
        print("M =", M)
        for ChnlRl_itr in range(NumChnlRealization):
            h, h_s, G = realize_channels(K, M, mu, sigma2)
#            AO_params = []
            for init_itr in range(NumInitialPoints):
#                print("M=", M, "Channel Realization Itr=", ChnlRl_itr)
                theta_init = np.random.uniform(0, 2 * pi, M)

#                print(theta_init.shape, h.shape, G.shape, h_s.shape, a.shape)
                init_achv_rate[M_itr, ChnlRl_itr, init_itr] = \
                    IRS_C_and_F_rate(theta_init, h, G, h_s, a, SNR)

 #               AO_params.append( (theta_init, h, G, h_s, a, step_size, delta, SNR, max_ao_itr) )

#                AO_rate, AO_rate_vec, AO_theta = AO_on_Theta_and_alpha()
#            AO_rslts = pool.map(AO_on_Theta_and_alpha, AO_params)
            for i in range(NumInitialPoints):
#                AO_rate = rslt['rate']
#                AO_achv_rates[M_itr, ChnlRl_itr, i] = AO_rate
                print("M =", M, ", Initial achieved rate =", init_achv_rate[M_itr, ChnlRl_itr, i])

            print("----------")

    sio.savemat('CandF_rslt-Random_Search-K={}_SNR={}_M=0-50.mat'.format(K, SNRdb), {'K': K, 'M_list': M_list, 'a': a,\
                                                                'SNRdb': SNRdb, 'init_achv_rate': init_achv_rate})
