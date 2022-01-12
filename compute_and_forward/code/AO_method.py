#--------------------------------------------------------------------
# The following functions implement the alternating optimization
#   method to optimize thetas of an IRS array to maximize compute and
#   forward rate.
# To run the AO method one has to call "AO_on_Theta_and_alpha" function
#--------------------------------------------------------------------

import numpy as np
from numpy import exp as exp
import copy
from math import log, pi
from required_funcs import IRS_C_and_F_rate

# Find the gradient of |alpha*h_eff-a|^2 at theta
def err_gradient(theta, h, G, h_s, alpha, a, delta):
    # theta is a numpy array
    M = theta.shape[0]
    grad = np.zeros(M)
    for i in range(M):
        theta_plus = copy.deepcopy(theta)
        theta_plus[i] = theta_plus[i] + delta
        Theta_plus = np.diagflat(exp(1j*np.array(theta_plus)))
        h_eff = h + G @ Theta_plus @ h_s #np.matmul(np.matmul(G, Theta_plus), h_s)
#        print(type(alpha), type(h_eff))
#        tttmp = alpha*h_eff
#        print(h_eff.shape, a.shape, alpha.shape)
        err_plus = np.linalg.norm(alpha*h_eff-a) ** 2

        theta_minus = copy.deepcopy(theta)
        theta_minus[i] = theta_minus[i] - delta
        Theta_minus = np.diagflat(exp(1j*np.array(theta_minus)))
        h_eff = h + G @ Theta_minus @ h_s #h + np.matmul(np.matmul(G, Theta_minus), h_s)
        err_minus = np.linalg.norm(alpha*h_eff-a) ** 2

        grad[i] = (err_plus-err_minus)/(float(2)*delta)

    return grad



#--------------------------------------------------------------------
# Gradient descent to optimize theta
def grad_descent_on_theta(init_theta, h, G, h_s, alpha, a, step_size, delta, max_itr):

    theta = copy.deepcopy(init_theta)
    for i in range(max_itr):
        grad = err_gradient(theta, h, G, h_s, alpha, a, delta)
        theta -= (step_size * grad)

    return theta



#--------------------------------------------------------------------
# Alternating optimization
def AO_on_Theta_and_alpha(params):

    theta_init, h, G, h_s, a, step_size, delta, SNR, max_ao_itr = params

    rate_vec = []
    theta = copy.deepcopy(theta_init)

    Theta_init = np.diagflat(exp(1j*np.array(theta)))
#    h_eff = h + np.matmul(np.matmul(G, Theta_init), h_s)
    h_eff = h + G @ Theta_init @ h_s
    alpha = np.array(SNR * np.matmul(np.matrix(h_eff).getH(), a) / (1 + SNR * np.linalg.norm(h_eff) ** 2))

    # Alternating optimization between alpha and Theta
    for i in range(max_ao_itr):
#        print("AO iteration:", i)
        theta = grad_descent_on_theta(theta, h, G, h_s, alpha, a, step_size, delta, 25)

        Theta = np.diagflat(exp(1j*np.array(theta)))
        h_eff = h + G @ Theta @ h_s #h + np.matmul(np.matmul(G, Theta), h_s)
        alpha = np.array(SNR * np.matmul(np.matrix(h_eff).getH(), a) / (1 + SNR * np.linalg.norm(h_eff) ** 2))

        rate = IRS_C_and_F_rate(theta, h, G, h_s, a, SNR)
        rate_vec.append(rate)

#    print("AO Rate=", rate)

    return {'rate': rate, 'rate_vec': rate_vec, 'theta': theta}
