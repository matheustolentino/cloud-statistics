#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
# from scipy.special import gamma
from scipy.stats import gamma
from scipy.optimize import curve_fit 
from scipy.signal import argrelextrema, find_peaks
from typing import List, Union
import os
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from pdb import set_trace
from scipy.stats import gaussian_kde

plt.ion()
plt.close('all')
#**************************************************************************************************
PATH_SIZDIST = '../../../data/cloud_sizdist/'
PATH_FIG     = '../figures/'
PARAM = {}
BOUND = {}
PARAM['cloud12b.csv']  = [3000, 2000, 100, 50., 8., 15., 1., 17., .8]
BOUND['cloud12b.csv']  = (0,[1e4, 1e4, 1e4, 70, 8, 40, 2, 48, 5])
PARAM['cloud15b1.csv'] = [1000, 100, .1, 20., 1., 20., 1, 30., .8]
BOUND['cloud15b1.csv'] = (0,[1e4, 1e4, 1e4, 70, 5, 50, 10, 30, 3])
PARAM['cloud15b2.csv'] = [3000, 2000, 100, 50., 8., 28., 2., 17., .8]
BOUND['cloud15b2.csv'] = (0,[1e7, 1e7, 1e7, 70, 8, 60, 5, 48, 5])
PARAM['cloud15b4.csv'] = [3000, 2000, 100, 50., 8., 28., 2., 17., .8]
BOUND['cloud15b4.csv'] = (0,[1e7, 1e7, 1e7, 70, 8, 60, 5, 48, 5])
PARAM['cloud16b.csv']  = [3000, 2000, 100, 50., 8., 28., 2., 17., .8]
BOUND['cloud16b.csv']  = (0,[1e7, 1e7, 1e7, 70, 8, 60, 5, 48, 5])
PARAM['cloud15b3.csv'] = [3000, 2000, 100, 50., 8., 28., 2., 17., .8]
BOUND['cloud15b3.csv'] = (0,[1e7, 1e7, 1e7, 70, 8, 60, 5, 48, 5])
PARAM['cloud13b2.csv'] = [3000, 2000, 100, 50., 8., 28., 2., 17., .8]
BOUND['cloud13b2.csv'] = (0,[1e7, 1e7, 1e7, 70, 8, 60, 5, 48, 5])
PARAM['cloud14b.csv']  = [3000, 2000, 100, 50., 8., 28., 2., 17., .8]
BOUND['cloud14b.csv']  = (0,[1e7, 1e7, 1e7, 70, 8, 60, 5, 48, 5])
PARAM['cloud13b1.csv'] = [3000, 2000, 100, 50., 8., 28., 2., 17., .8]
BOUND['cloud13b1.csv'] = (0,[1e7, 1e7, 1e7, 70, 8, 60, 5, 48, 5])
PARAM['cloud18b.csv']  = [3000, 2000, 100, 50., 8., 28., 2., 17., .8]
BOUND['cloud18b.csv']  = (0,[1e7, 1e7, 1e7, 70, 8, 60, 5, 48, 5])
PARAM['cloud13b3.csv'] = [3000, 2000, 100, 50., 8., 28., 2., 17., .8]
BOUND['cloud13b3.csv'] = (0,[1e7, 1e7, 1e7, 70, 8, 60, 5, 48, 5])

#**************************************************************************************************
filenames = []
for file in os.listdir(PATH_SIZDIST):
    if file.startswith("cloud"):
        filenames.append(file)
        #print(file)
#**************************************************************************************************
def identify_modes(data: np.ndarray, prominence: float = 0.1, width: float = 1) -> List[int]:
    """
    Identify the modes in a distribution.

    Args:
        data (np.ndarray): The data array.
        prominence (float): The minimum prominence of a peak to be considered a mode.
        width (float): The minimum width of a peak to be considered a mode.

    Returns:
        List[int]: A list of indices corresponding to the positions of the identified modes.
    """
    # Find peaks in the distribution
    peaks, _ = find_peaks(data, prominence=prominence, width=width)

    # Return the indices of the peaks (modes)
    return peaks

def mixture_gamma(x: np.ndarray, *params: float) -> np.ndarray:
    """
    Calculate the probability density function (PDF) of a mixture of gamma distributions.

    Parameters:
        x (np.ndarray): Input values at which to evaluate the PDF.
        params (float): Variable-length argument list containing the parameters.
                        The parameters should be provided in the following order:
                        [weights, shapes, scales]

    Returns:
        np.ndarray: The PDF values at the given input values x.
    """
    num_distributions = len(params) // 3
    weights = params[:num_distributions]
    shapes = params[num_distributions:2*num_distributions]
    scales = params[2*num_distributions:]

    pdf = np.zeros_like(x)
    for w, a, b in zip(weights, shapes, scales):
        pdf += w * gamma.pdf(x, a, scale=b)
    return pdf

def neg_log_likelihood(params):
    weights = params[:3]
    shapes = params[3:6]
    scales = params[6:]
    pdf = mixture_gamma(x_data, weights, shapes, scales)
    log_likelihood = np.sum(np.log(pdf))
    return -log_likelihood

def find_point(dist_new):
    aux_dist = np.diff(dist_new)
    ind_sep_aux = np.where(aux_dist > -0.00001)[0]
    
    return ind_sep_aux[0]

def find_max_min(arr):
    if len(arr) == 0:
        return None, None  # Return None for both max and min if the array is empty

    max_val = min_val = arr[0]  # Initialize max and min with the first element

    for element in arr:
        if element > max_val:
            max_val = element  # Update max if a larger element is found
        elif element < min_val:
            min_val = element  # Update min if a smaller element is found

    return max_val, min_val

def find_valleys(arr):
    valleys = []
    n = len(arr)
    
    for i in range(1, n - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            valleys.append(i)
    
    return valleys

# def find_mountains(arr, epsilon):
#     mountains = []
#     n = len(arr)
    
#     for i in range(1, n - 1):
#         if arr[i] - epsilon > arr[i - 1] and arr[i] - epsilon > arr[i + 1]:
#             mountains.append(i)
    
#     return mountains

def find_mountains(arr):
    mountains = []
    n = len(arr)
    
    for i in range(1, n - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            mountains.append(i)
    
    return mountains

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

#**************************************************************************************************
# Teste mixture of gamma functions
# x = np.linspace(0.5, 15, 25)
# true_params = [.8, .3, 12, 20, .2, .3]
# y = mixture_gamma(x, *true_params)

# f               = interpolate.interp1d(x, y, kind='linear')
# diameter_interp = np.linspace(x[0], x[-1], x.shape[0]*2)
# dist_interp     = f(diameter_interp)

# aux_dist     = np.diff(dist_interp)
# global_min_x = np.argmin(aux_dist)
# xnew         = diameter_interp[global_min_x:]
# ind_sep      = np.where(aux_dist[global_min_x:] > -.001)[0][0]
# xsep         = xnew[ind_sep]
# ynew         = dist_interp[global_min_x:]

# diameter_test = x
# dist_test = y
# x1 = diameter_test[diameter_test<=xsep]
# x2 = diameter_test[diameter_test>xsep]
# y1 = dist_test [diameter_test<=xsep]
# y2 = dist_test [diameter_test>xsep]

# n1 = integrate.simps(y1, x1)
# n2 = 2*integrate.simps(y2, x2)

# m11  = integrate.simps(x1*y1, x1)/n1
# m12  = integrate.simps(x1**2*y1,x1)/n1
# var1 =  m12 - m11**2
# nu1 = m11**2/var1
# bt1 = var1/m11

# m21  = integrate.simps(x2*y2, x2)/n2
# var2 =  m12 - m11**2
# nu2 = m21**2/var2
# bt2 = var2/m21

# param_sinthetic, cov_sinthetic = curve_fit(mixture_gamma, x, y, p0=[.8, .3, 12, 20, .2, .3])
# param_calc, cov_calc = curve_fit(mixture_gamma, x, y, p0=[n1,n2,nu1,nu2,bt1,bt2])
# fig, ax = plt.subplots()
# ax.plot(x, y, 'o')
# ax.plot(diameter_interp, dist_interp, '-m')
# ax.plot(xnew[ind_sep], ynew[ind_sep], marker=">", color="k")
# ax.plot(x, mixture_gamma(x, *param_sinthetic[0:6:2]), '--r',
#          label=r"$\nu_1$=%.1f, $\beta_1$=%.1f"%(nu1/true_params[2], bt1/true_params[4]))
# ax.plot(x, mixture_gamma(x, *param_sinthetic[1:6:2]), '--g', 
#         label=r"$\nu_2$=%.1f, $\beta_2$=%.1f"%(nu2/true_params[3], bt2/true_params[5]))
# # ax.plot(diameter_interp, mixture_gamma(diameter_interp, *param_calc), 'oy')
# ax.set_yscale('log')
# ax.legend()
# plt.show()
#**************************************************************************************************
standard_diameter    = np.loadtxt(PATH_SIZDIST+'diameters.csv', delimiter=',')
surface     = 0.24*10**(-2) # mm ^2 -> cm^2
#fitted_dist = np.zeros((len(filenames), diameter.shape[0]))

file = filenames[1]
# 7 deu problema, maybe 3 modes
data = np.loadtxt(PATH_SIZDIST+file, delimiter=',')
# # for i,file in enumerate(filenames):
# for i in range(data.shape[0]):
for i in range(3):
    dist     = data[i, 8:]
    diameter = standard_diameter
   
    if dist.max() > 5:
        # dist = dist/integrate.simps(dist, diameter)
        # derivative = np.diff(dist)
        # ind_neg = np.where(derivative[:2] < 0)[0]
        # if ind_neg.any() and ind_neg.shape[0] < 2:
        #     print("negative derivative")
        #     dist = dist[ind_neg[0]+1:]
        #     diameter = diameter[ind_neg[0]+1:]
        
        # dist = savgol_filter(dist, 9, 3)
        

        filtered_dist = gaussian_filter1d(dist, 1)
        
        # if n_montains > 1:
        #     dist = savgol_filter(dist, 3, 2)

        # # Create a new list with non-zero values from dist and corresponding values from diameter
        # fig, ax = plt.subplots()
        # ax.plot(diameter[:12], dist[:12], 'ob')
        # ax.grid()
        # plt.show()
        #**************************************************************************************************
        f = interpolate.interp1d(diameter, filtered_dist, kind='cubic')
        diameter_interp = np.arange(diameter[0], diameter[-1], .01)
        # diameter_interp = np.linspace(diameter[0], diameter[-1], diameter.shape[0]*5)
        dist_interp     = f(diameter_interp)

        
        # Calculate the first derivative (approximate) using finite differences
        first_derivative = np.diff(dist_interp)  # This is an approximation
        
        # Find the indices where the first derivative changes from negative to positive
        minima_indices = np.where((np.diff(np.sign(first_derivative)) > 0))[0]

        # Find the indices where the first derivative changes from positive to negative
        maxima_indices = np.where((np.diff(np.sign(first_derivative)) < 0))[0]
        
        threshold = 2
        index_valid_modes = maxima_indices[dist_interp[maxima_indices] > threshold]
        # if index_valid_modes:


        # else:
        #     second_derivative = np.diff(dist_interp,2)
        #     # Find the indices where the sign of the second derivative changes
        #     inflection_indices = np.where(np.diff(np.sign(second_derivative)))[0]

        # # Get the corresponding x and y values for inflection points
        # inflection_points_x = diameter_interp[inflection_indices]
        # inflection_points_y = dist_interp[inflection_indices]

        index_mountains = find_mountains(dist_interp)
        index_valleys   = find_valleys(dist_interp)
        print(diameter_interp[index_valleys])
        
        fig, ax = plt.subplots()
        ax.plot(diameter_interp[:-1], np.diff(dist_interp,1), '--b')

        # ax.plot(x1, y1, '-r')
        ax.grid()
        plt.show()

        # Remover zeros aqui, e nao precisa abaixo
        mask_non_zero = np.where(dist_interp == 0)[0]
        if mask_non_zero.any():
            print("heve zeros")
            dist_new, diameter_new = dist_interp[~mask_non_zero], diameter_interp[~mask_non_zero]
        n0 = integrate.simps(dist_interp, diameter_interp)

        fig, ax = plt.subplots()
        ax.plot(diameter, dist, 'ob')
        ax.plot(diameter, filtered_dist, 'or')
        ax.plot(diameter_interp, dist_interp, '--m')
        ax.grid()
        plt.show()

        #**************************************************************************************************