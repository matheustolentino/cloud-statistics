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
from pdb import set_trace

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

def find_mountains(arr):
    mountains = []
    n = len(arr)
    
    for i in range(1, n - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            mountains.append(i)
    
    return mountains

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

file = filenames[10]
# 7 deu problema, maybe 3 modes
data = np.loadtxt(PATH_SIZDIST+file, delimiter=',')
# # for i,file in enumerate(filenames):
for i in range(data.shape[0]):
# for i in range(20):
    dist     = data[i, 8:]
    diameter = standard_diameter
   

    dist = dist/integrate.simps(dist, diameter)
    derivative = np.diff(dist)
    ind_neg = np.where(derivative[:2] < 0)[0]
    if ind_neg.any() and ind_neg.shape[0] < 2:
        print("negative derivative")
        dist = dist[ind_neg[0]+1:]
        diameter = diameter[ind_neg[0]+1:]
    
    n_montains = len(find_mountains(dist[:10]))
    if n_montains > 1:
        dist = savgol_filter(dist, 4, 2)
    # # Create a new list with non-zero values from dist and corresponding values from diameter
    # fig, ax = plt.subplots()
    # ax.plot(diameter[:12], dist[:12], 'ob')
    # ax.grid()
    # plt.show()
    #**************************************************************************************************
    f = interpolate.interp1d(diameter, dist, kind='linear')
    diameter_interp = np.arange(diameter[0], diameter[-1], .01)
    # diameter_interp = np.linspace(diameter[0], diameter[-1], diameter.shape[0]*5)
    dist_interp     = f(diameter_interp)

    # Remover zeros aqui, e nao precisa abaixo
    mask_non_zero = np.where(dist_interp > 0)[0]
    if mask_non_zero.any():
        dist_new, diameter_new = dist_interp[mask_non_zero], diameter_interp[mask_non_zero]
    n0 = integrate.simps(dist_new, diameter_new)
    
    # fig, ax = plt.subplots()
    # ax.plot(diameter, dist, 'ob')
    # ax.plot(diameter_new, dist_new, '-r')
    # # ax.set_yscale('log')
    # ax.grid()
    # ax.set_ylabel(r'Counts [#]') 
    # # ax.set_ylim([.1, max(dist)+100])
    # plt.show()
    #**************************************************************************************************
    # data = np.loadtxt(PATH_SIZDIST+file, delimiter=',')
    # num  = np.zeros(data[:, 8:].shape)
    # for i in range(data.shape[0]):
    #     app_pas  = data[i, 7]
    #     volume   = app_pas*100*surface*10 # PAS * Area * time (10s)
    #                                      # (m s^-1)*100 * cm^2 * s -> cm^3
    #     num[i,:] = data[i, 8:]/volume
    
    # ntime       = data[:, 8:].shape[0]
    # counts_mean = np.mean(data[:, 8:],axis=0)
    # n0_count    = integrate.simps(counts_mean, diameter)
    # std_count   = np.std(data[:, 8:], axis=0)/np.sqrt(ntime)
    # #ipoints    = 17 # for more mode analisys
    # ipoints     = 11 # for more mode analisys
    # ***********************************************************
    # # Perform polynomial interpolation
    # interp_func = interpolate.interp1d(diameter, data[i,8+8:], kind='quaratic')
    # xnew = np.arange(diameter[0], diameter[-1], .01)
    # ynew = interp_func(xnew )
    # maximus = argrelextrema(ynew, np.greater)
    # minimus = argrelextrema(ynew, np.less)
    # # first mode analysis ***************************************
    # # x_data      = diameter[:ipoints]
    # # y_data      = counts_mean[:ipoints]
    # # s_data      = std_count[:ipoints]
    # # ***********************************************************
    # # second mode analysis **************************************
    # # x_data      = diameter[ipoints-1:ipoints+8]
    # # y_data      = counts_mean[ipoints-1:ipoints+8]
    # # s_data      = std_count[ipoints-1:ipoints+8]
    # # ************************************************************
    # # first and second mode analysis *****************************
    # x_data      = diameter[:ipoints+8]
    # y_data      = counts_mean[:ipoints+8]
    # s_data      = std_count[:ipoints+8]
    # # ************************************************************
    # #s_data = np.ones(ipoints)
    # #s_data = np.ones(8)
    # dnew        = np.arange(x_data[0], x_data[-1], .1)
    #print((np.mean(y_data[:15])/np.std(y_data[:15]))**2)
    # ************************************************************
    aux_dist     = np.diff(dist_new)
    global_min_x = np.argmin(aux_dist)
    xnew         = diameter_new[global_min_x:]
    ind_sep_aux  = np.where(aux_dist[global_min_x:] > -.0001)[0]
    if ind_sep_aux.any():
        ind_sep  = ind_sep_aux[0]
    else:
        ind_sep  = xnew.shape[0] -2
    xsep         = xnew[ind_sep]
    ynew         = dist_new[global_min_x:]

    # aux_line = np.arange(min(aux_dist), max(aux_dist), 0.001)
    # fig, ax  = plt.subplots()
    # ax.plot(diameter_new[:-1], aux_dist, '-or')
    # ax.plot(np.ones(aux_line.shape[0])*xnew[ind_sep], aux_line, color="k")
    # plt.show()
     # ************************************************************
    # plt.figure()
    # plt.plot(diameter_new[:-1], np.diff(dist_new))
    # plt.show()
    # x1 = diameter[diameter<=xsep]
    # x2 = diameter[diameter>xsep]
    # y1 = dist[diameter<=xsep]
    # y2 = dist[diameter>xsep]

    num_x2_original = sum(dist[diameter > xsep] != 0)
    num_x1_original = sum(dist[diameter <= xsep] != 0)

    x2_test = standard_diameter[standard_diameter>xsep]
    y2_test = data[i, 8:][standard_diameter>xsep]
    y1_test = data[i, 8:][standard_diameter<=xsep]
    # y2_test = savgol_filter(y2_test, 5, 3)
    ind_sep_x2 = find_point(y2_test)
    
    if n_montains == 1 or ind_sep_x2 != 0:
        valleys  = find_valleys(np.diff(y2_test[:ind_sep_x2]))
    else:
        valleys  = find_valleys(np.diff(y2_test))
    
    if len(valleys) > 1:
        dist_new = savgol_filter(dist_new, 100, 3)
    # # montains = np.diff(y2_test[:2])
    # fig, ax = plt.subplots()
    # ax.plot(x2_test[:-1], np.diff(y2_test,1), '-ob', label="valleys = %d, montains = %d"%(len(valleys), n_montains))
    # ax.plot(np.ones(x2_test.shape[0]-1)*x2_test[ind_sep_x2], np.diff(y2_test,1), color="k")
    # ax.legend()
    # # ax.plot(x1, y1, '-r')
    # ax.grid()
    # plt.show()
    x1 = diameter_new[diameter_new<=xsep]
    x2 = diameter_new[diameter_new>xsep]
    y1 = dist_new[diameter_new<=xsep]
    y2 = dist_new[diameter_new>xsep]

    n1 = integrate.simps(y1, x1)
    n2 = integrate.simps(y2, x2)

    m11  = integrate.simps(x1*y1, x1)/n1
    m12  = integrate.simps(x1**2*y1,x1)/n1
    var1 =  m12 - m11**2
    nu1 = m11**2/var1
    bt1 = var1/m11

    m21  = integrate.simps(x2*y2, x2)/n2
    m22  = integrate.simps(x2**2*y2,x2)/n2
    var2 =  m12 - m11**2
    nu2 = m21**2/var2
    bt2 = var2/m21
    
    # num_non_zero2 = np.where(y2 != 0)[0].shape[0]
    #print("Conc second mode",np.sum(y1), np.sum(y2))
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    f0       = axs[0].plot(diameter, dist, 'ob')
    f1       = axs[0].plot(diameter_new, dist_new, '-m')
    f30      = axs[0].plot(xnew[ind_sep], ynew[ind_sep], marker=">", color="k")
    f11 = axs[0].fill_between([x1[0], x1[-1]], [max(dist), max(dist)], alpha=.5)
    f12 = axs[0].fill_between([x2[0], x2[-1]], [max(dist), max(dist)], alpha=.5)
    # axs[0].set_yscale('log')
    # axs[0].set_ylim([.1, max(dist)+100])
    # if x1.shape[0] > 5 and num_non_zero2 < 5: # If using original distribution, less resolution 

    if num_x1_original > 3:
        if y1_test.sum() > 10 and len(valleys) == 0 and n_montains == 1: # If using interpolated distribution
            s_data = np.ones(x1.shape[0])
            initial_params=[n1, nu1, bt1]
            pcount, covcount = curve_fit(mixture_gamma, x1, y1, sigma=s_data, 
                                p0=initial_params)
            xnew = np.linspace(x1[0], x1[-1], 100)
            fnew = mixture_gamma(xnew,*pcount)
            residual  = y1 - mixture_gamma(x1,*pcount)
            chi2 = sum(residual)**2
            # print(bt1, pcount[2])
            f1 = axs[0].plot(xnew, fnew, '-r', label=r"1 mode - $\nu$=%.1f, $\beta$=%.1f, $\xi$=%.1f"%(pcount[1], pcount[2], chi2))
            axs[0].set_title("num_x2_original = %d, y2max = %.3f, n1/n0 = %.3f"%(num_x2_original, y2.max(), n1/n0))
            axs[0].set_ylabel(r'Counts [#]')
            axs[0].grid()  
            axs[0].legend()
            
            f4 = axs[1].plot(x1, residual/max(abs(residual)), '--ob',markersize=4.)
            axs[1].set_ylabel('Relative Res [#]')
            axs[1].set_xlabel(r'Diameter [$\mu$m]')
            axs[1].set_xlim([x1[0], x2[-1]])
            #axs[1].set_ylim([-1.2,1.2])
            axs[1].grid()
            # plt.suptitle(file)
            plt.show()

        # elif num_non_zero2  > 5: # If using original distribution, less resolution
        elif len(valleys) > 0 or n_montains > 1: 
            # print(n2/n0, num_x2_original)
            # initial_params=[n1, 2*n2, 12, 8, .3, .5]
            initial_params=[n1, n2, nu1, nu2, bt1, bt2]
            
            # Find the indices where dist is non-zero
            # non_zero_indices = np.where(dist != 0)[0]
            # dist_new, diameter_new = dist[non_zero_indices], diameter[non_zero_indices]
            s_data = np.ones(diameter_new.shape[0])

            pcount, covcount = curve_fit(mixture_gamma, diameter_new, dist_new, sigma=s_data, 
                                        p0=initial_params)
            
            dnew = np.arange(diameter_new[0], diameter_new[-1], .1)
            fnew = mixture_gamma(dnew,*pcount)
            residual  = dist_new - mixture_gamma(diameter_new,*pcount)
            chi2 = sum(residual)**2
            # print(nu1, pcount[2])
            f1       = axs[0].plot(dnew, fnew, '-r', label="%.1f, %.1f, %.1f,%.1f, %.1f, %.1f"%(pcount[0],pcount[1],pcount[2], pcount[3],pcount[4],pcount[5]))
            f2 = axs[0].plot(dnew, mixture_gamma(dnew, pcount[0],pcount[2],pcount[4]), '--r')
            f3 = axs[0].plot(dnew, mixture_gamma(dnew, pcount[1],pcount[3],pcount[5]), '--g')
            
            axs[0].set_title("sum y2 = %d, chi2 = %.3f"%(y2.sum(), chi2))
            axs[0].set_ylabel(r'Counts [#]')
            axs[0].grid()
            axs[0].legend()
            # axs[0].set_yscale('log')

            f4 = axs[1].plot(diameter_new, residual/max(abs(residual)), '--ob',markersize=4.)
            axs[1].set_ylabel('Relative Res [#]')
            axs[1].set_xlabel(r'Diameter [$\mu$m]')
            axs[1].set_xlim([x1[0], x2[-1]])
            axs[1].set_ylim([-1.2,1.2])
            axs[1].grid()
            # plt.suptitle(file)
            plt.show()
        
        # i_peaks  = identify_modes(dist, prominence=.1)
        # print(i_peaks)
        # ini      = 0
        # final    = i_peaks[0]+5 
        # w_guess  = []
        # nu_guess = []
        # sc_guess = []
        
        # fnew = mixture_gamma(dnew,*pcount)
        # residual  = dist[:ir_max] - mixture_gamma(diameter,*pcount)
        
        # f1       = axs[0].plot(dnew, fnew, '-r', label="%.1f, %.1f, %.1f,%.1f, %.1f, %.1f"\
        #                        %(pcount[0],pcount[1],pcount[2], pcount[3],pcount[4],pcount[5]))
        # for j in range(2):
        #     di       = diameter[ini:final]
        #     dist_new = dist[ini:final]
        #     w_guess.append(integrate.simps(dist_new,di))
        #     re = integrate.simps(di**3*dist_new, di)/integrate.simps(di**2*dist_new,di)
        #     rm = integrate.simps(di*dist_new, di)/w_guess[j]
        #     nu_guess.append(2*rm/(re-rm))
        #     sc_guess.append(rm/nu_guess[j])
        #     f12 = axs[0].fill_between([di[0], di[-1]], [5, 5], alpha=.5)
        #     ini   = final
        #     final = ir_max+1
        # initial_params=[w_guess[0], w_guess[1], nu_guess[0], nu_guess[0]-3, sc_guess[0], sc_guess[1]]
        # # N0 = integrate.simps(dist[ini:final],dist[ini:final])
        # # initial_params=[N0, n_tot-N0, .2, 15, 1, 4]
        # print(initial_params)
        # pcount, covcount = curve_fit(mixture_gamma, diameter, dist, sigma=s_data, 
        #                      p0=initial_params)
        # fnew = mixture_gamma(dnew,*pcount)
        # residual  = dist[:ir_max] - mixture_gamma(diameter,*pcount)
        
        # f1       = axs[0].plot(dnew, fnew, '-r', label="%.1f, %.1f, %.1f,%.1f, %.1f, %.1f"\
        #                        %(pcount[0],pcount[1],pcount[2], pcount[3],pcount[4],pcount[5]))
    
    # if file == 'cloud13b2.csv':
    #     #param0 = [.5, .6, 20., 3., 9., .5]
    #     initial_params = np.array([n0_count*.5, n0_count*.05, n0_count*.4, 20., 20., 10., 1/3., 1., 1/.6])
    #     #param0 = [1.e-5 ,1.e-6, 1.e-6, 25., 3., 14., 6, 10., 6]
    #     pcount, covcount = curve_fit(mixture_gamma, x_data, y_data, 
    #                              p0=initial_params)
        
    #     #result    = minimize(neg_log_likelihood, param0, method='Nelder-Mead')
    #     fnew = mixture_gamma(dnew,*pcount)
    #     residual  = y_data - mixture_gamma(x_data,*pcount)
    #     print(file)
    #     print(pcount[0],pcount[1],pcount[2])
    #     print(pcount[3],pcount[4])
    #     print(pcount[5],pcount[6])
    #     print(pcount[7],pcount[8])
    #     print("\n")
    # else:
    #     # first mode analysis *****************************************************************
    #     #initial_params = [10000, 15., 2]
    #     #pcount, covcount = curve_fit(gamma_pdf, x_data, y_data, 
    #     #                         p0=initial_params, sigma=s_data, bounds=(0,[1e7, 48, 10]))
    #     #fnew = gamma_pdf(dnew,*pcount)
    #     #residual  = y_data - gamma_pdf(x_data,*pcount)
    #     #print(pcount[1], pcount[2])
    #     # **************************************************************************************

    #     # first and second mode analysis *******************************************************
    #     initial_params = [1000, 100., 20., 16, 1/3., 1/2.]
    #     pcount, covcount = curve_fit(mixture_gamma, x_data, y_data, 
    #                              p0=initial_params, sigma=s_data)
    #     fnew = mixture_gamma(dnew,*pcount)
    #     residual  = y_data - mixture_gamma(x_data,*pcount)
    #     print(file)
    #     print(pcount[0],pcount[1])
    #     print(pcount[2],pcount[3])
    #     print(pcount[4],pcount[5])
    #     print("max:",xnew[maximus])
    #     print("min:",xnew[minimus])
    #     print("\n")
        # ***************************************************************************************
        
        # first, second and third mode analysis *************************************************
        #pcount, covcount = curve_fit(gamma_3pdf, x_data, y_data, 
        #                         p0=PARAM[file], sigma=s_data, bounds=BOUND[file])
        #fnew = gamma_3pdf(dnew,*pcount)
        #residual  = y_data - gamma_3pdf(x_data,*pcount)
        #print(file)
        #print(pcount[3],pcount[5],pcount[7])
        #print(pcount[4],pcount[6],pcount[8])
        #print("\n")

        #initial_params = np.array([n0_count*.8, n0_count*.5, n0_count*.2, 52, 57, 4, 8, 3, 1])
        #result = minimize(neg_log_likelihood, initial_params, method='Nelder-Mead')
        #print(pcount)
        #print(result.x)
        # *****************************************************************************************


    # f0 = axs[0].errorbar(diameter, counts_mean, yerr=std_count,
    #                       fmt='--ob', markersize=4.)
    
    # Uncomment for 1 mode gamma function analisys **********************************************************
    # f1 = axs[0].plot(dnew, fnew, '-r',
    #                 label=r"$\nu$ = %.2f"%(pcount[1]))
    # ********************************************************************************************************
    
#     # Uncomment for 2 mode gamma function analisys **********************************************************
#     f1 = axs[0].plot(dnew, fnew, '-r',
#                      label=r"$\nu_1$ = %.2f, $\nu_2$ = %.2f"%(pcount[2],pcount[4]))
    
#     # gamma_total= gamma_pdf(x_data, pcount[0],pcount[2],pcount[3]) + gamma_pdf(x_data, pcount[1],pcount[4],pcount[5])
#     # f1 = axs[0].plot(x_data, gamma_total, '-r',
#     #                  label=r"$\nu_1$ = %.2f, $\nu_2$ = %.2f"%(pcount[2],pcount[4]))
    
#     f2 = axs[0].plot(dnew, mixture_gamma(dnew, pcount[0],pcount[2],pcount[4]), '--m')
#     f3 = axs[0].plot(dnew, mixture_gamma(dnew, pcount[1],pcount[3],pcount[5]), '--g')
#     # ********************************************************************************************************

#     # Uncomment for 3 mode gamma function analisys **********************************************************
#     #f1 = axs[0].plot(dnew, fnew, '-r',
#     #                 label=r"$\nu_3$ = %.2f, $\nu_2$ = %.2f, $\nu_3$ = %.2f"%(pcount[3],pcount[5],pcount[7]))
    
#     #f2 = axs[0].plot(dnew, gamma_dist(dnew, pcount[0],pcount[3],pcount[4]), '--m')
#     #f3 = axs[0].plot(dnew, gamma_dist(dnew, pcount[1],pcount[5],pcount[6]), '--g')
#     #f4 = axs[0].plot(dnew, gamma_dist(dnew, pcount[2],pcount[7],pcount[8]), '--y')
#     # ********************************************************************************************************
   
#     axs[0].set_ylabel(r'Counts [#]')
#     axs[0].grid()  
#     axs[0].legend()
#     #axs[0].set_ylim([1e-2, 1.e4])
#     axs[0].set_yscale('log')

#     #f2 = axs[1].plot(diameter, num_mean, '--ob',markersize=4.)
#     #f3 = axs[1].plot(diameter, gamma_pdf(diameter,*pnum), '-r',
#     #                label=r"$\alpha$ = %.1f"%(pnum[0]))
#     f2 = axs[1].plot(x_data, residual/max(abs(residual)), '--ob',markersize=4.)
#     #axs[1].set_ylabel(r'# cm$^{-3}$')
#     axs[1].set_ylabel('Relative Res [#]')
#     axs[1].set_xlabel(r'Diameter [$\mu$m]')
#     axs[1].set_xlim([diameter[0], x_data[-1]+3])
#     axs[1].set_ylim([-1.2,1.2])
#     # axs[1].set_xscale("log")
#     #axs[1].ticklabel_format(axis='y', style='sci', scilimits=(4,4))
#     axs[1].grid()
#     plt.suptitle(file)
#     plt.show()
# #**************************************************************************************************
# 0: datenum
# 
# 1: ambient temperature (c)
# 
# 2: pas (m/s)
# 
# 3: number conc (/cm^3)
# 
# 4: LWC (g/m^3)
# 
# 5: MVD (um)
# 
# 6: ED (um)
# 
# 7: Applied PAS (m/s)
# 
# 8: Fog Monitor Bin 1
# 
# 9: ...
# 
#**************************************************************************************************

# data_test = np.loadtxt(PATH_SIZDIST+filenames[0], delimiter=',')
# i=7
# Nd0              = data_test[i, 3]                  # cm^-3
# sample_flow_rate = 1                           # m^3/minute
# app_pas          = data_test[i, 2]                  # m s^-1
# number           = data_test[i, 8:]                 # counts
# #aux = number/pas 
# #number          = np.mean(data_test[:, 8:],axis=0) # minute^-1

# surface = 0.24*10**(-2) # mm ^2 -> cm^2
# volume  = app_pas*100*surface*10 # PAS * Area * time
#                                  # time = 10s
#                                  # (m s^-1)*100 * cm^2 * s -> cm^3
# concentration = number/volume

# print( integrate.trapz(concentration, np.log(diameter)) )
# print( np.sum(concentration) )
# #print( np.sum(number) )
# print(Nd0)

# plt.figure()
# plt.plot(number, concentration, '-o')
# plt.xlabel("counts")
# plt.ylabel("counts/cm^3")
# #plt.xscale('log')
# plt.show()

# #**************************************************************************************************
# corr = np.diff(np.log(diameter))
# corr1 = np.zeros(30)
# for i in range(len(corr)):
#     corr1[i] = corr[i]

# corr1[-1] = corr[-1]
# conc_corr = concentration*corr1
# np.sum(conc_corr)
#corr = [corr, corr[-1]]


# for i in range(data.shape[0]):fig, axs = plt.subplots();axs.plot(diameter, data[i, 8:],color=np.random.rand(3),marker='*',label="%r"
#     ...: %(identify_modes(data[i, 8:29], prominence=.01, width=1)));axs.set_yscale("log");axs.legend()

# for i in range(20):fig, axs = plt.subplots();axs.plot(diameter, data[i, 8:],color=np.random.rand(3),marker='*',label="%r"%(identify_modes(data[i, 8:29], prominence=.01, width=1)));axs.plot(diameter[:-1], 10*np.abs(np.diff(data[i, 8:])),"r");axs.set_yscale("log");axs.legend();axs.grid();axs.set_xlim(3, 30)

# for i in range(20):f= interpolate.interp1d(diameter, data[i, 8:], kind="quadratic");fig, axs = plt.subplots();axs.plot(diameter, data[i, 8:],color=np.random.rand(3),marker='*',label="%r"%(identify_modes(data[i, 8:], prominence=.01, width=1)));axs.plot(diameter[:-1],10*np.abs(np.diff(data[i, 8:])),"r");axs.plot(np.arange(diameter[0], diameter[-1], .1),f(np.arange(diameter[0], diameter[-1], .1)),'r');axs.set_yscale("log");axs.legend();axs.grid();axs.set_xlim(3, 30)