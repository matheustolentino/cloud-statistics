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
from scipy.stats import mode
from sklearn.mixture import GaussianMixture
from itertools import chain
import seaborn as sns

plt.ion()
plt.close('all')
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.5})
#**************************************************************************************************
PATH_SIZDIST = '../../../data/cloud_sizdist/'
PATH_FIG     = '../figures/'
#**************************************************************************************************
filenames = []
for file in os.listdir(PATH_SIZDIST):
    if file.startswith("cloud"):
        filenames.append(file)
        #print(file)
#**************************************************************************************************
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

def mixture_lognormal(x: np.ndarray, *params: float) -> np.ndarray:
    """
    Calculate the probability density function (PDF) of a mixture of lognormal distributions.

    Parameters:
        x (np.ndarray): Input values at which to evaluate the PDF.
        params (float): Variable-length argument list containing the parameters.
                        The parameters should be provided in the following order:
                        [weights, means, sigmas]

    Returns:
        np.ndarray: The PDF values at the given input values x.
    """
    num_distributions = len(params) // 3
    weights = params[:num_distributions]
    means   = params[num_distributions:2*num_distributions]
    sigmas  = params[2*num_distributions:]
    pdf = np.zeros_like(x)
    for w, mu, sigma in zip(weights, means, sigmas):
        pdf += w * (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))
    return pdf

def calculate_gamma_parameters(diameter, counts, peak_info=None, mode=0):
    
    if peak_info is None:
        x = diameter
        y = counts
    else:
        ini = peak_info['left_bases'][mode]
        end = peak_info['right_bases'][mode]
        x = diameter[ini:end]
        y = counts[ini:end]

    n = integrate.simps(y, x)
    first_mom = integrate.simps(x*y, x)/n
    second_mom  = integrate.simps(x**2*y,x)/n
    var1  =  second_mom - first_mom**2
    nu    = first_mom**2/var1
    scale = var1/first_mom

    return  n, nu, scale, x, y

def calculate_lognormal_parameters(diameter, counts, peak_info=None, mode=0):
    if peak_info is None:
        x = diameter
        y = counts
    else:
        ini = peak_info['left_bases'][mode]
        end = peak_info['right_bases'][mode]
        x = diameter[ini:end]
        y = counts[ini:end]

    n = integrate.simps(y, x)
    first_mom = integrate.simps(x*y, x)/n
    second_mom  = integrate.simps(x**2*y,x)/n
    mu    = np.log(first_mom**2/np.sqrt(second_mom))
    sigma = np.sqrt(second_mom/first_mom**2)

    # Arithmetic moments section shows how to obtain logno,mal parameters mu y sigma^2: https://en.wikipedia.org/wiki/Log-normal_distribution

    return  n, mu, sigma, x, y

def plot_fit_gamma(
    param: List[float],
    diameter: List[float],
    filtered_dist: List[float],
    diameter_interp: List[float],
    dist_interp: List[float],
    fnew: List[float],
    residual: int | List[float],
    get_params: bool,
) -> None:
    """
    Plot and show distributions with parameter information.

    Args:
        param (List[float]): List of parameters, including weights, shapes, and scales.
        diameter (List[float]): List of diameters.
        filtered_dist (List[float]): List of filtered distributions.
        diameter_interp (List[float]): List of interpolated diameters.
        dist_interp (List[float]): List of interpolated distributions.
        fnew (List[float]): List of new distributions.
        residual (List[float]): List of residuals.
        peaks_res (int): Number of peaks for the residuals.

    Returns:
        None
    """
    num_distributions = len(param) // 3
    weights = param[:num_distributions]
    shapes = param[num_distributions:2 * num_distributions]
    scales = param[2 * num_distributions:]
   
    # Create the table format
    table_format = f"{'weights':<10} {'shapes':<10} {'scales':<10}\n"
    table_format += '-' * 40 + '\n'

    # Populate the table with data
    for i in range(len(weights)):
        table_format += f"{weights[i]:<10.1f} {shapes[i]:<10.2f} {scales[i]:<10.2f}\n"

    # Add parameter information as text
    text_x = 0.3  # X-coordinate for text annotations
    text_y = 0.8  # Y-coordinate for text annotations

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(7, 6))
    f0 = axs[0].plot(diameter, filtered_dist, 'om', label='Smoothed distribution')   
    # axs[0].plot(diameter_interp, dist_interp, '--m')
    axs[0].plot(diameter_interp, fnew, '-r', label='Fit')
    if num_distributions > 1:
        for i in range(num_distributions):
            # axs[0].plot(diameter_interp, weights[i] * gamma.pdf(diameter_interp, shapes[i], scale=scales[i]), '--')
            axs[0].plot(diameter_interp, mixture_gamma(diameter_interp, weights[i], shapes[i], scales[i]), '--')
    # axs[0].set_yscale('log')
    axs[0].set_ylabel(' Count [#]')
    # Create a white background text box for the table
    axs[0].text(
        text_x,
        text_y,
        table_format,
        transform=axs[0].transAxes,
        backgroundcolor='white',
        fontsize=10,  # Adjust font size as needed
        verticalalignment='top'  # Adjust vertical alignment as needed
    )
    axs[0].grid()
    axs[0].legend()

    f1 = axs[1].plot(diameter_interp, residual, '--ob', markersize=5., label='get_params = %r'%get_params)
    axs[1].set_ylabel(' Res [#]')
    axs[1].set_xlabel(r'Diameter [$\mu$m]')
    axs[1].set_xlim([diameter_interp[0], diameter_interp[-1]])
    axs[1].legend()
    axs[1].grid()
    plt.show()

def plot_fit_lognormal(
    param: List[float],
    diameter: List[float],
    filtered_dist: List[float],
    diameter_interp: List[float],
    dist_interp: List[float],
    fnew: List[float],
    residual: int | List[float],
    get_params: bool,
    title: str
) -> None:
    """
    Plot and show distributions with parameter information.

    Args:
        param (List[float]): List of parameters, including weights, means, and sigmas.
        diameter (List[float]): List of diameters.
        filtered_dist (List[float]): List of filtered distributions.
        diameter_interp (List[float]): List of interpolated diameters.
        dist_interp (List[float]): List of interpolated distributions.
        fnew (List[float]): List of new distributions.
        residual (List[float]): List of residuals.
        peaks_res (int): Number of peaks for the residuals.

    Returns:
        None
    """
    num_distributions = len(param) // 3
    weights = param[:num_distributions]
    means = param[num_distributions:2 * num_distributions]
    sigmas = param[2 * num_distributions:]
   
    # Create the table format
    table_format = f"{'weights':<10} {'means':<10} {'sigmas':<10}\n"
    table_format += '-' * 40 + '\n'

    # Populate the table with data
    for i in range(len(weights)):
        table_format += f"{weights[i]:<10.1f} {means[i]:<10.2f} {sigmas[i]:<10.2f}\n"

    # Add parameter information as text
    text_x = 0.3  # X-coordinate for text annotations
    text_y = 0.8  # Y-coordinate for text annotations

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
    f0 = axs[0].plot(diameter, filtered_dist, 'om', label='Smoothed distribution')   
    # axs[0].plot(diameter_interp, dist_interp, '--m')
    axs[0].plot(diameter_interp, fnew, '-r', label='Fit')
    if num_distributions > 1:
        for i in range(num_distributions):
            # axs[0].plot(diameter_interp, weights[i ] * gamma.pdf(diameter_interp, shapes[i], scale=scales[i]), '--')
            axs[0].plot(diameter_interp, mixture_lognormal(diameter_interp, weights[i], means[i], sigmas[i]), '--')
    # axs[0].set_yscale('log')
    axs[0].set_ylabel(' Count [#]')
    # Create a white background text box for the table
    axs[0].text(
        text_x,
        text_y,
        table_format,
        transform=axs[0].transAxes,
        backgroundcolor='white',
        fontsize=10,  # Adjust font size as needed
        verticalalignment='top'  # Adjust vertical alignment as needed
    )
    axs[0].grid()
    axs[0].legend()
    axs[0].set_title(title)
    # axs[0].set_yscale('log')    

    f1 = axs[1].plot(diameter_interp, residual, '--ob', markersize=5., label='get_params = %r'%get_params)
    axs[1].set_ylabel(' Res [#]')
    axs[1].set_xlabel(r'Diameter [$\mu$m]')
    axs[1].set_xlim([diameter_interp[0], diameter_interp[-1]])
    axs[1].legend()
    axs[1].grid()
    plt.show()


#**************************************************************************************************
# Teste mixture of gamma functions
# **************************************************************************************************
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
    
#**************************************************************************************************
# Teste mixture of lognormal functions
# **************************************************************************************************
# x = np.linspace(0.5, 100, 100)
# true_params = [50, 10, np.log(20), np.log(50), .3, .2]
# y = mixture_lognormal(x, *true_params)

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111)
# ax.plot(x, y, 'o')
# ax.set_yscale('log')
# ax.set_ylim([.01, 50])

# peaks, peak_info = find_peaks(y, height=0.2, prominence=0.01)
# if peaks.any():
#     for i, peak in enumerate(peaks):
#         n, mu, sigma, xnew, ynew = calculate_lognormal_parameters(x, y, peak_info, i)
#         print(f"n = {n}, mu = {mu}, sigma = {sigma}, mode = {x[peak]}")
#         new_par, new_cov = curve_fit(mixture_lognormal, xnew, ynew, p0=[n, mu, sigma])
#         ax.plot(x, mixture_lognormal(x, *new_par), '--', label=f"Mode {x[peak]:.1f}")
# ax.set_xlabel('Diameter [um]')
# ax.set_ylabel('Count [#]')
# ax.legend()
# plt.show()
#**************************************************************************************************
SHAPE_THRESHOLD = 50
DISTANCE_MODE_THRESHOLD = 6

standard_diameter    = np.loadtxt(PATH_SIZDIST+'diameters.csv', delimiter=',')
surface     = 0.24*10**(-2) # mm ^2 -> cm^2
#fitted_dist = np.zeros((len(filenames), diameter.shape[0]))
#**************************************************************************************************
def get_parameters_gamma_function(filenames, standard_diameter, PATH_SIZDIST):
    display        = False
    diameters_eff  = []
    diameters_mean = []
    fitted_weights = []
    fitted_shapes  = []
    fitted_scales  = []
    count_occurence = 0
    dic_param      = {}
    count_distributions = 0
    count_used_distributions = 0

    for file in filenames:
        data = np.loadtxt(PATH_SIZDIST+file, delimiter=',')
        for i_file in range(data.shape[0]):
            dist     = data[i_file, 8:]
            diameter = standard_diameter

            if dist.max() > 5:
                filtered_dist = gaussian_filter1d(dist, 1)
                f = interpolate.interp1d(diameter, filtered_dist, kind='cubic')
                delta_diameter = .01
                diameter_interp = np.arange(diameter[0], diameter[-1], delta_diameter)
                dist_interp     = f(diameter_interp)

                peaks, peak_info = find_peaks(dist_interp, height=5, prominence=5)

                try:
                    if not peaks.any():
                        n, nu, scale, _, _ = calculate_gamma_parameters(diameter_interp, dist_interp)
                        initial_params=[n, nu, scale]

                        param, pcov = curve_fit(mixture_gamma, diameter_interp, dist_interp, 
                                                p0=initial_params)

                    elif peaks.shape[0] == 1:
                        n, nu, scale, x, y = calculate_gamma_parameters(diameter_interp, 
                                                                        dist_interp, 
                                                                        peak_info)
                        initial_params=[n, nu, scale]

                        param, pcov = curve_fit(mixture_gamma, x, y, 
                                                p0=initial_params)

                        fnew = mixture_gamma(x, *param)
                        residual  = y - fnew

                        peaks_res, peak_res_info = find_peaks(residual, height=5, prominence=10, width=5/delta_diameter)

                        if peaks_res.any():
                            n, nu, scale, _, _ = calculate_gamma_parameters(x, y, peak_res_info)
                            initial_params=[param[0], n, param[1], nu, param[2], scale]
                            n_param = len(initial_params)
                            param_aux, pcov_aux = curve_fit(mixture_gamma, x, y, 
                                                p0=initial_params, maxfev=2000)

                            frac = param_aux[1]/param_aux[0]

                            if param_aux[1] > 0 and param_aux[2] > 5 and param_aux[3] > 5 and frac > .1 and frac < 2:
                                param = param_aux
                                pcov  = pcov_aux

                    elif peaks.shape[0] > 1:
                        ns     = []
                        nus    = []
                        scales = []
                        for i, peak in enumerate(peaks):
                            n, nu, scale, _, _ = calculate_gamma_parameters(diameter_interp,
                                                                    dist_interp,
                                                                    peak_info,
                                                                    i)
                            ns.append(n)
                            nus.append(nu)
                            scales.append(scale)
                        initial_params = ns + nus + scales
                        n_param = len(initial_params)
                        bounds = (0, n_param*[np.inf])

                        param, pcov = curve_fit(mixture_gamma, diameter_interp, dist_interp, 
                                                p0=initial_params, maxfev=2000, bounds=bounds)

                    num_distributions = len(param) // 3
                    weights = param[:num_distributions]
                    shapes = param[num_distributions:2 * num_distributions]
                    scales = param[2 * num_distributions:]

                    dmode_aux = (shapes-1)*scales
                    dmean_aux = scales*shapes
                    deff_aux  = scales*(shapes + 2)

                    n_modes = len(dmode_aux)
                    distance_mode = 999*np.ones(n_modes)
                    if n_modes > 1:
                        distance_mode = np.abs(np.diff(dmode_aux))

                    fnew = mixture_gamma(diameter_interp, *param)
                    residual = dist_interp - fnew
                    chi = np.mean(residual)

                    possible_shape = np.all(shapes < SHAPE_THRESHOLD)
                    possible_mode_distance = np.all(distance_mode > DISTANCE_MODE_THRESHOLD)

                    get_params = np.all([possible_shape,possible_mode_distance])

                    if display and n_modes > 1:
                        count_occurence += 1
                        plot_fit_gamma(param, diameter, filtered_dist, diameter_interp, dist_interp, fnew, residual, get_params)
                        if count_occurence > 40:
                            set_trace()
                            plt.close('all')
                            count_occurence = 0
                        display = False

                    if get_params:
                        diameters_mean.append(dmean_aux.tolist())
                        diameters_eff.append(deff_aux.tolist())
                        fitted_weights.append(weights.tolist())
                        fitted_shapes.append(shapes.tolist())
                        fitted_scales.append(scales.tolist())
                        count_used_distributions += 1

                except RuntimeError as e:
                    print(f"Error - Curve_fit failed: {e}")
                    fig, ax = plt.subplots()
                    ax.plot(diameter, filtered_dist, 'or')
                    ax.plot(diameter_interp, dist_interp, '--m')
                    ax.grid()
                    plt.show()

            count_distributions += 1

    flattened_diameter_mean = np.array( list(chain(*diameters_mean)) )
    flattened_diameter_eff = np.array( list(chain(*diameters_eff)) )
    flattened_weights = np.array( list(chain(*fitted_weights)) )
    flattened_shapes = np.array( list(chain(*fitted_shapes)) )
    flattened_scales = np.array( list(chain(*fitted_scales)) )

    return flattened_diameter_mean, flattened_diameter_eff, flattened_weights, flattened_shapes, flattened_scales, count_distributions, count_used_distributions

def get_parameters_lognomal_function(filenames, standard_diameter, PATH_SIZDIST, surface=None):
    display_plot        = True
    diameters_eff  = []
    diameters_mean = []
    fitted_weights = []
    fitted_means   = []
    fitted_sigmas  = []
    fitted_concentration = []
    count_occurence = 0
    dic_param      = {}
    count_distributions = 0
    count_used_distributions = 0

    for file in filenames:
        data = np.loadtxt(PATH_SIZDIST+file, delimiter=',')
        for i_time in range(data.shape[0]):
            dist     = data[i_time, 8:]
            # Now, converto to number concentration [# cm^-3]
            # if surface is not None:
            #     app_pas   = data[i_time, 2]  # m s^-1
            #     volume    = app_pas*100*surface*10 # PAS * Area * time
            #                                        # time = 10s
            #                                        # (m s^-1)*100 * cm^2 * s -> cm^3
            #     dist_concentration  = dist/volume
            diameter = standard_diameter
            get_params_plot = False
            get_parameter = False
            if dist.max() > 5:
                filtered_dist = gaussian_filter1d(dist, 1)
                f = interpolate.interp1d(diameter, filtered_dist, kind='cubic')
                delta_diameter = .01
                diameter_interp = np.arange(diameter[0], diameter[-1], delta_diameter)
                dist_interp     = f(diameter_interp)

                peaks, peak_info = find_peaks(dist_interp, height=5, prominence=5)

                try:
                    if not peaks.any():
                        print('No peaks found')
                        title_plot = 'No peaks found'
                        get_params_plot = True
                        n, mu, sigma, x, y = calculate_lognormal_parameters(diameter_interp, dist_interp)
                        initial_params=[n, mu, sigma]

                        param, pcov = curve_fit(mixture_lognormal, x, y, 
                                                p0=initial_params)
                        get_parameter = True
                    elif peaks.shape[0] == 1:
                        print('One peak found')
                        n, mu, sigma, x, y = calculate_lognormal_parameters(diameter_interp, 
                                                                        dist_interp, 
                                                                        peak_info)
                        initial_params=[n, mu, sigma]

                        param, pcov = curve_fit(mixture_lognormal, x, y, 
                                                p0=initial_params)

                        fnew = mixture_lognormal(x, *param)
                        residual  = y - fnew

                        peaks_res, peak_res_info = find_peaks(residual, height=5, prominence=10, width=5/delta_diameter)
                        
                        if peaks_res.any():
                            n, mu, sigma, _, _ = calculate_lognormal_parameters(x, y, peak_res_info)
                            initial_params=[param[0], n, param[1], mu, param[2], sigma]
                            n_param = len(initial_params)
                            param_aux, pcov_aux = curve_fit(mixture_lognormal, x, y, 
                                                p0=initial_params, maxfev=2000)

                            frac = param_aux[1]/param_aux[0]
                            
                            # Verifications:
                            # 1. The shape parameter is positive
                            # 2. The mode distance is greater than a threshold
                            n_aux       = len(param_aux) // 3
                            weights_aux = param_aux[:n_aux]
                            means_aux  = param_aux[n_aux:2 * n_aux]
                            sigmas_aux = param_aux[2 * n_aux:]
                            dmode_aux  = np.exp(means_aux - sigmas_aux**2)
                            diff_mode  = np.abs( np.diff(dmode_aux) )
                            title_plot = 'One peak found' +f"- Diff mode = {diff_mode}"
                            if np.all(weights_aux > 1) and np.all(diff_mode > DISTANCE_MODE_THRESHOLD) and np.all(diff_mode < 50) and np.all(means_aux < 10):
                                param = param_aux
                                pcov  = pcov_aux
                                get_params_plot = True
                        get_parameter = True
                    elif peaks.shape[0] > 1:
                        print('More than one peak found')
                        ns     = []
                        mus    = []
                        sigmas = []
                        for i, peak in enumerate(peaks):
                            n, mu, sigma, _, _ = calculate_lognormal_parameters(diameter_interp,
                                                                    dist_interp,
                                                                    peak_info,
                                                                    i)
                            ns.append(n)
                            mus.append(mu)
                            sigmas.append(sigma)
                        initial_params = ns + mus + sigmas
                        n_param = len(initial_params)
                        bounds = (0, n_param*[np.inf])

                        param, pcov = curve_fit(mixture_lognormal, diameter_interp, dist_interp, 
                                                p0=initial_params, maxfev=2000, bounds=bounds)
                        
                        n_aux       = len(param) // 3
                        weights_aux = param[:n_aux]
                        means_aux  = param[n_aux:2 * n_aux]
                        sigmas_aux = param[2 * n_aux:]
                        dmode_aux  = np.exp(means_aux - sigmas_aux**2)
                        diff_mode  = np.abs( np.diff(dmode_aux) )
                        title_plot = 'More than one peak found' +f"- Diff mode = {diff_mode}"
                        if np.all(weights_aux > 1) and np.all(diff_mode > DISTANCE_MODE_THRESHOLD) and np.all(diff_mode < 50) and np.all(means_aux < 10):
                            get_params_plot = True
                            get_parameter = True
                        
                    num_distributions = len(param) // 3
                    weights = param[:num_distributions]
                    # Now, converto to number concentration [# cm^-3]
                    if surface is not None:
                        app_pas   = data[i_time, 2]  # m s^-1
                        volume    = app_pas*100*surface*10 # PAS * Area * time
                                                           # time = 10s
                                                           # (m s^-1)*100 * cm^2 * s -> cm^3
                        weight_concentration  = weights/volume
                    
                    means = param[num_distributions:2 * num_distributions]
                    sigmas = param[2 * num_distributions:]
                   
                    dmode_aux = np.exp(means - sigmas**2)
                    dmean_aux = np.exp(means + sigmas**2/2)
                    deff_aux  = np.exp(means + sigmas**2)

                    n_modes = len(weights)
                    distance_mode = 999*np.ones(n_modes)
                    if n_modes > 1:
                        distance_mode = np.abs(np.diff(dmode_aux))

                    fnew = mixture_lognormal(diameter_interp, *param)
                    residual = dist_interp - fnew
                    chi = np.mean(residual)

                    # if get_params:
                    #     print('Display plot')
                    #     count_occurence += 1
                    #     plot_fit_lognormal(param, diameter, filtered_dist, diameter_interp, dist_interp, fnew, residual, get_params, title_plot)
                    #     if count_occurence > 30:
                    #         set_trace()
                    #         plt.close('all')
                    #         count_occurence = 0
                    #     # display = False

                    if get_parameter:
                        diameters_mean.append(dmean_aux.tolist())
                        diameters_eff.append(deff_aux.tolist())
                        fitted_weights.append(weights.tolist())
                        fitted_means.append(means.tolist())
                        fitted_sigmas.append(sigmas.tolist())
                        fitted_concentration.append(weight_concentration.tolist())
                        count_used_distributions += 1

                except RuntimeError as e:
                    print(f"Error - Curve_fit failed: {e}")
                    fig, ax = plt.subplots()
                    ax.plot(diameter, filtered_dist, 'or')
                    ax.plot(diameter_interp, dist_interp, '--m')
                    ax.grid()
                    plt.show()
            count_distributions += 1

    flattened_diameter_mean = np.array( list(chain(*diameters_mean)) )
    flattened_diameter_eff = np.array( list(chain(*diameters_eff)) )
    flattened_weights = np.array( list(chain(*fitted_weights)) )
    flattened_means = np.array( list(chain(*fitted_means)) )
    flattened_sigmas = np.array( list(chain(*fitted_sigmas)) )
    flattened_concentration = np.array( list(chain(*fitted_concentration)) )

    return flattened_diameter_mean, flattened_diameter_eff, flattened_weights, flattened_concentration, flattened_means, flattened_sigmas, count_distributions, count_used_distributions

lognorm_diameter_mean, lognorm_diameter_eff, lognorm_weights, lognorm_concentration, lognorm_mu, lognorm_sigmas, lognorm_dist_tot, lognorm_dist_used = get_parameters_lognomal_function(filenames, standard_diameter, PATH_SIZDIST, surface=surface)

with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(17, 9))
    f0 = sns.histplot(lognorm_diameter_mean, stat='count',
                            binwidth=1,
                            kde=True,
                            color='red',
                            alpha=0.4,
                            linewidth=1.2,
                            element='bars',
                            ax=axs[0])
    axs[0].set_xlabel('Mean Diameter [um]')
    axs[0].set_ylabel('Count')

    f1 = sns.histplot(lognorm_diameter_eff, stat='count',
                            binwidth=1,
                            kde=True,
                            color='green',
                            alpha=0.4,
                            linewidth=1.2,
                            element='bars',
                            ax=axs[1])
    axs[1].set_xlabel('Effective Diameter [um]')
    axs[1].set_ylabel('Count')
    axs[1].set_xlim([0, 30])
    # putting sup title
    fig.suptitle(r'N$_{dist}$ = %d, N$_{used}$ = %d'%(lognorm_dist_tot, lognorm_dist_used))
    # fig.savefig(PATH_FIG+'histogram_mean_eff_diameters.png', dpi=300)
    plt.tight_layout()
    plt.show()

    # find_modes_with_gmm(lognorm_diameter_mean, 2, nbin=40, title='Histogram of Mean Diameters')
    # find_modes_with_gmm(lognorm_diameter_eff, 2, nbin=40, title='Histogram of Effective Diameters')

# Plot the second histogram
fig, axs = plt.subplots(1, 3, figsize=(17, 7))
f0 = sns.histplot(lognorm_mu, stat='count',
                    binwidth=.1,
                    kde=True,
                    color='magenta',
                    alpha=0.4,
                    linewidth=1.2,
                    element='bars',
                    ax=axs[0])

axs[0].set(xlabel=r'$\mu$', ylabel='Count')
mean_mu = np.mean(lognorm_mu)
median_mu = np.median(lognorm_mu)
axs[0].legend([f"Mean: {mean_mu:.2f}", f"Median: {median_mu:.2f}"])
plt.show()

# Plot the third histogram
f2 = sns.histplot(lognorm_sigmas, stat='count',
                kde=True,
                binwidth=.02,
                color='blue',
                alpha=0.4,
                linewidth=1.2,
                ax=axs[1])
axs[1].set(xlabel=r'$\sigma$', ylabel='Count')
mean_sigma = np.mean(lognorm_sigmas)
median_sigma = np.median(lognorm_sigmas)
axs[1].legend([f"Mean: {mean_sigma:.2f}", f"Median: {median_sigma:.2f}"])
axs[1].set_xlim([0, .8])

f3 = sns.histplot(lognorm_concentration, stat='count',
                kde=True,
                binwidth=30,
                color='green',
                alpha=0.4,
                linewidth=1.2,
                ax=axs[2])
axs[2].set(xlabel=r'$\rho$ (cm$^{-3}$)', ylabel='Count')
mean_concentration = np.mean(lognorm_concentration)
median_concentration = np.median(lognorm_concentration)
axs[2].legend([f"Mean: {mean_concentration:.2f}", f"Median: {median_concentration:.2f}"])
fig.subplots_adjust(wspace=0.3)
fig.suptitle(r'N$_{dist}$ = %d, N$_{used}$ = %d'%(lognorm_dist_tot, lognorm_dist_used))
fig.savefig(PATH_FIG+'histogram_mu_sigma_concentration_lognormal.png', dpi=300)
plt.tight_layout()
plt.show()
# find_modes_with_gmm(lognorm_mu, 1, nbin=30, title='Histogram of Means')
# find_modes_with_gmm(lognorm_sigmas, 2, nbin=40, title='Histogram of Sigmas')
#**************************************************************************************************
set_trace()
#**************************************************************************************************
gamma_diameter_mean, gamma_diameter_eff, gamma_weights, gamma_shapes, gamma_scales, gamma_dist_tot, gamma_dist_used = get_parameters_gamma_function(filenames, standard_diameter, PATH_SIZDIST)

with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(17, 9))
    f0 = sns.histplot(gamma_diameter_mean, stat='count',
                            binwidth=1, 
                            kde=True, 
                            color='red', 
                            alpha=0.4,
                            linewidth=1.2, 
                            element='bars',
                            ax=axs[0])
    axs[0].set_xlabel('Mean Diameter [um]')
    axs[0].set_ylabel('Count')
    mean_mean = np.mean(gamma_diameter_mean)
    median_mean = np.median(gamma_diameter_mean)
    axs[0].legend([f"Mean: {mean_mean:.2f}", f"Median: {median_mean:.2f}"])

    f1 = sns.histplot(gamma_diameter_eff, stat='count',
                            binwidth=1, 
                            kde=True, 
                            color='green', 
                            alpha=0.4,
                            linewidth=1.2, 
                            element='bars',
                            ax=axs[1])
    axs[1].set_xlabel('Effective Diameter [um]')
    axs[1].set_ylabel('Count')
    axs[1].set_xlim([0, 30])
    mean_eff = np.mean(gamma_diameter_eff)
    median_eff = np.median(gamma_diameter_eff)
    axs[1].legend([f"Mean: {mean_eff:.2f}", f"Median: {median_eff:.2f}"])
    # putting sup title
    fig.suptitle(r'N$_{dist}$ = %d, N$_{used}$ = %d'%(gamma_dist_tot, gamma_dist_used))
    fig.savefig(PATH_FIG+'histogram_mean_eff_diameters_gamma_dist.png', dpi=300)
    plt.tight_layout()
    plt.show()

    # find_modes_with_gmm(gamma_diameter_mean, 2, nbin=40, title='Histogram of Mean Diameters')
    # find_modes_with_gmm(gamma_diameter_eff, 2, nbin=40, title='Histogram of Effective Diameters')


# Plot the second histogram
fig, axs = plt.subplots(1, 2, figsize=(17, 9))
f0 = sns.histplot(gamma_shapes, stat='count',
                    binwidth=2, 
                    kde=True, 
                    color='magenta', 
                    alpha=0.4, 
                    linewidth=1.2, 
                    element='bars',
                    ax=axs[0])
axs[0].set(xlabel=r'$\nu$', ylabel='Count')
axs[0].set_title('Histogram of Shapes')
mean_shapes = np.mean(gamma_shapes)
median_shapes = np.median(gamma_shapes)
axs[0].legend([f"Mean: {mean_shapes:.2f}", f"Median: {median_shapes:.2f}"])
plt.show()
# Plot the third histogram
f2 = sns.histplot(gamma_scales, stat='count', 
                bins=40, 
                kde=True,
                binwidth=.1, 
                color='blue',
                alpha=0.4,  
                linewidth=1.2, 
                ax=axs[1])
axs[1].set(xlabel=r'$\theta$', ylabel='Count')
axs[1].set_title('Histogram of Scales')
axs[1].set_xlim([0, 3])
mean_scales = np.mean(gamma_scales)
median_scales = np.median(gamma_scales)
axs[1].legend([f"Mean: {mean_scales:.2f}", f"Median: {median_scales:.2f}"])
fig.suptitle(r'N$_{dist}$ = %d, N$_{used}$ = %d'%(gamma_dist_tot, gamma_dist_used))
fig.savefig(PATH_FIG+'histogram_shapes_scales_gamma_dist.png', dpi=300)
plt.tight_layout()
plt.show()
    # find_modes_with_gmm(gamma_shapes, 1, nbin=30, title='Histogram of Shapes')
    # find_modes_with_gmm(gamma_scales, 2, nbin=40, title='Histogram of Scales')