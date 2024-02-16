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

def find_modes_with_gmm(data, n_components=2, plot=True, nbin=60, title='Histogram'):
    # Fit a Gaussian Mixture Model to the data
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data.reshape(-1, 1))

    # Get the means and covariances of the GMM components
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()

    # Sort components by mean (modes)
    sorted_indices = np.argsort(means)
    modes = means[sorted_indices]
    mode_covariances = covariances[sorted_indices]

    if plot:
        # Plot the data and GMM components
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=nbin, density=True, alpha=0.5, color='blue', label='Data Histogram')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'GMM with Modes - {title}')
        
        for mode, mode_cov in zip(modes, mode_covariances):
            x = np.linspace(mode - 3 * np.sqrt(mode_cov), mode + 3 * np.sqrt(mode_cov), 1000)
            y = (1.0 / (np.sqrt(2 * np.pi * mode_cov))) * np.exp(-0.5 * ((x - mode) ** 2) / mode_cov)
            plt.plot(x, y, label=f'Mode {mode:.2f}')

        plt.legend()
        plt.show()

    return modes, mode_covariances

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

def plot_and_show_distributions(
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
SHAPE_THRESHOLD = 50
DISTANCE_MODE_THRESHOLD = 3

standard_diameter    = np.loadtxt(PATH_SIZDIST+'diameters.csv', delimiter=',')
surface     = 0.24*10**(-2) # mm ^2 -> cm^2
display     = False
#fitted_dist = np.zeros((len(filenames), diameter.shape[0]))
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
    # # for i,file in enumerate(filenames):
    # fitted_v = np.zeros((data.shape[0], 5))
    # fitted_reff = np.zeros((data.shape[0], 5))
    # fitted_rmode = np.zeros((data.shape[0], 5))
    # fitted_rmean = np.zeros((data.shape[0], 5))
    for i_file in range(data.shape[0]):
    # for i_file in range(50):
        dist     = data[i_file, 8:]
        diameter = standard_diameter
        
        if dist.max() > 5:
            # Apply Gaussian filter to smooth the distribution
            filtered_dist = gaussian_filter1d(dist, 1)
            
            # Interpolate the distribution
            f = interpolate.interp1d(diameter, filtered_dist, kind='cubic')
            delta_diameter = .01
            diameter_interp = np.arange(diameter[0], diameter[-1], delta_diameter)
            dist_interp     = f(diameter_interp)

            # Find peaks in the distribution
            peaks, peak_info = find_peaks(dist_interp, height=5, prominence=5)
            # HEIGHT: This parameter specifies the minimum height (amplitude) 
            # that a point must have to be considered a peak. 
            # Any point in the input array with a value less than this height will 
            # be ignored. This can be used to filter out small or insignificant peaks.
            
            # PROMINENCE: This parameter specifies the minimum prominence of peaks.
            # The prominence of a peak is the minimum height by which a peak is separated 
            # from its neighboring valleys. If a peak's prominence is less than 
            # the specified value, it will not be considered a peak.

            # print("diameter = %f, peaks = %r, left_diameter = %r, right_diameter = %r"%(data[i, 0], 
            #                                                                          peaks, 
            #                                                                          diameter_interp[peak_info['left_bases']], 
            #                                                                          diameter_interp[peak_info['right_bases']]))

            try:
                if not peaks.any():
                    n, nu, scale, _, _ = calculate_gamma_parameters(diameter_interp, dist_interp)
                    initial_params=[n, nu, scale]

                    param, pcov = curve_fit(mixture_gamma, diameter_interp, dist_interp, 
                                            p0=initial_params)

                elif peaks.shape[0] == 1:
                    # display = True
                    n, nu, scale, x, y = calculate_gamma_parameters(diameter_interp, 
                                                                    dist_interp, 
                                                                    peak_info)
                    initial_params=[n, nu, scale]
                    
                    param, pcov = curve_fit(mixture_gamma, x, y, 
                                            p0=initial_params)
                    
                    fnew = mixture_gamma(x, *param)
                    residual  = y - fnew

                    peaks_res, peak_res_info = find_peaks(residual, height=5, prominence=10, width=5/delta_diameter)
                    # WIDTH: The width parameter can be used to specify the minimum width of peaks. 
                    # It sets the minimum number of data points between the left 
                    # and right bases of a peak. If a peak is narrower than this width, 
                    # it will not be considered a peak.
                    
                    if peaks_res.any():
                        n, nu, scale, _, _ = calculate_gamma_parameters(x, y, peak_res_info)
                        initial_params=[param[0], n, param[1], nu, param[2], scale]
                        n_param = len(initial_params)
                        # bounds = (0, n_param*[np.inf])
                        # print(initial_params)
                        param_aux, pcov_aux = curve_fit(mixture_gamma, x, y, 
                                            p0=initial_params, maxfev=2000)
                        
                        frac = param_aux[1]/param_aux[0]
                        # param_aux[2] > 5
                        # param_aux[3] > 5

                        if param_aux[1] > 0 and param_aux[2] > 5 and param_aux[3] > 5 and frac > .1 and frac < 2:
                            param = param_aux
                            pcov  = pcov_aux
                
                elif peaks.shape[0] > 1:
                    # display = True
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
                    plot_and_show_distributions(param, diameter, filtered_dist, diameter_interp, dist_interp, fnew, residual, get_params)
                    if count_occurence > 40:
                        set_trace()
                        plt.close('all')
                        count_occurence = 0
                    display = False
                
                # if display:
                #     plot_and_show_distributions(param, diameter, filtered_dist, diameter_interp, dist_interp, fnew, residual, distance_mode)
                #     display = False
                
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

with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(17, 9))
    f0 = sns.histplot(flattened_diameter_mean, stat='count',
                            binwidth=1, 
                            kde=True, 
                            color='red', 
                            alpha=0.4,
                            linewidth=1.2, 
                            element='bars',
                            ax=axs[0])
    axs[0].set_xlabel('Mean Diameter [um]')
    axs[0].set_ylabel('Count')

    f1 = sns.histplot(flattened_diameter_eff, stat='count',
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
    fig.suptitle(r'N$_{dist}$ = %d, N$_{used}$ = %d'%(count_distributions, count_used_distributions))
    fig.savefig(PATH_FIG+'histogram_mean_eff_diameters.png', dpi=300)
    plt.tight_layout()
    plt.show()

    # find_modes_with_gmm(flattened_diameter_mean, 2, nbin=40, title='Histogram of Mean Diameters')
    # find_modes_with_gmm(flattened_diameter_eff, 2, nbin=40, title='Histogram of Effective Diameters')


# Plot the second histogram
    fig, axs = plt.subplots(1, 2, figsize=(17, 9))
    f0 = sns.histplot(flattened_shapes, stat='count',
                        binwidth=2, 
                        kde=True, 
                        color='magenta', 
                        alpha=0.4, 
                        linewidth=1.2, 
                        element='bars',
                        ax=axs[0])
    axs[0].set(xlabel=r'$\nu$', ylabel='Count')
    axs[0].set_title('Histogram of Shapes')
    plt.show()
    # Plot the third histogram
    f2 = sns.histplot(flattened_scales, stat='count', 
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
    fig.suptitle(r'N$_{dist}$ = %d, N$_{used}$ = %d'%(count_distributions, count_used_distributions))
    fig.savefig(PATH_FIG+'histogram_shapes_scales.png', dpi=300)
    plt.tight_layout()
    plt.show()
    # find_modes_with_gmm(flattened_shapes, 1, nbin=30, title='Histogram of Shapes')
    # find_modes_with_gmm(flattened_scales, 2, nbin=40, title='Histogram of Scales')