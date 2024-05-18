import numpy as np
import xarray as xr

from phd_clouds.utils import find_local_maxima, fit_gaussian,\
        get_smoothed_for_weight, fit_gaussian_bimode, fit_gaussian_triple, \
        gaussian_distribution, calculate_r_squared, get_lv0_gfatpy, estimate_baseline
from phd_clouds.constants import CLASSIFICATION_TICK_LABELS, CLASSIFICATION_COLORS, CLASSIFICATION_TICK_LABELS, GRANADA_ALTITUDE 
import time
from tabulate import tabulate
from pdb import set_trace
from rpgpy import rpg2nc as rpg2nc_rpgpy
from gfatpy.radar.rpg_nc import rpg as gfp_radarnc
from gfatpy.radar.retrieve.retrieve import (
    add_all_products_from_LV0,
    add_all_products_from_LV1,
    retrieve_dBZe,
)
# from gfatpy.radar.rpg_binary import rpg as gfp_radarbin
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cloudnetpy.plotting import generate_figure
import datetime
import glob
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('matplotlib', 'inline')

script_dir = '/home/matheustolen/Documentos/matheus_doctorado/phd-clouds/phd_clouds/scripts'
os.chdir(script_dir)
PATH_FIG          = '../figures/'
PATH_CLOUDNET_DER = '../../../output_retrievals'
PATH_CLASS        = '/media/matheustolen/Seagate Basic/cloudnet/classification'
PATH_CAT          = '/media/matheustolen/Seagate Basic/cloudnet/categorize'
classifications_cloudnet = CLASSIFICATION_TICK_LABELS[:-1]

classification_cmap   = plt.cm.colors.ListedColormap(CLASSIFICATION_COLORS)

fontsize = 14
# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize

plt.ion()
plt.close('all')

def weight_finding_algorithm(time_guess,
                            raw_dataset,
                            class_dataset,
                            ranges):
    
    """
    Finds weights based on the given parameters.

    Args:
        time_guess (str): Time in the format 'YYYYMMDDTHHMMSS.S'.
        raw_dataset (xr.Dataset): Raw LV0 radar dataset.
        class_dataset (xr.Dataset): Classification dataset.
        ranges (numpy.ndarray): Array of range values.

    Returns:
        maxima_arr (list): List of maximum points.
        minima_arr (list): List of minimum points.
        hydrometeor_ratios (list): List of hydrometeor ratios.
        hydrometeor_errors (list): List of hydrometeor errors.
        hydrometeor_scales (list): List of hydrometeor scales.
        r_squares (list): List of R-squared values.
        rsses (list): List of RSS values.
    """
    # fig, ax1 = plt.subplots(1, figsize=[9, 6])
    # colors = color_list(len(ranges))
    fig_1, axes = plt.subplots(int(len(ranges)), 2, figsize=[30, 6*int(len(ranges))])

    print('start:')

    maxima_arr = []
    minima_arr = []
    hydrometeor_ratios = []
    hydrometeor_errors = []
    hydrometeor_scales = []
    r_squares = []
    rsses = []
    sigmas = []


    #Abritrary parameters
    initial_params_single = [420, 1, 1, 0]  
    initial_params_bimode = [300, 2, 1, 420, 1, 1, 0]  
    initial_params_triple = [0, 10, 1, 0, 10, 1, 0, 10, 1, 0]

    
    table_data = []
    headers = ["Height", "Cloudnet Classification", "Type", "Ratio 1", "Ratio 2", "r2", "RSS", "runtime"]

    max_err = 0
    start_time = time.time()
    
    list_doopler_spectrum = []
    list_fits = []
    
    for index, height in enumerate(ranges):
        print_data = []
        print_data.append(height)
        
        class_ = classifications_cloudnet[int(class_dataset['target_classification'].sel(time = time_guess, height = height, method = 'nearest').values)]
        
        print_data.append(class_)

        
        #Smooth doppler spectrum
        a = get_smoothed_for_weight(dataset=raw_dataset, time=time_guess, height=height)
        
        indx_aux = np.where(height < 1200, 0, np.where(height > 5500, 2, 1))
        list_doopler_spectrum.append([raw_dataset['velocity_vectors'][indx_aux], a, height])

        
        #Find max points
        local_max, local_min, minlev = find_local_maxima(a)


        maxima_arr.append(local_max)
        minima_arr.append(local_min)

        #Collect data and normalize y_data
        x_data = raw_dataset['spectrum']
        y_data_raw = a
        y_data = a / np.sum(a)

        size = len(local_max)

        axes[index][0].set_title(int(height),fontsize=16)
        axes[index][0].set_xlabel('spectrum')
        axes[index][0].set_ylabel('Intensity [Ze]')

        if size == 1:

            #Guess values and bounds for fitting. 
            initial_params_single[0] = local_max[0]
            lower_bounds = [local_max[0] - 10, 0, 0, 0]
            upper_bounds = [local_max[0] + 10, np.inf, np.inf, .0005]
            
            # Adjust code so mu is given and not fitted to a small range? might give slight improvements
            fit_params, fit_data, fit_err = fit_gaussian(x_data, y_data, initial_params_single, 1600, bounds = (lower_bounds, upper_bounds))
            
            print_data.append('single')
            axes[index][0].plot(x_data, y_data, c = 'r', label = 'smoothed')
            axes[index][0].plot(x_data, fit_data, color = 'b', label = 'fit')
            axes[index][0].legend(fontsize=16)
            
            hydrometeor_ratios.append(1)


            #Error Calculations:
            r_err =  y_data - fit_data
            hydrometeor_errors.append(r_err)

            axes[index][1].scatter(raw_dataset['spectrum'], r_err)
            axes[index][1].axhline(0, c='black', linestyle = '--') 

            print_data.append('---')
            print_data.append('---')

            
            r_2, rss = calculate_r_squared(y_data, fit_data)
            
            print_data.append("{:.4e}".format(r_2))
            print_data.append("{:.4e}".format(rss))

            r_squares.append(r_2)
            rsses.append(rss)
            sigmas.append([fit_params[1]])

        if size == 2:

            print_data.append('double')

            initial_params_bimode[0] = local_max[0]
            initial_params_bimode[3] = local_max[1]
            lower_bounds = [initial_params_bimode[0] - 10, 0, 0, initial_params_bimode[3] - 10, 0, 0, 0]
            upper_bounds = [initial_params_bimode[0] + 10, np.inf, np.inf, initial_params_bimode[3] + 10, np.inf, np.inf, .0005]

            fit_params, fit_data, fit_err, params_covariance = fit_gaussian_bimode(x_data, y_data, initial_params_bimode, 1600, bounds = (lower_bounds, upper_bounds))
            
            initial_params_bimode = fit_params
            axes[index][0].plot(x_data, (y_data), 'r', label = 'smoothed')
            axes[index][0].plot(x_data, (fit_data), 'b', label = 'fit')

            #calculate the individual modes
            mode_1 = gaussian_distribution(x_data, fit_params[0], fit_params[1],fit_params[2])
            mode_2 = gaussian_distribution(x_data, fit_params[3], fit_params[4],fit_params[5])

            axes[index][0].plot(x_data, (mode_1), 'cyan', label = 'hydrometeor 1')
            axes[index][0].plot(x_data, (mode_2), 'olive', label = 'hydrometeor 2')

            weight_1 = np.sum(mode_1)
            weight_2 = np.sum(mode_2)
            ratio = float(weight_1/(weight_1+ weight_2))

            hydrometeor_ratios.append(ratio)
            axes[index][0].legend(fontsize=16)

            print_data.append("{:.4f}".format(ratio))

            #Error calculations        
            r_err = y_data - fit_data
            hydrometeor_errors.append(r_err)

            axes[index][1].scatter(raw_dataset['spectrum'], r_err)
            axes[index][1].axhline(0, c='black', linestyle = '--')

            print_data.append('---')

            r_2, rss = calculate_r_squared(y_data, fit_data)
            #Error prints
            print_data.append("{:.4e}".format(r_2))
            print_data.append("{:.4e}".format(rss))

            r_squares.append(r_2)
            rsses.append(rss)
            sigmas.append([fit_params[1], fit_params[4]])


        if size == 3:
            print_data.append('triple')

            initial_params_triple[0] = local_max[0]
            initial_params_triple[3] = local_max[1]
            initial_params_triple[6] = local_max[2]

            lower_bounds = [
                initial_params_triple[0] - 10, 3, 0,
                initial_params_triple[3] - 10, 3, 0,
                initial_params_triple[6] - 10, 3, 0,
                0
            ]
            upper_bounds = [
                initial_params_triple[0] + 10, 100, np.inf,
                initial_params_triple[3] + 10, 100, np.inf,
                initial_params_triple[6] + 10, 100, np.inf,
                0.0005
            ]
            
            fit_params, fit_data, fit_err, params_covariance = fit_gaussian_triple(x_data, y_data, initial_params_triple, 1600, bounds = (lower_bounds, upper_bounds))
            initial_params_trimode = fit_params

            mode_1 = gaussian_distribution(x_data, fit_params[0], fit_params[1], fit_params[2])
            mode_2 = gaussian_distribution(x_data, fit_params[3], fit_params[4], fit_params[5])
            mode_3 = gaussian_distribution(x_data, fit_params[6], fit_params[7], fit_params[8])

            axes[index][0].plot(x_data, (y_data), 'r', label = 'smoothed')
            axes[index][0].plot(x_data, (mode_1), 'cyan', label = 'hydrometeor 1')
            axes[index][0].plot(x_data, (mode_2), 'olive', label = 'hydrometeor 2')
            axes[index][0].plot(x_data, (mode_3), 'plum', label = 'hydrometeor 3')
            axes[index][0].legend(fontsize=16)

            weight_1 = np.sum(mode_1)
            weight_2 = np.sum(mode_2)
            weight_3 = np.sum(mode_3)
            ratio1 = float(weight_1/(weight_1+ weight_2+weight_3))
            ratio2 = float(weight_2/(weight_1+ weight_2+weight_3))

            hydrometeor_ratios.append((ratio1,ratio1+ ratio2))

            print_data.append("{:.4f}".format(ratio1))
            print_data.append("{:.4f}".format(ratio2))

            #Error Calculations
            r_err = y_data- fit_data
            hydrometeor_errors.append(r_err)

            axes[index][1].scatter(raw_dataset['spectrum'], r_err)
            axes[index][1].axhline(0, c='black', linestyle = '--')

            r_2, rss = calculate_r_squared(y_data, fit_data)
            
            print_data.append("{:.4e}".format(r_2))
            print_data.append("{:.4e}".format(rss))
            
            r_squares.append(r_2)
            rsses.append(rss)
            sigmas.append([fit_params[1], fit_params[4], fit_params[7]])

        if size > 3:
            #No good system for more than 3 peaks yet. 
            hydrometeor_errors.append(0)
            axes[index][0].plot(x_data, (y_data), 'b', label = 'smoothed')
            print_data.append('error - too many peaks')
            r_2, rss = calculate_r_squared(y_data, y_data)
        
        list_fits.append([fit_data, y_data])

        start_time = time.time()
        table_data.append(print_data)

        max_err = max(max_err, np.max(np.abs(r_err)))
        axes[index][1].legend([f'r - {"{:.4e}".format(rss)}', f'r^2 - {"{:.4e}".format(r_2)}'], fontsize = 20)


    # for i in range(axes.shape[0]):
    #     axes[i, 1].set_ylim(-max_err, max_err)

    table = tabulate(table_data, headers=headers)
    print(table)
    # fig.savefig(PATH_FIG + 'weight_finding.png', dpi = 300)
    # fig_1.savefig(PATH_FIG + 'weight_finding_smoothed.png', dpi = 300)
    # plt.show()
    
    return maxima_arr, minima_arr, hydrometeor_ratios, hydrometeor_errors, hydrometeor_scales,\
          r_squares, rsses, sigmas, table_data, list_doopler_spectrum, list_fits,


# def weight_finding_results(raw_dataset, num_colors, ranges, maxima_arr, minima_arr, hydrometeor_ratios, 
#                            hydrometeor_errors, hydrometeor_scales, r_squares, rsses, sigmas, table_data_):
#     """
#     Generates plots and visualizations based on the weight finding results.

#     Args:
#         maxima_arr (list): List of maximum points.
#         minima_arr (list): List of minimum points.
#         hydrometeor_ratios (list): List of hydrometeor ratios.
#         hydrometeor_errors (list): List of hydrometeor errors.
#         hydrometeor_scales (list): List of hydrometeor scales.
#         r_squares (list): List of R-squared values.
#         rsses (list): List of RSS values.
#         sigmas (list): List of std deviations
#         table_data_ (array): Unused array. Put in so inputs are equal to outputs of weight_finding_algorithm 

#     Returns:
#         None
#     """
#     fig_1, axes = plt.subplots(1,3, figsize=[16, 10])
#     fig, axes_errors = plt.subplots(1,2, figsize=[16, 10])

#     #Ratios:
#     #Makes the arrays the correct size for plotting multiple values for the same height
#     for ratios, value in zip(hydrometeor_ratios, ranges):
#         if isinstance(ratios, (float, int)):
#             x_vals = [ratios]
#             y_vals = [value]
#         else:
#             x_vals = [ratios] 
#             y_vals = [value]* len(ratios)
#         axes[0].scatter(x_vals, y_vals, c = 'teal')

#     axes[0].set_xlim([0, 1.1])

#     axes[0].set_xlabel('hydrometeor percentage')
#     axes[0].set_ylabel('height [m]')
#     axes[0].set_title('a) hydrometeor - height profile', fontsize=16)

#     velocity_vectors = raw_dataset['velocity_vectors']
    
#     # Plot the maxima and minima as scatter points
#     for i in range(len(ranges)):
#         maxima = maxima_arr[i]
#         minima = minima_arr[i]
#         velocity_vector = velocity_vectors[0] if ranges[i] < 1200 else (
#             velocity_vectors[1] if 1200 < ranges[i] < 5500 else velocity_vectors[2]
#         )
#         red = axes[1].scatter(velocity_vector[maxima], [ranges[i]] * len(maxima), color='r', marker='|',s = 40)

#         if len(minima) > 0:
#             blue = axes[1].scatter(velocity_vector[minima], [ranges[i]] * len(minima), color='b', marker='|', s = 40)

            
#     axes[1].set_xlabel('Velocity [m/s]')
#     axes[1].set_title('b) Maxima and Minima Profile', fontsize=16)
#     axes[1].legend([red, blue], ['Maxima', 'Minima'])
    
#     range_coords = []
#     max_coords = []
#     sigma_coords = []
#     const_width = np.sqrt(2*np.log(2))
    


#     #Full width half max error bars:
#     # Iterate over each range value and its corresponding maxima and sigma values
#     for i, (maxima_values, sigma) in enumerate(zip(maxima_arr, sigmas)):

#         range_coords.extend([ranges[i]] * len(maxima_values))
#         max_coords.extend(maxima_values)
#         sigma_coords.extend(sigma)

#     cmap = plt.cm.get_cmap('Set1')
#     color_values = np.arange(num_colors)
#     colors = cmap(color_values)

#     # Plot the scatter points with adjusted colors

#     sigma_coords = np.array(sigma_coords)
    
   
    
#     vel_vec_plot = []
#     nyquist_vals = [10.53, 6.49, 4.53]
    
#     #convert error bars from spectrum to velocity
#     for idx, (height, wavelength) in enumerate(zip(range_coords, max_coords)):
#         vv_idx = 0 if height < 1200 else 1 if 1200 < height < 5500 else 2
        
#         velocity_vector = velocity_vectors[vv_idx]
#         vel_vec_plot.append(velocity_vector[wavelength])
#         sigma_coords[idx] = sigma_coords[idx] * nyquist_vals[vv_idx]/512
    

    
#     axes[2].errorbar(vel_vec_plot, range_coords, xerr=sigma_coords*const_width, fmt='o', ecolor = 'b', capsize=10, elinewidth=0.5, markeredgewidth=0.5,  alpha=0.5,c = 'r')
#     axes[2].set_title('c) Full Width at Half Maximum profile', fontsize=16)
#     axes[2].set_xlabel('Velocity [m/s]')
            
            

#     axes_errors[0].scatter(r_squares, ranges)
#     axes_errors[0].set_title('fit error')
#     axes_errors[0].set_xlabel('r-squared value')
    
#     axes_errors[1].scatter(rsses, ranges)
#     axes_errors[1].set_title('fit error')
#     axes_errors[1].set_xlabel('RSS value')

#     for ax in axes:
#         ax.grid(axis='y', linestyle='dotted')
#         ax.grid(axis='x', linestyle='dotted')

#     fig, error_plot = plt.subplots(1, figsize=[6, 6])
#     error_plot.scatter(rsses, r_squares)
#     error_plot.set_xlabel('R Squared')
#     error_plot.set_ylabel('Residual sum of squares')
#     error_plot.set_title('R^2 vs RSS Plot')
    
# #     fig_1.savefig('weight_plots', dpi = 600)
#     plt.tight_layout()
#     plt.show()

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Main code
# -------------------------------------------------------------------------------------------------------------------------------------------------
instrument_name   = 'nephele'

date = datetime.datetime.strptime('2021-09-13', '%Y-%m-%d') # el codigo de cris funciona para chirps en esta fecha
# date = datetime.datetime.strptime('2022-11-16', '%Y-%m-%d')


filepath_classification = f"{PATH_CLASS}/{date.strftime('%Y%m%d')}_granada_classification.nc"
filepath_categorize     = f"{PATH_CAT}/{date.strftime('%Y%m%d')}_granada_categorize.nc"
filepath_der            = f"{PATH_CLOUDNET_DER}/{date.strftime('%Y%m%d')}_granada_der.nc"
filepath_output         = '/media/matheustolen/Seagate Basic/cloudnet/cloud_reflectivity_sp_analysis'

#-------------------------------------------------------------------------------------------------------------------------------------------------
ds_classification = xr.open_dataset(filepath_classification) # used by the code
ds_categorize     = xr.open_dataset(filepath_categorize) # not used by the code
ds_der            = xr.open_dataset(filepath_der) # not used by the code

#-------------------------------------------------------------------------------------------------------------------------------------------------
# time_series    = ds_der.time.to_series()
# new_start_time = time_series.min().replace(hour=0, minute=0, second=15, microsecond=0)
# new_end_time   = time_series.max().replace(hour=23, minute=59, second=59, microsecond=0)
# new_time_index = pd.date_range(start=new_start_time, end=new_end_time, freq="30S")
#-------------------------------------------------------------------------------------------------------------------------------------------------
ds_diff_der = (ds_der['der_scaled'] - ds_der['der'])*1e6
ds_diff_der = ds_diff_der.rename('diff_der')
ds_diff_der.attrs['long_name'] = 'difference between scaled and not scaled droplet re'
ds_diff_der.attrs['units'] = r'$\mu$m'

mask_high_diff = (ds_diff_der > 60) # pixels with more than 10 um of difference
time_mask      = np.max(mask_high_diff, axis=1)
time_high_diff = ds_diff_der.time[time_mask]

dic_range_for_time = {}
for t in time_high_diff:
    dic_range_for_time[t.values] = ds_diff_der['height'][mask_high_diff.sel(time=t).values].values

#-------------------------------------------------------------------------------------------------------------------------------------------------
time_guess_chriss = np.datetime64('2021-09-13T11:30:00') ##CHANGE THIS
# time_to_study = time_high_diff[0].values
time_to_study = time_guess_chriss
ti_datetime = pd.to_datetime(time_to_study)
time_guess = ti_datetime.strftime('%Y%m%dT%H%M%S.%f')[:-3]
# ranges_to_study = dic_range_for_time[time_to_study] ##########CHANGE THIS
ranges_to_study = ranges = np.arange(2000, 3000, 60)

#-------------------------------------------------------------------------------------------------------------------------------------------------
pattern               = f"{ti_datetime.strftime('%y%m%d_%H')}*.LV0"
directorypath_rawdata = f"/home/matheustolen/shared/NAS_raw_data/UGR/{instrument_name}/{ti_datetime.strftime('%Y/%m/%d')}"

filepath_rawdata = glob.glob(os.path.join(directorypath_rawdata, pattern))
raw_filename     = os.path.basename(filepath_rawdata[0]).replace('LV0', 'LV0.nc')
filepath_raw_netcdf = os.path.join(filepath_output, raw_filename)
# check if raw netcdf file exists
if not os.path.exists(filepath_raw_netcdf):
    rpg2nc_rpgpy(filepath_rawdata[0], output_file=filepath_raw_netcdf)
    # rpg2nc_rpgpy(filepath_rawdata[0]+"/*LV0", output_file=filepath_output+"/huge_day_file.nc") # If we want to concatenate one day file
    print(f"File created at {filepath_raw_netcdf}")
else:
    print(f"File {filepath_raw_netcdf} already exists")

ds_radar_raw      = xr.open_dataset(filepath_raw_netcdf)
#-------------------------------------------------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, .02], hspace=0.08, wspace=0.01)

# Plot the difference between scaled and not scaled droplet re
ax1 = fig.add_subplot(gs[0,0])
pc = ax1.pcolormesh(ds_diff_der.time, ds_diff_der.height, ds_diff_der.T, shading='auto', vmin=0, vmax=ds_diff_der.max(), cmap='jet')
ax1.set_title(ds_diff_der.attrs['long_name'])
ax1.set_ylabel('Height [m]')
ax1.xaxis.set_tick_params(labelbottom=False)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

cax1 = fig.add_subplot(gs[0,1])
cbar1 = plt.colorbar(pc, cax=cax1, label=ds_diff_der.attrs['units'])

# Plot the classification
ax2 = fig.add_subplot(gs[1,0], sharex=ax1, sharey=ax1)
pc2 = ax2.pcolormesh(ds_classification.time, ds_classification.height, ds_classification.target_classification.T,
                     shading='auto', cmap=classification_cmap, vmin=0, vmax=len(CLASSIFICATION_TICK_LABELS))

pc2 = ax2.pcolormesh(ds_categorize.time, ds_classification.height, ds_classification.target_classification.T,
                     shading='auto', cmap=classification_cmap, vmin=0, vmax=len(CLASSIFICATION_TICK_LABELS))
ax2.set_xlabel('Time')
ax2.set_ylabel('Height [m]')
ax2.set_xlim(time_to_study - pd.Timedelta(minutes=1), time_to_study + pd.Timedelta(minutes=1))
ax2.set_ylim(ranges_to_study[0]-50, ranges_to_study[-1]+50)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

cax2 = fig.add_subplot(gs[1,1])
cbar2 = plt.colorbar(pc2, cax=cax2,  ticks=[], orientation='vertical')

for idx, (color, name) in enumerate(zip(classification_cmap.colors, CLASSIFICATION_TICK_LABELS)):
    rect = plt.Rectangle((0, idx), 1, 1, color=color)
    cbar2.ax.add_patch(rect)
    cbar2.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=14)
plt.show()

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Chris results to different mode identification and fitting
# -------------------------------------------------------------------------------------------------------------------------------------------------
# results = weight_finding_algorithm(time_guess, ds_radar_raw, ds_classification, ranges_to_study)
# weight_finding_results(ds_radar_raw, num_colors=len(x), ranges=ranges, maxima_arr=x[0], minima_arr=x[1], hydrometeor_ratios=x[2],
#                           hydrometeor_errors=x[3], hydrometeor_scales=x[4], r_squares=x[5], rsses=x[6], sigmas=x[7], table_data_=x[8])
# -------------------------------------------------------------------------------------------------------------------------------------------------

path_lv0nc  = Path(filepath_raw_netcdf)
gpy_radarnc = gfp_radarnc(path_lv0nc)

assert gpy_radarnc.level == 0
assert gpy_radarnc.type == 'ZEN'

# gpy_radarnc._data = gpy_radarnc._data.resample(time='30S').mean()
nearest_time_raw = gpy_radarnc.data.time.sel(time=time_to_study, method='nearest').values

if len(ranges_to_study) > 1:
    fig, filepath =  gpy_radarnc.plot_spectra_by_range(target_time=time_to_study, 
                                                   range_slice=ranges_to_study.tolist(),
                                                   **{"savefig": False, "velocity_limits": (-8, 0)}
                                                   )
else:
    fig, filepath = gpy_radarnc.plot_spectra_by_time(target_range=ranges_to_study[0],
                                                     time_slice=(nearest_time_raw-pd.Timedelta(minutes=2), nearest_time_raw+pd.Timedelta(minutes=2)),
                                                     **{"savefig": False}
                                                     )
fig, filepath = gpy_radarnc.plot_2D_spectrum(
        target_time=time_to_study,
        range_limits=(ranges_to_study[0], ranges_to_study[-1]),
        vmin=-1,
        vmax=1,
        **{"savefig": False}
    )
# get axis from fig
# ax = fig.get_axes()
# for axis in ax:
#     axis.set_xlim(-2.5, 2.5)
# fig.show()

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Example
# -------------------------------------------------------------------------------------------------------------------------------------------------
ds_pixel = gpy_radarnc.data.sel(time=time_to_study, range=ranges_to_study[2], method='nearest')

mask_finite = np.isfinite(ds_pixel["doppler_spectrum"].values)

# Convert to dBZe
ds_pixel["doppler_spectrum_dBZe"] = retrieve_dBZe(ds_pixel["doppler_spectrum"], gpy_radarnc.band)
ds_pixel["doppler_spectrum_dBZe"].attrs = {
            "long_name": "Power density",
            "units": "dB",
        }

spectrum = ds_pixel["doppler_spectrum_dBZe"].values[mask_finite]
velocity = ds_pixel["velocity_vectors"].sel(chirp=ds_pixel["chirp_number"]).values[mask_finite]


smoothed_spectrum = gaussian_filter1d(spectrum, 3)
baseline          = estimate_baseline(smoothed_spectrum, 1)
peaks, peak_info  = find_peaks(smoothed_spectrum,prominence=10)
baseline          = estimate_baseline(smoothed_spectrum, 1)
peak_values       = smoothed_spectrum[peaks]
peak_velocities   = velocity[peaks]
print(peak_info)

fig, ax = plt.subplots(1, figsize=[9, 6])
ax.plot(velocity, spectrum)
ax.plot(velocity, smoothed_spectrum, "-k", linewidth=2)
ax.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
for peak_value, peak_velocity in zip(peak_values, peak_velocities):
    ax.axvline(x=peak_velocity, color='g', linestyle='--', label=f'Peak: {peak_value:.2f} dB')
ax.set_xlabel("Velocity [m/s]")
ax.set_ylabel("Power density [dB]")
ax.set_title("Doppler spectrum")
ax.legend()
plt.show()
# # -------------------------------------------------------------------------------------------------------------------------------------------------

# # -------------------------------------------------------------------------------------------------------------------------------------------------
# gpy_radar = get_lv0_gfatpy(ti_datetime, instrument_name, filepath_output)

# # -------------------------------------------------------------------------------------------------------------------------------------------------
# slice_time_radarnc    = (gpy_radar.data.time[0].values, gpy_radar.data.time[-1].values)
# time_section_cloudnet = ds_classification.time.sel(time=slice(*slice_time_radarnc)).values
# # slice_time_cloudnet   = (time_section_cloudnet[0].values, time_section_cloudnet[-1].values)
# range_cloudnet        = ds_classification.height.values - GRANADA_ALTITUDE

# # Initialize an empty xarray dataset with the same dimensions as gpy_radar.data
# ds_modes = xr.Dataset(
#     {
#         "target_mode": (("time", "height"), np.zeros((len(ds_classification.time), len(ds_classification.height)))),
#     },
#     coords={"time": ds_classification.time, "height": ds_classification.height-GRANADA_ALTITUDE},
# )

# for time_cloudnet in time_section_cloudnet:
#     closest_time_radarnc = gpy_radar.data.time.sel(time=time_cloudnet, method='nearest').values
#     rpg_radar            = get_lv0_gfatpy(time_cloudnet, instrument_name, filepath_output)
#     sdata                = rpg_radar.data.sel(time=closest_time_radarnc)
#     sdata["doppler_spectrum_dBZe"] = retrieve_dBZe(sdata["doppler_spectrum"], rpg_radar.band)
#     for i, z in enumerate(range_cloudnet):
#         closest_range_radarnc = sdata["range"].sel(range=z, method='nearest').values
#         spectrum              = sdata["doppler_spectrum_dBZe"].sel(range=closest_range_radarnc).values
#         mask_spectrum          = np.isfinite(spectrum)
#         if np.any(mask_spectrum):
#             vel                   = sdata["velocity_vectors"].sel(chirp=sdata["chirp_number"].values[i]).values[mask_spectrum]
#             smoothed_spectrum      = gaussian_filter1d(spectrum[mask_spectrum], 3)
#             baseline               = estimate_baseline(smoothed_spectrum, 1)
#             peaks, peak_info       = find_peaks(smoothed_spectrum, prominence=3)
#             n_modes               = len(peaks)
#             ds_modes["target_mode"].loc[{"time": time_cloudnet, "height": z}] = n_modes

# sliced_modes = ds_modes.sel(time=slice(*slice_time_radarnc))
# print(np.sum(sliced_modes['target_mode'].values==2))