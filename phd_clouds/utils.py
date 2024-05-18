#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
various utilities
"""
import os
import datetime
import pandas as pd
import requests
from pdb import set_trace
from scipy import ndimage
import numpy as np
import re
from dateutil.relativedelta import relativedelta
import xarray as xr
import matplotlib.pylab as plab
from scipy.optimize import curve_fit
from scipy.stats import gamma
from scipy import integrate
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from gfatpy.radar.rpg_nc import rpg as gfp_radarnc
from pathlib import Path
import glob
from rpgpy import rpg2nc as rpg2nc_rpgpy

def groupSequence(lst, cloud_bins=0):
    # Initialize a result list with the first element of the input list.
    res = [[lst[0]]]

    # Loop through the input list starting from the second element.
    for i in range(1, len(lst)):
        # Check if the current element is consecutive to the previous element.
        if lst[i - 1] + 1 == lst[i]:
            # If consecutive, add the current element to the last sub-list in 'res'.
            res[-1].append(lst[i])
        else:
            # If not consecutive, create a new sub-list in 'res' with the current element.
            res.append([lst[i]])

    # Filter out sub-lists with length less than 'cloud_bins'.
    filtered_res = [sublist for sublist in res if len(sublist) >= cloud_bins]

    # Return the final list of grouped sequences with minimum length 'cloud_bins'.
    return filtered_res

def common_prefix_of_filenames(paths, extension):
    # Get the filenames from each path with the given extension
    filenames_per_path = []
    for path in paths:
        filenames = [file[:8] for file in os.listdir(path) if file.endswith(extension)]
        filenames_per_path.append(set(filenames))

    # Find the common prefixes among the filenames in both paths
    common_prefixes = set.intersection(*filenames_per_path)

    # Convert common prefixes to datetime objects
    datetime_prefixes = []
    for prefix in common_prefixes:
        try:
            datetime_obj = datetime.datetime.strptime(prefix, '%Y%m%d')
            datetime_prefixes.append(datetime_obj)
        except ValueError:
            # If a prefix is not in the expected format, it will be skipped
            pass

    datetime_prefixes.sort()

    return pd.DatetimeIndex(datetime_prefixes)

def extract_date_from_radar_filename(filename):
    pattern = r'(\d{2})(\d{2})(\d{2})'
    match = re.search(pattern, filename)
    if match:
        year = '20' + match.group(1)
        month = match.group(2)
        day = match.group(3)
    else:
        year = None
        month = None
        day = None
    return year, month, day

def download_cloudnet_products(date_ini, date_end, path_output, product='model', site='granada'):
    
    # Define the folder path where you want to save the downloaded files
    # Download the files from the cloudnetpy API
    url = 'https://cloudnet.fmi.fi/api/files'
    payload = {
        'product': product,
        'site': site,
        'dateFrom': date_ini,
        'dateTo': date_end
    }
    metadata = requests.get(url, params=payload).json()

    # Ensure the output folder exists; create it if not.
    os.makedirs(path_output, exist_ok=True)
    
    for row in metadata:
        res = requests.get(row['downloadUrl'])
        
        # Create the full file path by joining the output folder and the filename
        file_path = os.path.join(path_output, row['filename'])
        
        with open(file_path, 'wb') as f:
            f.write(res.content)

def get_complete_time(chunked_dataset, chunk_size: int = 1000, freq_index="30S", start_month=None):
    time_series = chunked_dataset.indexes['time'] # Convert to pandas DateTimeIndex
    if start_month is not None:
        new_start_time = time_series.min().replace(day=1, hour=0, minute=0, second=15, microsecond=0, month=1)
    else:
        # Calculate the new start time as the first 15 seconds of the day
        new_start_time = time_series.min().replace(day=1, hour=0, minute=0, second=15, microsecond=0)
    
    new_end_time = time_series.max().replace(day=1, hour=23, minute=59, second=59, microsecond=0) + relativedelta(day=31)

    # Create a new time index starting from the new_start_time and ending at the end of the day
    new_time_index = pd.date_range(start=new_start_time, end=new_end_time, freq=freq_index)

    return new_time_index

def assign_season(ds, seasons):
    """
    Assign season labels to a given xarray dataset based on specified season ranges.

    Parameters:
    ds (xarray.Dataset): The input xarray dataset with a 'time' dimension.
    seasons (dict): A dictionary that defines the season ranges in terms of months.

    Returns:
    xarray.Dataset: A new dataset with an additional 'season' coordinate.
    """

    # Create a mask for each season
    conditions = []
    for name, (start, end) in seasons.items():
        if name == 'winter':
            # Handle the wrap-around between December and January
            cond = (ds['time.month'] >= start) | (ds['time.month'] <= end)
        else:
            cond = (ds['time.month'] >= start) & (ds['time.month'] <= end)
        conditions.append(cond)

    # List of corresponding season names
    season_names = list(seasons.keys())

    # Use np.select to assign season names based on conditions
    season_data = xr.DataArray(
        np.select(conditions, season_names, default='unknown'),
        dims=('time',)
    )

    ds = ds.assign_coords(season=season_data)

    return ds

def convToDatetime(sec):
    origin = pd.Timestamp('2001-01-01')
    delta = pd.to_timedelta(sec, unit = 's')
    return origin + delta


def convIndexToDatetime(index, dataset):
    """
    Converts an index in a LV0 dataset to the corresponding datetime value.

    Args:
        index (int): The index of the time value in the dataset.
        dataset (pandas.DataFrame): The LV0 dataset containing time values.

    Returns:
        pandas.Timestamp: The datetime value corresponding to the index in the dataset.
    """

    origin = pd.Timestamp('2001-01-01')
    time_timedelta = pd.to_timedelta(dataset.time.values[index], unit='s')

    return origin + time_timedelta


def convert_to_seconds(timestamp):
    """
    Converts a pandas Timestamp to the amount of seconds since 01/01/2001: the unit used in LV0 data time values.

    Args:
        timestamp (str or pandas.Timestamp): The timestamp to convert to seconds.

    Returns:
        float: The number of seconds between the timestamp and 01/01/2001.
    """

    origin = pd.Timestamp('2001-01-01')
    target = pd.Timestamp(timestamp)
    time_difference = target - origin
    seconds = time_difference.total_seconds()
    return seconds

def convert_to_index(time_str, dataset):
    """
    Converts a time string to the index in an LV0 time array.

    Args:
        time_str (str): The time string to convert to an index.
        dataset (pandas.DataFrame): The dataset containing the time array.

    Returns:
        int: The index corresponding to the provided time in the dataset's time array.
    """

    datetime = pd.to_datetime(time_str, format="%Y%m%dT%H%M%S.%f")
    secs = convert_to_seconds(datetime)
    index = np.searchsorted(dataset['time'], secs)

    diff = abs(secs - dataset['time'][index])/60
    if diff > 1:
        print(f"Warning: The time difference is {diff} minutes.")
        print(f"The diferece is {secs - dataset['time'][index]}")

    return index

def convert_to_sec_safe(time_str, dataset):
    """
    Retrieves the closest time value that exists within an LV0 time values array.

    Args:
        time_str (str): The time string to find the closest time value for.
        dataset (pandas.DataFrame): The dataset containing the time values array.

    Returns:
        float: The closest time value to the provided time string in the dataset's time values array.
    """

    idx = convert_to_index(time_str, dataset)
    sec = dataset['time'][idx]
    return sec

def getHeightIdx(target, dataset):
    """
    Retrieves the index of the closest height value within a dataset's range_layers array.

    Args:
        target (float): The target height value to find the closest index for.
        dataset (pandas.DataFrame): The dataset containing the range_layers array.

    Returns:
        int: The index of the closest height value to the provided target in the dataset's range_layers array.
    """

    idx = np.searchsorted(dataset['range_layers'], target)
    return idx

def color_list(n):    
    colors = plab.cm.jet(np.linspace(0,1,n))
    return colors

def calculate_r_squared(observed, predicted):
    """
    Calculate the R-squared (coefficient of determination) for observed and predicted values.
    
    Args:
        observed (array-like): Array or list of observed values.
        predicted (array-like): Array or list of predicted values.
        
    Returns:
        float: R-squared value.
    """
    # Convert the input to numpy arrays
    observed = np.array(observed)
    predicted = np.array(predicted)

    # Calculate the mean of the observed values
    mean_observed = np.mean(observed)

    # Calculate the total sum of squares (TSS)
    tss = np.sum((observed - mean_observed) ** 2)

    # Calculate the residual sum of squares (RSS)
    rss = np.sum((observed - predicted) ** 2)

    # Calculate R-squared
    r_squared = 1 - (rss / tss)

    return r_squared, rss

def estimate_baseline(signal: np.ndarray, window_size: int) -> float:
    """
    Estimate the baseline of a signal using the Rolling Ball algorithm.

    Args:
        signal (np.ndarray): Input signal array.
        window_size (int): Size of the rolling window for baseline estimation.

    Returns:
        float: Estimated baseline of the signal.
    """
    base_signal = np.median(ndimage.filters.minimum_filter(signal, size=window_size, mode='reflect'))
    
    return base_signal

def find_local_maxima(data):
    '''
    Count the number of local max above an empirical threshold determined by multiplier
    
    This function is way more confusing than it should be. It holds data about aliasing but isn't used for that
    Once a function is created that correctly estimates the threshold the code should do this:
    
    For each local max, check if it is above threshold
    Find the lowest value between two peaks. If min(peak1, peak2)  * multiplier > min, count peak2
    Then make peak1 = peak2 and repeat for the next max
    
    
    '''
    minval = np.min(data)
    maxval = np.max(data)

    baseval = estimate_baseline(data, 1)
    diff = baseval - minval
    toplevel = baseval + diff * 200

    if toplevel > maxval:
        toplevel = baseval + diff * 100
        if toplevel > maxval:
            toplevel = baseval + diff * 2
    min_threshold = toplevel
    
    local_maxima_indices = np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]) & (data[1:-1] > min_threshold))[0] + 1
    
    left_maximum_index = 0
    if len(local_maxima_indices) > 0:
        left_maximum_index = local_maxima_indices[0]
        if left_maximum_index < 600:
            hasFalling = True
            
    left_value = data[left_maximum_index]

    local_maxima = [left_maximum_index]
    local_minima = []
    
    for i in local_maxima_indices:
        intermediate_range = np.arange(left_maximum_index, i)
        midpoint_threshold = min(left_value, data[i]) 

        if (data[intermediate_range] < midpoint_threshold).any():
            local_minima.append(left_maximum_index + np.argmin(data[intermediate_range]))
            local_maxima.append(i)
            
            left_maximum_index = i
            left_value = data[i]
            if hasFalling:
                if i > 900:
                    aliasing = True
        else:
            if data[i] > left_value:
                left_maximum_index = i
                left_value = data[i]
                
    #If the count is very high, try once to increase the min threshold
    
    return local_maxima, local_minima, min_threshold

#------------------------------------------------------------------------------------#
# -------------------------- Gaussian Distribution Fitting --------------------------#
#------------------------------------------------------------------------------------#
# NOTE: Bellow we have a set of functions that are used to fit gaussian distributions, 
# but they are dumb. We should create a more inteligent ones like the right bellow examples
def calculate_ssd(array1, array2):
    squared_diff = (array1 - array2) ** 2
    ssd = np.sum(squared_diff)
    return ssd

def gaussian_distribution_fit(x, mu, sigma, scale,b):
    return scale * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + b

def gaussian_distribution(x, mu, sigma, scale):
    return scale * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) 
    
def gaussian_distribution_combo(x, mu_1, sigma_1, scale_1, mu_2, sigma_2, scale_2, b):
    normal_1 = scale_1*np.exp(-((x - mu_1) ** 2) / (2 * sigma_1 ** 2))
    normal_2 = scale_2*np.exp(-((x - mu_2) ** 2) / (2 * sigma_2 ** 2))
    return normal_1 + normal_2 + b

def gaussian_distribution_triple(x, mu_1, sigma_1, scale_1, mu_2, sigma_2, scale_2, mu_3, sigma_3, scale_3, b):
    normal_1 = scale_1*np.exp(-((x - mu_1) ** 2) / (2 * sigma_1 ** 2))
    normal_2 = scale_2*np.exp(-((x - mu_2) ** 2) / (2 * sigma_2 ** 2))
    normal_3 = scale_3*np.exp(-((x - mu_3) ** 2) / (2 * sigma_3 ** 2))

    return normal_1 + normal_2 + normal_3 +b

def fit_gaussian(x_data, y_data, initial_params,maxloops,bounds):

    optimal_params, cov_matrix = curve_fit(gaussian_distribution_fit, x_data, y_data, p0=initial_params, maxfev = maxloops, bounds=bounds)
    y_fit = gaussian_distribution_fit(x_data, *optimal_params)
    ssd = calculate_ssd(y_fit, y_data)
    
    return optimal_params, y_fit, ssd

def fit_gaussian_bimode(x_data, y_data, initial_params, maxloops,bounds):

    optimal_params, cov_matrix = curve_fit(gaussian_distribution_combo, x_data, y_data, p0=initial_params, maxfev = maxloops, bounds=bounds)
    y_fit = gaussian_distribution_combo(x_data, *optimal_params)
    ssd = calculate_ssd(y_fit, y_data)
    
    return optimal_params, y_fit, ssd, cov_matrix

def fit_gaussian_triple(x_data, y_data, initial_params, maxloops,bounds):

    optimal_params, cov_matrix = curve_fit(gaussian_distribution_triple, x_data, y_data, p0=initial_params, maxfev = maxloops, bounds=bounds)
    y_fit = gaussian_distribution_triple(x_data, *optimal_params)
    ssd = calculate_ssd(y_fit, y_data)
    
    return optimal_params, y_fit, ssd, cov_matrix

#---------------------------------------------------------------------------------------#
#-----------------------------------------END-------------------------------------------#
#---------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
# -------------------------- Size Distribution Fitting Functions -------------------------- #
#-------------------------------------------------------------------------------------------#
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

def calculate_gaussian_parameters(diameter, counts, peak_info=None, mode=0):
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
    sigma = np.sqrt(second_mom - first_mom**2)

    # https://proofwiki.org/wiki/Variance_of_Gaussian_Distribution
    return  n, first_mom, sigma, x, y

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

def mixture_gaussian(x: np.ndarray, *params: float) -> np.ndarray:
    """
    Calculate the probability density function (PDF) of a mixture of Gaussian distributions.

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
        pdf += w * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return pdf

#---------------------------------------------------------------------------------------#
def denoise_signal(signal, sigma):
    smoothed_signal = gaussian_filter1d(signal, sigma)
    return smoothed_signal

def get_smoothed_for_weight(
    dataset: xr.Dataset,
    time: str,
    height: str,
    ax: plt.Axes = None,
    smoothed_color=None,
) -> Tuple[np.ndarray,Optional[float], Optional[float]]:
    """
    Return the smoothed DVS. Runs much faster than plotfulldopspec
    
    Args:
        dataset (xr.Dataset): LV0 dataset containing the Doppler spectrum data.
        time (str): Time in the format 'yyyymmddThhmmss.s'.
        height (str): Height in meters.
        ax (Optional[plt.Axes], optional): Matplotlib Axes object to plot on. If None, a new figure and axes will be created. Defaults to None.
        
    Returns:
        Smoothed spectrum (np.array): smoothed doppler spectrum
    """
    
    smoothed_spectrum = denoise_signal(dataset['doppler_spectrum'].loc[convert_to_sec_safe(time, dataset), getHeightIdx(height, dataset), :], 15)
    # new_time = convIndexToDatetime(convert_to_index(time, dataset), dataset)

    # ax.plot(dataset['velocity_vectors'][x], smoothed_spectrum, label=str(height), color=smoothed_color)

    # ax.legend()

    return smoothed_spectrum

#---------------------------------------------------------------------------------------

def get_lv0_gfatpy(ti_datetime, instrument_name, filepath_output):
    if not isinstance(ti_datetime, datetime.datetime):
        ti_datetime = pd.Timestamp(ti_datetime)
    pattern = f"{ti_datetime.strftime('%y%m%d_%H')}*.LV0"
    directorypath_rawdata = f"/home/matheustolen/shared/NAS_raw_data/UGR/{instrument_name}/{ti_datetime.strftime('%Y/%m/%d')}"

    filepath_rawdata = glob.glob(os.path.join(directorypath_rawdata, pattern))
    raw_filename = os.path.basename(filepath_rawdata[0]).replace('LV0', 'LV0.nc')
    filepath_raw_netcdf = Path(os.path.join(filepath_output, raw_filename))
    # check if raw netcdf file exists
    if not os.path.exists(filepath_raw_netcdf):
        rpg2nc_rpgpy(filepath_rawdata[0], output_file=filepath_raw_netcdf)
        # rpg2nc_rpgpy(filepath_rawdata[0]+"/*LV0", output_file=filepath_output+"/huge_day_file.nc") # If we want to concatenate one day file
        print(f"File created at {filepath_raw_netcdf}")
    
    return gfp_radarnc(filepath_raw_netcdf)