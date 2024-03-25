# Clear local and global variables
# locals().clear()
# globals().clear()
#-------------------------------------------------------------------------------------------------------
# import classes
#-------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import datetime
import pandas as pd
from scipy import integrate, interpolate
import os
from functools import reduce
import operator
from typing import List, Tuple, Dict, Any
import itertools
import xarray as xr
import seaborn as sns
import colorcet as cc  # Import the colorcet library
import multiprocessing 
import time as time_module
from pdb import set_trace
import seaborn as sns
import glob
from matplotlib import cm
from matplotlib.patches import Patch
# from cloudnetpy_qc import quality
#-------------------------------------------------------------------------------------------------------
from cloudnetpy.products import generate_lwc
from cloudnetpy.products import generate_iwc
from cloudnetpy.products import generate_ier
from cloudnetpy.products import generate_der
from cloudnetpy.products.der import Parameters
from cloudnetpy.plotting import plotting
import pandas as pd
#-------------------------------------------------------------------------------------------------------
plt.ion()
plt.close('all')
# Set the font to Times New Roman using LaTeX
fontsize = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize
#from cloud_classes import Intersection_products, HMmodel, CloudProcess
#-------------------------------------------------------------------------------------------------------
# paths
#-------------------------------------------------------------------------------------------------------
PATH_CLASS        = '/media/matheustolen/Seagate Basic/cloudnet/classification/'
PATH_CATE         = '/media/matheustolen/Seagate Basic/cloudnet/categorize/'
PATH_RADAR        = '/media/matheustolen/Seagate Basic/cloudnet/radar/'
PATH_FIG          = '../figures/'
PATH_CLOUDNET_LWC = '../../../output_retrievals/'
PATH_CLOUDNET_IWC = '../../../output_retrievals/'
PATH_CLOUDNET_DER = '../../../output_retrievals/'
PATH_CLOUDNET_IER = '../../../output_retrievals/'
PATH_GEN_CLOUDNET = "/home/matheustolen/Documentos/matheus_doctorado/output_retrievals/generated_cloudnet_product/"
#-------------------------------------------------------------------------------------------------------
# constants
#-------------------------------------------------------------------------------------------------------
CLEAR_SKY                       = 0   # Clear sky
CLOUD_LIQUID                    = 1   # Cloud liquid droplets only
DRIZZLE_OR_RAIN                 = 2   # Drizzle or rain
DRIZZLE_OR_RAIN_LIQUID_DROPLETS = 3   # Drizzle or rain coexisting with cloud liquid droplets
ICE_PARTICLES                   = 4   # Ice particles
ICE_WITH_SUP_WATER              = 5   # Ice coexisting with supercooled liquid droplets
MELTING_ICE                     = 6   # Melting ice particles
MELTING_ICE_LIQUID_DROPLETS     = 7   # Melting ice particles coexisting with cloud liquid droplets
AERO_NO_CLOUD                   = 8   # Aerosol particles, no cloud or precipitation
INSECT_NO_CLOUD                 = 9   # Insects, no cloud or precipitation
AERO_WITH_INSECT_NO_CLOUD       = 10  # Aerosol coexisting with insects, no cloud or precipitation
NBINS_BETWEEN_CLOUD             = 5
NBINS_BETWEEN_HYDRO             = 1
NBINS_CLOUD                     = 3
NBINS_GET_RAIN                  = 20
THRESHOLD_BELLOW                = 100 # [ m ]
THRESHOLD_ABOVE                 = 100 # [ m ]
THRESHOLD_LWP                   = 5e3 # [ g m-2 ]
FONT                            = {'family': 'serif',
                                   'color':  'black',
                                   'weight': 'normal',
                                   'size': 17,
                                   }
CLASSIFICATION_TICK_LABELS = ['Clear  sky', 
               'Droplets', 
               'Drizzle or rain', 
               'Drizzle & droplets', 
               'Ice', 
               'Ice & droplets', 
               'Melting ice', 
               'Melting & droplets', 
               'Aerosol',
               'Insect', 
               'Aerosol & insect',
               'No Data']
        
HYDRO_TYPES = { "Liquid"          : [CLOUD_LIQUID,DRIZZLE_OR_RAIN,DRIZZLE_OR_RAIN_LIQUID_DROPLETS],
                "Ice"             : [ICE_PARTICLES],
                "Mixed_phase"     : [ICE_WITH_SUP_WATER,\
                                        MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS],
                "Total"    : [CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS,\
                                        ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE,\
                                                MELTING_ICE_LIQUID_DROPLETS]
                }

TARG_BET_HYDRO = [CLEAR_SKY, AERO_NO_CLOUD, INSECT_NO_CLOUD,\
                                                AERO_WITH_INSECT_NO_CLOUD]

CLOUD_TYPES = { "Liquid"          : [CLOUD_LIQUID,DRIZZLE_OR_RAIN_LIQUID_DROPLETS],
                    "Ice"             : [ICE_PARTICLES],
                    "Mixed_phase"     : [CLOUD_LIQUID,ICE_PARTICLES,ICE_WITH_SUP_WATER,\
                                        MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS],
                    "Pre_liquid"      : [CLOUD_LIQUID,DRIZZLE_OR_RAIN_LIQUID_DROPLETS],
                    "Pre_mixed_phase" : [CLOUD_LIQUID, ICE_PARTICLES,ICE_WITH_SUP_WATER,MELTING_ICE,\
                                        MELTING_ICE_LIQUID_DROPLETS]
                }  

TARG_BET_CLOUD = {  "Liquid"          : [CLEAR_SKY, AERO_NO_CLOUD, INSECT_NO_CLOUD,\
                                                AERO_WITH_INSECT_NO_CLOUD],
                        "Ice"             : [CLEAR_SKY, CLOUD_LIQUID, AERO_NO_CLOUD,\
                                                INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD],
                        "Mixed_phase"     : [CLEAR_SKY, DRIZZLE_OR_RAIN, AERO_NO_CLOUD,\
                                                INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD],
                        "Pre_liquid"      : [CLEAR_SKY, AERO_NO_CLOUD, INSECT_NO_CLOUD,\
                                                AERO_WITH_INSECT_NO_CLOUD],
                        "Pre_mixed_phase" : [CLEAR_SKY, DRIZZLE_OR_RAIN, AERO_NO_CLOUD,\
                                                INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD]}

TARG_TO_FILTER = {"Liquid"          : [DRIZZLE_OR_RAIN, ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE,\
                                        MELTING_ICE_LIQUID_DROPLETS],
                    "Ice"             : [CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS,\
                                        ICE_WITH_SUP_WATER, MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS],
                    "Mixed_phase"     : [DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS],
                    "Pre_liquid"      : [ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS],
                    "Pre_mixed_phase" : []}

TARG_TO_GET_BELLOW = {"Pre_liquid"      : [DRIZZLE_OR_RAIN],
                        "Pre_mixed_phase" : [DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS]}

CLOUD_PHASE = ["single_phase", "single_phase", "mixed_phase", "single_phase", "mixed_phase", "single_phase"]
#-------------------------------------------------------------------------------------------------------
# functions and classes
#-------------------------------------------------------------------------------------------------------
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

def count_sublists_larger_than_1(lst):
    count = 0
    for sublist in lst:
        if len(sublist) > 1:
            count += 1
    return count

def round_datetimeindex_to_seconds(datetimeindex):
    rounded_datetimeindex = datetimeindex.round('1s')
    return rounded_datetimeindex

def flatten_list_with_itertools(lst):
    return list(itertools.chain.from_iterable(lst))

def sublist_lengths(lst):
    lengths = [len(sublist) for sublist in lst]
    return lengths

def mask_target(df, list_targets):
    """
    This function takes a DataFrame 'df' and a list of target values 'list_targets' 
    and returns a boolean mask indicating whether each element in the DataFrame 
    matches any of the target values.

    Parameters:
        df (DataFrame): The input DataFrame containing the data.
        list_targets (list): A list of target values to match against the DataFrame.

    Returns:
        merged_conditions (DataFrame): A boolean mask with the same shape as 'df', 
                                       where 'True' indicates a match with any of the 
                                       target values, and 'False' otherwise.
    """
    
    conditions = []
    
    # Iterate over each target value in the list of targets
    for targ in list_targets:
        # Create a condition that checks if the DataFrame 'df' is equal to the target value
        conditions.append(df == targ)
    
    # Merge all the conditions using the 'operator.or_' function, which performs element-wise
    # OR operation on DataFrames, resulting in a single boolean mask
    merged_conditions = reduce(operator.or_, conditions)
    
    return merged_conditions

def group_by_month(time_series_list):
    grouped_data = {}
    for data_point in time_series_list:
        month_key = data_point[0].strftime("%Y-%m")  # Grouping by year-month format

        if month_key not in grouped_data:
            grouped_data[month_key] = []
        grouped_data[month_key].append(data_point)

    return grouped_data

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

# def compare_radar_chirp_configurations(start_date: datetime, 
#                                        end_date: datetime,
#                                        database_intersection: List[datetime.datetime],
#                                        path_radar: str) -> Tuple[List[datetime.datetime], List[datetime.datetime], List[np.ndarray], List[np.ndarray]]:
#     """
#     Compare radar chirp configurations for multiple dates from NetCDF files.

#     Parameters:
#         start_date (datetime): The start date from which the processing should begin.
#         end_date (datetime): The end date to which the processing should continue.
#         database_intersection (List[datetime]): List of dates to process radar data for.
#         path_radar (str): The path where the NetCDF radar data files are located.

#     Returns:
#         Tuple[List[datetime], List[datetime], List[np.ndarray], List[np.ndarray]]: A tuple containing:
#             - start_chirp (List[datetime]): List of dates when the radar chirp configuration changed.
#             - end_chirp (List[datetime]): List of dates when the radar chirp configuration changed, except for the last one which is 'end_date'.
#             - chirp_zres (List[np.ndarray]): List of arrays containing the range resolutions for each date.
#             - height (List[np.ndarray]): List of arrays containing the height data for each date.
#     """
#     # Filter database_intersection to only include dates within the range
#     valid_dates = [date for date in database_intersection if start_date <= date <= end_date]
    
#     # Initialize lists to store results
#     start_chirp = [start_date]
#     end_chirp = []
#     chirp_zres = []
#     height = []
    
#     # Read and store the initial range resolution and chirp configuration
#     radar = nc.Dataset(path_radar + start_date.strftime('%Y%m%d') + "_granada_rpg-fmcw-94.nc")
#     range_resolution = np.round(radar['range_resolution'][:], 1)
#     chirp_zres.append(range_resolution)
#     height.append(radar['range'][:])
    
#     # Loop through the filtered dates and compare chirp configurations
#     for date in valid_dates:
#         radar = nc.Dataset(path_radar + date.strftime('%Y%m%d') + "_granada_rpg-fmcw-94.nc")
#         new_range_res = np.round(radar['range_resolution'][:], 1)
        
#         # Check if the new range resolution is the same as any previous one
#         if not any(np.array_equal(new_range_res, res) for res in chirp_zres):
#             print("Range resolution is not the same for all dates")
#             # Store the new date, chirp configuration, and height data if different

#             start_chirp.append(date)
#             end_chirp.append(date - timedelta(days=1))
#             chirp_zres.append(new_range_res)
#             height.append(radar['range'][:])
    
#     end_chirp.append(end_date)
#     return start_chirp, end_chirp, chirp_zres, height, valid_dates

def compare_radar_chirp_configurations(start_date: datetime, 
                                       end_date: datetime,
                                       database_intersection: List[datetime.datetime],
                                       path_radar: str) -> dict:
    """
    Compare radar chirp configurations for multiple dates from NetCDF files.

    Parameters:
        start_date (datetime): The start date from which the processing should begin.
        end_date (datetime): The end date to which the processing should continue.
        database_intersection (List[datetime]): List of dates to process radar data for.
        path_radar (str): The path where the NetCDF radar data files are located.

    Returns:
        dict: A dictionary containing range resolutions as keys and lists of corresponding dates as values.
    """
    # Filter database_intersection to only include dates within the range
    valid_dates = [date for date in database_intersection if start_date <= date <= end_date]
    
    # Initialize dictionary to store range resolutions and corresponding date lists
    resolution_dict = {}
    height_dict = {}
    current_resolution = None
    radar_file_pattern = '_granada_rpg-fmcw-94*.nc'
    
    # Loop through the filtered dates and compare chirp configurations
    for date in valid_dates:
        radar_file_pattern = date.strftime('%Y%m%d')+'_granada_rpg-fmcw-94*.nc'
        file_paths = glob.glob(os.path.join(path_radar,radar_file_pattern))[0]
        radar = nc.Dataset(file_paths)
        new_resolution = tuple(radar['range_resolution'][:])
        
        if current_resolution is None:
            current_resolution = new_resolution
        
        if new_resolution != current_resolution:
            if new_resolution not in resolution_dict:
                resolution_dict[new_resolution] = []
                height_dict[new_resolution] = radar['range'][:]
            resolution_dict[new_resolution].append(date)
        else:
            if current_resolution not in resolution_dict:
                resolution_dict[current_resolution] = []
                height_dict[current_resolution] = radar['range'][:]
            resolution_dict[current_resolution].append(date)
        
        current_resolution = new_resolution
    return resolution_dict, height_dict

# Define a function to handle serialization of individual columns
def handle_serialization(column):
    if isinstance(column, np.ma.MaskedArray):
        return column.filled(np.nan).tolist()
    return column

def plot_cloud_type(df_class, df_ze, cloud_filter, name_title, z_min, z_max, color_names, plot_ze=True, 
                    cloud_type= List[int], targ_between_cloud= List[int], integrated_variables=pd.DataFrame(), 
                    cloud_prop=xr.Dataset(), cloud_prop_corrected=xr.Dataset(), plot_cloud_prop=False):
    
        # List of manually specified colors (replace these with your desired colors)
    manual_colors = ["#57A1F7","#007CFF", "#0A2658", "#FFFF00", "#4EF6C1",\
                      "#D05BAC", "#BFBD8D", "#118527","#8794B3", "#DA6F49", "#88183E", "#DDDEDA"]
    ncolors = len(color_names)

    time_series    = df_class.index
    new_start_time = time_series.min().replace(hour=0, minute=0, second=15, microsecond=0)
    new_end_time   = time_series.max().replace(hour=23, minute=59, second=59, microsecond=0)
    new_time_index = pd.date_range(start=new_start_time, end=new_end_time, freq="30S")

    df_ze = df_ze[cloud_filter.cloud_mask()]
    df_class_filtered = df_class[cloud_filter.cloud_mask()]

    df_class_complete_time = df_class.reindex(index=new_time_index, fill_value=11)
    df_ze_complete_time = df_ze.reindex(index=new_time_index, fill_value=np.nan)
    df_class_filtered_complete_time = df_class_filtered.reindex(index=new_time_index, fill_value=11)


    # Create a figure and an array of subplots

    if plot_ze:
        fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(15, 10))

        # Create a ListedColormap using the manual colors
        manual_cmap = plt.cm.colors.ListedColormap(manual_colors)

        # Plot the classification heatmap
        f0 = axs[0].pcolormesh(df_class_complete_time.index, 
                                df_class_complete_time.columns/1000, 
                                np.transpose(df_class_complete_time),
                                cmap=manual_cmap,
                                vmin=0,
                                vmax=ncolors-1)

        i = 0
        for sublist in cloud_filter.cloud_base:
            axs[0].plot(np.tile(cloud_filter.time_cbt[i], len(sublist)), df_class.columns[sublist]/1000, "*", color='black', markersize=3)
            i += 1
        i = 0
        for sublist in cloud_filter.cloud_top:
            axs[0].plot(np.tile(cloud_filter.time_cbt[i], len(sublist)), df_class.columns[sublist]/1000, "*", color='red', markersize=3)
            i += 1

        axs[0].set_ylabel(r'Height [km]')
        axs[0].grid()
        
        # Plot the mixed phase categories heatmap
        f1 = axs[1].pcolormesh(df_class_filtered_complete_time.index, 
                            df_class_filtered_complete_time.columns/1000, 
                            np.transpose(df_class_filtered_complete_time),
                            cmap=manual_cmap,
                            vmin=0,
                            vmax=ncolors)
        
        i = 0
        for sublist in cloud_filter.cloud_base:
            axs[1].plot(np.tile(cloud_filter.time_cbt[i], len(sublist)), df_class.columns[sublist]/1000, "*", color='black', markersize=3)
            i += 1
        i = 0
        for sublist in cloud_filter.cloud_top:
            axs[1].plot(np.tile(cloud_filter.time_cbt[i], len(sublist)), df_class.columns[sublist]/1000, "*", color='red', markersize=3)
            i += 1

        axs[1].set_ylabel(r'Height [km]')
        axs[1].grid()
        
        # Create a colorbar with custom color patches and labels (vertical)
        colorbar1 = fig.colorbar(f1, ax=axs[0:2], ticks=[], orientation='vertical')

        # Adjust the position of colorbar and add color patches with names
        for idx, (color, name) in enumerate(zip(manual_cmap.colors, color_names)):
            rect = plt.Rectangle((0, idx), 1, 1, color=color)
            colorbar1.ax.add_patch(rect)
            colorbar1.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=17)


        # Plot the mixed phase Ze heatmap
        f2 = axs[2].pcolormesh(df_ze_complete_time.index, 
                            df_ze_complete_time.columns/1000, 
                            np.transpose(df_ze_complete_time),
                            cmap='viridis',
                            vmin=-40,
                            vmax=10)

        axs[2].set_ylabel(r'Height [km]')
        axs[2].set_xlabel(r'Time [UTC]')
        colorbar2 = fig.colorbar(f2, ax=axs[2:3])
        colorbar2.set_label('Ze [dBz]')  # Add a label to the colorbar
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Customize x-axis limits based on hour_s and hour_e
        # axs[2].set_xlim([cloud_filter.classification.index.date[0] + pd.DateOffset(hour=11), 
        #              cloud_filter.classification.index.date[0] + pd.DateOffset(hour=17)])
        
        # Set y-axis limits
        axs[2].set_ylim([z_min, z_max])
        axs[2].set_xlim([time_series.min(), time_series.max()])
        axs[2].grid()
        plt.suptitle(name_title)
        plt.show()
    else:
        fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(15, 10))
        
        # Create a ListedColormap using the manual colors
        manual_cmap = plt.cm.colors.ListedColormap(manual_colors)

        # Plot the classification heatmap
        f0 = axs[0].pcolormesh(df_class_complete_time.index, 
                                df_class_complete_time.columns/1000, 
                                np.transpose(df_class_complete_time),
                                cmap=manual_cmap,
                                vmin=0,
                                vmax=ncolors-1)

        i = 0
        for sublist in cloud_filter.cloud_base:
            axs[0].plot(np.tile(cloud_filter.time_cbt[i], len(sublist)), df_class.columns[sublist]/1000, "*", color='black', markersize=3)
            i += 1
        i = 0
        for sublist in cloud_filter.cloud_top:
            axs[0].plot(np.tile(cloud_filter.time_cbt[i], len(sublist)), df_class.columns[sublist]/1000, "*", color='red', markersize=3)
            i += 1

        axs[0].set_ylabel(r'Height [km]')
        axs[0].grid()
        
        # Plot the mixed phase categories heatmap
        f1 = axs[1].pcolormesh(df_class_filtered_complete_time.index, 
                            df_class_filtered_complete_time.columns/1000, 
                            np.transpose(df_class_filtered_complete_time),
                            cmap=manual_cmap,
                            vmin=0,
                            vmax=ncolors)
        
        i = 0
        for sublist in cloud_filter.cloud_base:
            axs[1].plot(np.tile(cloud_filter.time_cbt[i], len(sublist)), df_class.columns[sublist]/1000, "*", color='black', markersize=3)
            i += 1
        i = 0
        for sublist in cloud_filter.cloud_top:
            axs[1].plot(np.tile(cloud_filter.time_cbt[i], len(sublist)), df_class.columns[sublist]/1000, "*", color='red', markersize=3)
            i += 1

        axs[1].set_ylabel(r'Height [km]')
        axs[1].grid()
        
        # Create a colorbar with custom color patches and labels (vertical)
        colorbar1 = fig.colorbar(f1, ax=axs[0:2], ticks=[], orientation='vertical')

        # Adjust the position of colorbar and add color patches with names
        for idx, (color, name) in enumerate(zip(manual_cmap.colors, color_names)):
            rect = plt.Rectangle((0, idx), 1, 1, color=color)
            colorbar1.ax.add_patch(rect)
            colorbar1.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=20)
        
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Customize x-axis limits based on hour_s and hour_e
        axs[1].set_xlim([cloud_filter.classification.index.date[0] + pd.DateOffset(hour=12), 
                     cloud_filter.classification.index.date[0] + pd.DateOffset(hour=17)])
        
        axs[1].set_xlabel(r'Time [UTC]')
        # Set y-axis limits
        axs[1].set_ylim([z_min, z_max])
        # axs[1].set_xlim([time_series.min(), time_series.max()])
        axs[1].grid()
        plt.suptitle(f"{name_title} Cloud ({cloud_type}) - {df_class_filtered_complete_time.index.date[0]}\nB/W cloud: {np.array(color_names)[targ_between_cloud]}")
        # fig.savefig(PATH_FIG + name_title + '_example.png', dpi=300)
        plt.show()

def plot_cloud_comparison(df_class, df_ze, cloud_filter, name_title, z_min, z_max, color_names, plot_ze=True, 
                         cloud_type= List[int], targ_between_cloud= List[int], integrated_variables=pd.DataFrame(), 
                         cloud_prop=xr.Dataset(), plot_cloud_prop=False):
    
    time_series    = df_class.index
    new_start_time = time_series.min().replace(hour=0, minute=0, second=15, microsecond=0)
    new_end_time   = time_series.max().replace(hour=23, minute=59, second=59, microsecond=0)
    new_time_index = pd.date_range(start=new_start_time, end=new_end_time, freq="30S")
    date_string    = new_start_time.strftime("%Y%m%d")
    
    # List of manually specified colors (replace these with your desired colors)
    manual_colors = ["#57A1F7","#007CFF", "#0A2658", "#FFFF00", "#4EF6C1",\
                      "#D05BAC", "#BFBD8D", "#118527","#8794B3", "#DA6F49", "#88183E", "#DDDEDA"]
    ncolors = len(color_names)

    manual_cmap = plt.cm.colors.ListedColormap(manual_colors)

    df_ze = df_ze[cloud_filter.cloud_mask()]
    df_class_filtered = df_class[cloud_filter.cloud_mask()]
    ds_mask = xr.Dataset({'cloud_mask': (['time', 'height'], cloud_filter.cloud_mask())}, coords={'time': df_class.index, 'height': df_class.columns})
    
    # Nw, apply the mask. Where mask is False, put NaNs
    cloud_prop = cloud_prop.where(ds_mask.cloud_mask)
    
    df_ze_complete_time             = df_ze.reindex(index=new_time_index, fill_value=np.nan)
    df_class_filtered_complete_time = df_class_filtered.reindex(index=new_time_index, fill_value=np.nan)
    der_corrected_complete_time     = cloud_prop['der_corrected'].reindex(time=new_time_index, fill_value=np.nan)
    der_complete_time               = cloud_prop['der'].reindex(time=new_time_index, fill_value=np.nan)
    der_scaled_corrected_complete_time = cloud_prop['der_scaled_corrected'].reindex(time=new_time_index, fill_value=np.nan)
    der_scaled_complete_time        = cloud_prop['der_scaled'].reindex(time=new_time_index, fill_value=np.nan) 
    der_knist_complete_time         = cloud_prop['reff_kist_mh'].reindex(time=new_time_index, fill_value=np.nan)
    print("Der scaled corrected data points: ", der_scaled_corrected_complete_time.count().values)
    print("Der knist data points: ", der_knist_complete_time.count().values)
    # set_trace()

    if plot_cloud_prop:
        fig, axs = plt.subplots(5, sharex=True, sharey=True, figsize=(20, 18))

        # Plot the mixed phase Ze heatmap
        f0 = axs[0].pcolormesh(df_ze_complete_time.index, 
                            df_ze_complete_time.columns/1000, 
                            np.transpose(df_ze_complete_time),
                            cmap='viridis',
                            vmin=-40,
                            vmax=10)

        axs[0].set_ylabel(r'Height [km]')
        colorbar2 = fig.colorbar(f0, ax=axs[0])
        colorbar2.set_label('Ze [dBz]')  # Add a label to the colorbar
        
        # Customize x-axis limits based on hour_s and hour_e
        # axs[2].set_xlim([cloud_filter.classification.index.date[0] + pd.DateOffset(hour=11), 
        #              cloud_filter.classification.index.date[0] + pd.DateOffset(hour=17)])
        
        # Set y-axis limits
        # axs[0].set_ylim([z_min, z_max])
        # axs[0].set_xlim([time_series.min(), time_series.max()])
        axs[0].grid()
        plt.suptitle(name_title)

        # Plot der complete time:
        f1 = axs[1].pcolormesh(der_complete_time.time, 
                            der_complete_time.height/1000, 
                            der_complete_time.values.T,
                            cmap='jet',
                            vmin=0,
                            vmax=15/1e6)
        
        axs[1].set_ylabel(r'Height [km]')
        axs[1].set_title(f'{der_complete_time.long_name}')
        colorbar2 = fig.colorbar(f1, ax=axs[1])
        colorbar2.set_label(f'{der_complete_time.units}')  # Add a label to the colorbar

        # Plot der corrected complete time:
        f2 = axs[2].pcolormesh(der_corrected_complete_time.time, 
                            der_corrected_complete_time.height/1000, 
                            der_corrected_complete_time.values.T,
                            cmap='jet',
                            vmin=0,
                            vmax=15/1e6)
        
        axs[2].set_ylabel(r'Height [km]')
        axs[2].set_title(f'{der_complete_time.long_name} - Corrected')
        colorbar2 = fig.colorbar(f2, ax=axs[2])
        colorbar2.set_label(f'{der_complete_time.units}')  # Add a label to the colorbar
        axs[2].grid()

        # Plot der knist complete time:
        f3 = axs[3].pcolormesh(der_knist_complete_time.time, 
                            der_knist_complete_time.height/1000, 
                            der_knist_complete_time.values.T/1e6,
                            cmap='jet',
                            vmin=0,
                            vmax=15/1e6)
        
        axs[3].set_ylabel(r'Height [km]')
        axs[3].set_title(f'{der_complete_time.long_name} - Knist Corrected')
        colorbar2 = fig.colorbar(f3, ax=axs[3])
        colorbar2.set_label(f'm')  # Add a label to the colorbar
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axs[3].grid()

        # Plot the the classificatio 
                # Plot the mixed phase categories heatmap
        f4 = axs[4].pcolormesh(df_class_filtered_complete_time.index, 
                            df_class_filtered_complete_time.columns/1000, 
                            np.transpose(df_class_filtered_complete_time),
                            cmap=manual_cmap,
                            vmin=0,
                            vmax=ncolors)
        
        i = 0
        for sublist in cloud_filter.cloud_base:
            axs[4].plot(np.tile(cloud_filter.time_cbt[i], len(sublist)), df_class.columns[sublist]/1000, "*", color='black', markersize=3)
            i += 1
        i = 0
        for sublist in cloud_filter.cloud_top:
            axs[4].plot(np.tile(cloud_filter.time_cbt[i], len(sublist)), df_class.columns[sublist]/1000, "*", color='red', markersize=3)
            i += 1

        axs[4].set_ylabel(r'Height [km]')
        
        # Create a colorbar with custom color patches and labels (vertical)
        colorbar1 = fig.colorbar(f4, ax=axs[4], ticks=[], orientation='vertical')

        # Adjust the position of colorbar and add color patches with names
        for idx, (color, name) in enumerate(zip(manual_cmap.colors, color_names)):
            rect = plt.Rectangle((0, idx), 1, 1, color=color)
            colorbar1.ax.add_patch(rect)
            colorbar1.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=20)
        
        axs[4].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # # Customize x-axis limits based on hour_s and hour_e
        # axs[4].set_xlim([cloud_filter.classification.index.date[0] + pd.DateOffset(hour=14), 
        #              cloud_filter.classification.index.date[0] + pd.DateOffset(hour=15)])
        
        axs[4].set_xlabel(r'Time [UTC]')
        # Set y-axis limits
        # axs[4].set_ylim([0, 3.2])
        # axs[4].set_xlim([time_series.min(), time_series.max()])
        axs[4].grid()
        fig.savefig(f"{PATH_FIG}{date_string}_{name_title}_comparison_pcolor.png", dpi=300)
        plt.show()

        #Now, get all values of liquid radios and plot a PDF
        effective_radii_cloudnet_corr = der_corrected_complete_time.values.flatten()*1e6
        effective_radii_cloudnet = der_complete_time.values.flatten()*1e6
        effective_radii_cloudnet_scaled = der_scaled_complete_time.values.flatten()*1e6
        effective_radii_cloudnet_scaled_corrected = der_scaled_corrected_complete_time.values.flatten()*1e6
        effective_radii_knist = der_knist_complete_time.values.flatten()
        
        
        use_kde = True
        density_hist = 'count'
        alpha_hist = .4
        element_type = 'step'
        fig, axs = plt.subplots(figsize=(13, 7))
        sns.histplot(effective_radii_cloudnet_corr, bins=100, kde=use_kde, label=f'Cloudnet Corrected (Median: {np.nanmedian(effective_radii_cloudnet_corr):.1f} um)', element=element_type, color='blue', edgecolor='blue', stat=density_hist, alpha=alpha_hist)
        sns.histplot(effective_radii_cloudnet, bins=100, kde=use_kde, label=f'Cloudnet (Median: {np.nanmedian(effective_radii_cloudnet):.1f} um)', element=element_type, color='red', edgecolor='red', stat=density_hist, alpha=alpha_hist)
        sns.histplot(effective_radii_knist, bins=100, kde=use_kde, label=f'Knist (Median: {np.nanmedian(effective_radii_knist):.1f} um)', element=element_type, color='green', edgecolor='green', stat=density_hist, alpha=alpha_hist)
        sns.histplot(effective_radii_cloudnet_scaled, bins=100, kde=use_kde, label=f'Cloudnet Scaled (Median: {np.nanmedian(effective_radii_cloudnet_scaled):.1f} um)', element=element_type, color='purple', edgecolor='purple', stat=density_hist, alpha=alpha_hist)
        sns.histplot(effective_radii_cloudnet_scaled_corrected, bins=100, kde=use_kde, label=f'Cloudnet Scaled Corrected (Median: {np.nanmedian(effective_radii_cloudnet_scaled_corrected):.1f} um)', element=element_type, color='orange', edgecolor='orange', stat=density_hist, alpha=alpha_hist)
        axs.set_xlabel('Effective Radius [um]')
        axs.set_ylabel('Counts')
        axs.set_title(name_title +' Effective Radius Distribution')
        axs.legend()
        fig.savefig(f"{PATH_FIG}{date_string}_{name_title}_der_comparison_hist.png", dpi=300, bbox_inches='tight')
        plt.show()
# Example usage:
# plot_classification_and_phases(df_classification, mixed_phase_cat, mixed_phase_ze, classification_filter, cloud_base, name_title, hour_s, hour_e, z_min, z_max)
def plot_cloud_mask(df_complete, df_mask, name_title, z_min, z_max):
    # Create a figure and an array of subplots
    # df_new = df_complete[~df_mask] == np.nan
    fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(18, 10))

    # Plot the classification heatmap
    f0 = axs[0].pcolormesh(df_complete.index, 
                            df_complete.columns, 
                            np.transpose(df_complete),
                            cmap='tab10',
                            vmin=0,
                            vmax=10)

    axs[0].set_ylabel(r'Height [m]')
    axs[0].grid()
    fig.colorbar(f0, ax=axs[0])
    
    # Plot the mixed phase Ze heatmap
    f1 = axs[1].pcolormesh(df_mask.index, 
                           df_mask.columns, 
                          np.transpose(df_complete[df_mask]),
                          cmap='tab10',
                          vmin=0,
                          vmax=10)

    axs[1].set_ylabel(r'Height [m]')
    axs[1].set_xlabel(r'Time [UTC]')
    fig.colorbar(f1, ax=axs[1])
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Customize x-axis limits based on hour_s and hour_e
    # axs[2].set_xlim([cloud_filter.classification.index.date[0] + pd.DateOffset(hour=hour_s), 
    #             cloud_filter.classification.index.date[0] + pd.DateOffset(hour=hour_e)])
    
    # Set y-axis limits
    axs[1].set_ylim([z_min, z_max])
    
    axs[1].grid()
    plt.suptitle(name_title)
    plt.show()

def plot_integrated_variables(ds_cloud_prop: xr.Dataset):
    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the 'LWP' data on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time [UTC]')
    ax1.set_ylabel('LWP [g m$^{-2}$]', color=color)
    ax1.plot(ds_cloud_prop.time.values, ds_cloud_prop['LWP'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis
    ax2 = ax1.twinx()

    # Plot the 'IWP' data on the second y-axis
    color = 'tab:red'
    ax2.set_ylabel('IWP [g m$^{-2}]$', color=color)
    ax2.plot(ds_cloud_prop.time.values, ds_cloud_prop['IWP']*1e3, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Format x-axis labels to display hour and minute
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Set the super title with the date
    # plt.suptitle(ds_cloud_prop.time.values[0].strftime("%Y-%m-%d"))

    # Show the plot
    plt.title('LWP and IWP over Time')
    plt.show()

#------------------------------------------------------------------------------------------------
# cloudnet algorithm to generate netcdf files with liquid water content (lwc) and droplet 
# effective radius (der)
#------------------------------------------------------------------------------------------------
def generate_cloudnet_products(date, path_cate, path_cloudnet_lwc = None, path_cloudnet_iwc = None, path_cloudnet_der = None, path_cloudnet_ier = None, params_der=None):
 
    # Generate LWC
    if path_cloudnet_lwc is not None:
        lwc_input_path = os.path.join(path_cate, date.strftime('%Y%m%d') + "_granada_categorize.nc")
        lwc_output_path = os.path.join(path_cloudnet_lwc, date.strftime('%Y%m%d') + "_granada_" + 'lwc.nc')
        generate_lwc(lwc_input_path, lwc_output_path)

    # Generate IWC
    if path_cloudnet_iwc is not None:
        iwc_input_path = os.path.join(path_cate, date.strftime('%Y%m%d') + "_granada_categorize.nc")
        iwc_output_path = os.path.join(path_cloudnet_iwc, date.strftime('%Y%m%d') + "_granada_" + 'iwc.nc')
        generate_iwc(iwc_input_path, iwc_output_path)

    # Generate DER
    if path_cloudnet_der is not None:
        der_input_path = os.path.join(path_cate, date.strftime('%Y%m%d') + "_granada_categorize.nc")
        der_output_path = os.path.join(path_cloudnet_der, date.strftime('%Y%m%d') + "_granada_" + 'der.nc')
        if params_der is not None:
            # params = Parameters(2.0, 100.0e6, 200.0e6, 0.25, 0.1, 5.0e-3)
            generate_der(der_input_path, der_output_path, parameters=params_der)
        else:
            generate_der(der_input_path, der_output_path)
        # generate_der(der_input_path, der_output_path, parameters=params)

    # Generate IER
    if path_cloudnet_ier is not None:
        ier_input_path = os.path.join(path_cate, date.strftime('%Y%m%d') + "_granada_categorize.nc")
        ier_output_path = os.path.join(path_cloudnet_ier, date.strftime('%Y%m%d') + "_granada_" + 'ier.nc')
        generate_ier(ier_input_path, ier_output_path)

def check_time_resolution(classification, categorize, date):
    try:
        if (classification.dimensions['time'].size == categorize.dimensions['time'].size
            and sum(categorize['time'][:] == classification['time'][:]) == categorize.dimensions['time'].size):

            time_auxiliary = []
            for h in categorize['time']:
                time_auxiliary.append(date.strftime('%Y%m%d') + ' ' + str(timedelta(hours=float(h))))

            time = round_datetimeindex_to_seconds(pd.to_datetime(time_auxiliary, format='mixed', dayfirst=True))
            return time
        else:
            print("Red flag: classification and categorize files with different time resolution - ", date)
            return None
    except Exception as e:
        print("An error occurred:", str(e))
        return None

# Define a function to rechunk the dataset
def rechunk_dataset(ds, chunks):
    rechunked_ds = ds.chunk(chunks)
    return rechunked_ds

def process_cloud_data_parallel(args):
    # Unpack the arguments
    (height, nchirp, date, chirp_res)  = args
    # ------------------------------------------------------------------------------------------------
    # generate cloudnet products (Comment this line if you already have the files)
    #------------------------------------------------------------------------------------------------
    # generate_cloudnet_products(date, 
    #                             PATH_CATE,
    #                             path_cloudnet_lwc=PATH_CLOUDNET_LWC, 
    #                             path_cloudnet_iwc=PATH_CLOUDNET_IWC,
    #                             path_cloudnet_der=PATH_CLOUDNET_DER, 
    #                             path_cloudnet_ier=PATH_CLOUDNET_IER)
    #------------------------------------------------------------------------------------------------
    # reading categorize, classification and radar files
    #------------------------------------------------------------------------------------------------
    radar_file_pattern = '_granada_rpg-fmcw-94*.nc'
    radar_file_path    = glob.glob(os.path.join(PATH_RADAR, date.strftime('%Y%m%d')+radar_file_pattern))[0]

    categorize     = nc.Dataset(PATH_CATE+date.strftime('%Y%m%d')+"_granada_categorize.nc")
    classification = nc.Dataset(PATH_CLASS+date.strftime('%Y%m%d')+"_granada_classification.nc")
    radar          = xr.open_dataset(radar_file_path)
    # ------------------------------------------------------------------------------------------------
    # check if categorize and classification files have the same time resolution 
    # ------------------------------------------------------------------------------------------------
    time = check_time_resolution(classification, categorize, date)
    if not np.array_equal(radar['range_resolution'].values, chirp_res):
        print(f"Warning: radar range resolution and chirp resolution are different - {date} : {radar.range.size} vs {height.size}")
        # print(f"radar resolution: {radar['range_resolution'].values} vs heigth resolution: {np.unique(np.diff(height.data))}")
        # print(f" chirp resolution: {chirp_res}")
    if categorize['height'][:].size != height.size:
        print(f"Warning: radar and categorize files with different height bins - {date} : {categorize['height'][:].size} vs {height.size}")
    # if not np.array_equal(height, categorize['height'][:]-categorize['altitude'][:]):
    #     print(f"Warning: radar and categorize files with different height bins - {date} : {radar.range.size} vs {height.size}")
    # if not np.array_equal(radar.range.values, height):
    
    # ------------------------------------------------------------------------------------------------
    # reading some cloudnet products
    # ------------------------------------------------------------------------------------------------
    cloudnet_lwc = xr.open_dataset(PATH_CLOUDNET_LWC+date.strftime('%Y%m%d')+"_granada_"+'lwc-scaled-adiabatic.nc')
    cloudnet_iwc = xr.open_dataset(PATH_CLOUDNET_IWC+date.strftime('%Y%m%d')+"_granada_"+'iwc-Z-T-method.nc')
    cloudnet_der = xr.open_dataset(PATH_CLOUDNET_DER+date.strftime('%Y%m%d')+"_granada_"+'der.nc')
    cloudnet_ier = xr.open_dataset(PATH_CLOUDNET_IER+date.strftime('%Y%m%d')+"_granada_"+'ier.nc')

    cloud_products = xr.merge([cloudnet_lwc, cloudnet_iwc, cloudnet_der, cloudnet_ier], compat='no_conflicts', join='exact')
    cloud_products = cloud_products.reindex(time=time, method='nearest') # reindexing to merge with LWP and IWP
    cloud_products.coords['height'] = height
    # ------------------------------------------------------------------------------------------------
    # dataframes_list_ze.append(radar.Zh)
    # dataframes_list_vd.append(radar.v)
    ds_hydrometeor            = xr.Dataset(coords={'time': time, 'range': height})
    number_of_layers          = pd.DataFrame(index=time,
                                columns=["Liquid", "Ice", "Mixed_phase", "Pre_liquid", "Pre_mixed_phase"])
    height_cloud_base         = pd.DataFrame(index=time, 
                                    columns=["Liquid", "Ice", "Mixed_phase", "Pre_liquid", "Pre_mixed_phase"])
    height_cloud_top          = pd.DataFrame(index=time,
                                    columns=["Liquid", "Ice", "Mixed_phase", "Pre_liquid", "Pre_mixed_phase"])
    height_cloud_mean         = pd.DataFrame(index=time,
                                    columns=["Liquid", "Ice", "Mixed_phase", "Pre_liquid", "Pre_mixed_phase"])
    geometric_cloud_thickness = pd.DataFrame(index=time,
                                    columns=["Liquid", "Ice", "Mixed_phase", "Pre_liquid", "Pre_mixed_phase"])
    df_reflectivity   = pd.DataFrame(data =categorize['Z'][:],
                                        index =time,
                                        columns=height)
    
    df_classification = pd.DataFrame(data=classification['target_classification'][:],
                                        index  =time,
                                        columns=height)
    # ------------------------------------------------------------------------------------------------
    mask_outliers_lwp = categorize['lwp'][:] > THRESHOLD_LWP
    if np.any(mask_outliers_lwp):
        categorize['lwp'][mask_outliers_lwp] = np.nan
    # ------------------------------------------------------------------------------------------------
    ds_integrated_variables = xr.Dataset(
                    {'LWP': (['time'], categorize['lwp'][:]),
                     'IWP': (['time'], np.trapz(cloudnet_iwc.iwc, height, axis=1))},
                     coords={'time': time})

    cloud_physical_properties = xr.merge([cloud_products, ds_integrated_variables], join='exact')   
    # ------------------------------------------------------------------------------------------------
    xr_radar_variables = xr.merge([xr.DataArray(categorize['Z'][:], coords={'time': time, 'range': height}, name='Z'),
                           xr.DataArray(categorize['v'][:], coords={'time': time, 'range': height}, name='v')], compat='no_conflicts', join='exact')
    # ------------------------------------------------------------------------------------------------
    if process_for_specific_analysis:
        # ------------------------------------------------------------------------------------------------
        # generate corrected cloudnet products (comment if you dont want to compare products)
        #------------------------------------------------------------------------------------------------
        os.system("rm "+PATH_GEN_CLOUDNET+"*lwc.nc") # remove all lwc files from lwc path 
        os.system("rm "+PATH_GEN_CLOUDNET+"*der.nc") # remove all der files from der path
        os.system("rm "+PATH_GEN_CLOUDNET+"*iwc.nc") # remove all der files from der path
        os.system("rm "+PATH_GEN_CLOUDNET+"*ier.nc") # remove all der files from der path
        generate_cloudnet_products(date, 
                                    PATH_CATE,
                                    path_cloudnet_lwc=PATH_GEN_CLOUDNET, 
                                    path_cloudnet_iwc=PATH_GEN_CLOUDNET,
                                    path_cloudnet_der=PATH_GEN_CLOUDNET, 
                                    path_cloudnet_ier=PATH_GEN_CLOUDNET,
                                    params_der=Parameters(2.0, 150.0e6, 150.0e6, 0.28, 0.1, 5.0e-3))
        # ------------------------------------------------------------------------------------------------
        # reading corrected cloudnet products (comment if you dont want to compare products)
        # ------------------------------------------------------------------------------------------------
        cloudnet_der_corrected = xr.open_dataset(PATH_GEN_CLOUDNET+date.strftime('%Y%m%d')+"_granada_"+'der.nc')
        cloudnet_der_corrected = cloudnet_der_corrected.reindex(time=time, method='nearest') # reindexing to merge with LWP and IWP
        cloud_physical_properties['der_corrected'] = xr.DataArray(cloudnet_der_corrected['der'].values, coords=cloud_physical_properties.coords, dims=cloud_physical_properties.dims)
        cloud_physical_properties['der_scaled_corrected'] = xr.DataArray(cloudnet_der_corrected['der_scaled'].values, coords=cloud_physical_properties.coords, dims=cloud_physical_properties.dims)
    # # ------------------------------------------------------------------------------------------------
    # # Effective radius
    # # ------------------------------------------------------------------------------------------------
    # fig, ax = plt.subplots(figsize=(13, 6))
    # mesh = ax.pcolormesh(cloud_physical_properties.time.values, (cloud_physical_properties.height.values - cloud_physical_properties.altitude.values)/1000, cloud_physical_properties['der'].T, cmap='rainbow')
    # cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.04, shrink=1.0, aspect=30, label=f"{cloud_physical_properties['der'].units}")
    # ax.set_title(cloud_physical_properties['der'].long_name)
    # ax.set_xlabel('Time (UTC)')
    # ax.set_ylabel('Height (km)')
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # plt. show()
    # # ------------------------------------------------------------------------------------------------
    # fig, ax = plt.subplots(figsize=(13, 6))
    # mesh = ax.pcolormesh(cloud_physical_properties.time.values, (cloud_physical_properties.height.values - cloud_physical_properties.altitude.values)/1000, cloud_physical_properties['ier'].T, cmap='rainbow')
    # cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.04, shrink=1.0, aspect=30, label=f"{cloud_physical_properties['ier'].units}")
    # ax.set_title(cloud_physical_properties['ier'].long_name)
    # ax.set_xlabel('Time (UTC)')
    # ax.set_ylabel('Height (km)')
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # plt.show()
    # # ------------------------------------------------------------------------------------------------
    # # LWC
    # # ------------------------------------------------------------------------------------------------
    # fig = plt.figure(figsize=(12, 7))
    # gs = fig.add_gridspec(2, 2, width_ratios=[35, 1], height_ratios=[3, 1], hspace=0.01, wspace=0.05)

    # ax1 = fig.add_subplot(gs[0, 0])
    # mesh1 = ax1.pcolormesh(cloud_physical_properties.time.values, (cloud_physical_properties.height.values - cloud_physical_properties.altitude.values)/1000, cloud_physical_properties['lwc'].T, cmap='rainbow')
    # ax1.set_title(cloud_physical_properties['lwc'].long_name)
    # ax1.set_ylabel('Height (km)')
    # ax1.xaxis.set_visible(False)  # Hide the x-axis
    # # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # cax = fig.add_subplot(gs[0, 1])
    # cbar1 = plt.colorbar(mesh1, cax=cax, orientation='vertical', label=f"{cloud_physical_properties['lwc'].units}")

    # ax2 = fig.add_subplot(gs[1, 0])
    # mesh2 = ax2.plot(cloud_physical_properties.time.values, cloud_physical_properties['LWP'], label='LWP', color='blue')
    # ax2.set_xlabel('Time (UTC)')
    # ax2.set_ylabel(r'LWP (kg m$^{-2}$)', color='blue')
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # plt.show()
    # # ------------------------------------------------------------------------------------------------
    # # IWP
    # # ------------------------------------------------------------------------------------------------
    # fig = plt.figure(figsize=(12, 7))
    # gs = fig.add_gridspec(2, 2, width_ratios=[35, 1], height_ratios=[3, 1], hspace=0.01, wspace=0.05)

    # ax1 = fig.add_subplot(gs[0, 0])
    # mesh1 = ax1.pcolormesh(cloud_physical_properties.time.values, (cloud_physical_properties.height.values - cloud_physical_properties.altitude.values)/1000, cloud_physical_properties['iwc'].T, cmap='rainbow')
    # ax1.set_title(cloud_physical_properties['iwc'].long_name)
    # ax1.set_ylabel('Height (km)')
    # ax1.xaxis.set_visible(False)  # Hide the x-axis
    # # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # cax = fig.add_subplot(gs[0, 1])
    # cbar1 = plt.colorbar(mesh1, cax=cax, orientation='vertical', label=f"{cloud_physical_properties['iwc'].units}", 
    #                      ticks=np.arange(np.nanmin(cloud_physical_properties['iwc'].values), np.nanmax(cloud_physical_properties['iwc'].values), 0.0005))

    # ax2 = fig.add_subplot(gs[1, 0])
    # mesh2 = ax2.plot(cloud_physical_properties.time.values, cloud_physical_properties['IWP'], label='IWP', color='red')
    # ax2.set_xlabel('Time (UTC)')
    # ax2.set_ylabel(r'IWP (kg m$^{-2}$)', color='red')
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # plt.show()
    # # ------------------------------------------------------------------------------------------------
    # # Classification
    # # ------------------------------------------------------------------------------------------------
    # plotting.generate_figure(PATH_CLASS+date.strftime('%Y%m%d')+"_granada_classification.nc", field_names=['target_classification'], show=True)
    # # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    try: 
        process_cloud_data(date, xr_radar_variables, 
                           df_classification,
                           df_reflectivity, 
                           cloud_physical_properties,
                           number_of_layers,
                           height_cloud_base, 
                           height_cloud_top, 
                           height_cloud_mean, 
                           geometric_cloud_thickness,
                           ds_hydrometeor, 
                           time, 
                           height, 
                           nchirp)
    except Exception as e:
        print(f"An error occurred at date {date}", str(e))
        return None
    # ------------------------------------------------------------------------------------------------

def process_cloud_data(date: datetime.datetime,
                        xr_radar_variables: xr.Dataset,
                        df_classification: pd.DataFrame,
                        df_reflectivity: pd.DataFrame,
                        cloud_physical_properties: xr.Dataset,
                        number_of_layers: pd.DataFrame,
                        height_cloud_base: pd.DataFrame,
                        height_cloud_top: pd.DataFrame,
                        height_cloud_mean: pd.DataFrame,
                        geometric_cloud_thickness: pd.DataFrame,
                        ds_hydrometeor: xr.Dataset, 
                        time: pd.DatetimeIndex,
                        height: np.ndarray,
                        nchirp: int):
    # ------------------------------------------------------------------------------------------------
    # CLOUD filters for calculations of cloud properties 
    # ------------------------------------------------------------------------------------------------
    for i, cloud in enumerate(CLOUD_TYPES):
        classification_filter = CloudProcess(df_classification.copy(), 
                                             CLOUD_TYPES[cloud], 
                                             TARG_BET_CLOUD[cloud], 
                                             NBINS_BETWEEN_CLOUD,
                                             CLOUD_PHASE[i],
                                             NBINS_CLOUD)
        for target in TARG_TO_FILTER[cloud]:
            if target == DRIZZLE_OR_RAIN:
                classification_filter.filter_species(target, 10000, 200)
            else:
                classification_filter.filter_species(target, NBINS_BETWEEN_CLOUD, NBINS_BETWEEN_CLOUD)
        
        if cloud == "Pre_liquid" or cloud == "Pre_mixed_phase":
            classification_filter.get_specie_below(TARG_TO_GET_BELLOW[cloud], NBINS_GET_RAIN)
        classification_filter.calculate_cloud_properties(height_cloud_base,
                                                        height_cloud_top,
                                                        height_cloud_mean,
                                                        geometric_cloud_thickness,
                                                        cloud)
        
        number_of_layers.loc[classification_filter.time_cbt, cloud] = sublist_lengths(classification_filter.cloud_base)
        # ------------------------------------------------------------------------------------------------
        # All plots should be done here
        # ------------------------------------------------------------------------------------------------
        # with sns.axes_style("whitegrid"):
        #     plot_cloud_type(df_classification,
        #                     df_reflectivity, 
        #                     classification_filter, 
        #                     cloud, .1, 5., 
        #                     CLASSIFICATION_TICK_LABELS, 
        #                     plot_ze=False,
        #                     cloud_type= CLOUD_TYPES[cloud],
        #                     targ_between_cloud= TARG_BET_CLOUD[cloud],
        #                     integrated_variables = cloud_physical_properties)
        if process_for_specific_analysis:
            num_mh = pd.DataFrame(index=time,
                                data=np.full(time.size, np.nan)) # mh: homogeneos mixing 
            ref_mh = pd.DataFrame(data=np.full((time.size, height.size), np.nan), 
                                    index=time, 
                                    columns=height)  # mh: homogeneos mixing
            
            #OBS: The cloud mask is already applied in the cloud_physical_properties
            mask_reff_cloudnet = cloud_physical_properties['der'].isnull().to_numpy()
            ze_cloud           = df_reflectivity.where(~mask_reff_cloudnet, np.nan)
            # num_data = ze_cloud.count().sum()
            # ze_cloud      = df_reflectivity[classification_filter.cloud_mask()]
            # ------------------------------------------------------------------------------------------------
            for ind in time:
                z_profile                 = ze_cloud.loc[ind]
                ze                        = 10**(z_profile[z_profile.notnull()]/10) # Remove NaNs 
                                                                                    # and convert Z from 
                                                                                    # dBz to mm^6 m^-3
                                                                                    # bellow, converting from kg m^-2 --> g m^-2
                cloud_model               = HMmodel(cloud_physical_properties['LWP'].sel(time=ind).values*1e3,
                                                    ze.index.array,
                                                    ze,
                                                    v=13.51)
                num_mh.loc[ind]           = cloud_model.get_num()
                ref_mh.loc[ind, ze.index] = cloud_model.get_re()
            
            num_mh.replace([np.inf, -np.inf], np.nan, inplace=True)
            # ref_mh.replace(0, np.nan, inplace=True)

            cloud_physical_properties['num_knist_mh']  = xr.DataArray(num_mh.values[:,0], coords={'time': cloud_physical_properties['time'].values}, dims=['time'])
            cloud_physical_properties['reff_kist_mh']  = xr.DataArray(ref_mh.values, coords=cloud_physical_properties.coords, dims=cloud_physical_properties.dims)
            # getting data for same pixels as cloudnet products
            # mask_for_data = cloud_physical_properties['der'].isnull()
            # cloud_physical_properties['reff_kist_mh'] = cloud_physical_properties['reff_kist_mh'].where(mask_for_data)
            # set_trace()
            plot_cloud_comparison(df_classification,
                                    df_reflectivity, 
                                    classification_filter, 
                                    cloud, .1, 5., 
                                    CLASSIFICATION_TICK_LABELS, 
                                    cloud_type= CLOUD_TYPES[cloud],
                                    targ_between_cloud= TARG_BET_CLOUD[cloud],
                                    integrated_variables= cloud_physical_properties, 
                                    cloud_prop=cloud_physical_properties,
                                    plot_cloud_prop=True)
        # set_trace()
    #------------------------------------------------------------------------------------------------
    # Analisis hydrometeors 
    #------------------------------------------------------------------------------------------------ 
    for cloud in HYDRO_TYPES:
        classification_filter = CloudProcess(df_classification.copy(), 
                                             HYDRO_TYPES[cloud], 
                                             TARG_BET_HYDRO, 
                                             NBINS_BETWEEN_HYDRO)
        
        cloud_mask = classification_filter.cloud_mask()
        ds_hydrometeor[cloud] = xr.DataArray(cloud_mask, dims=('time', 'range'), coords={'time': time, 'range': height})
        # with sns.axes_style("whitegrid"):
        #     plot_cloud_type(df_classification,
        #                     df_reflectivity, 
        #                     classification_filter, 
        #                     cloud, .1, 12., CLASSIFICATION_TICK_LABELS)
    #------------------------------------------------------------------------------------------------
    number_of_layers.index.name = 'time'
    ds_layers          = xr.Dataset.from_dataframe(number_of_layers)
    name_folders_nc    = ["radar_variables", "cloud_physical_properties", "hydrometeor", "number_of_layers"]
    # ------------------------------------------------------------------------------------------------
    # Save the dataset as a NetCDF file inside the chirp folder
    # ------------------------------------------------------------------------------------------------
    if save_cloudnet_products:
        for ds, name in zip([xr_radar_variables, cloud_physical_properties, ds_hydrometeor, ds_layers], name_folders_nc):
            folder_name = f"../../../processed_data/chirp_{nchirp}/{name}/"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            
            output_path = os.path.join(folder_name, f"{date.strftime('%Y%m%d')}_{name}.nc")
            # Determine chunking size based on dimensions present in the dataset
            # chunks = {}
            # for dim in ds.dims:
            #     chunks[dim] = ds[dim].size
            # # Rechunk the dataset
            # rechunked_ds = rechunk_dataset(ds, chunks)
            # # Save the rechunked dataset
            # rechunked_ds.to_netcdf(output_path)
            ds.to_netcdf(output_path)
        # ------------------------------------------------------------------------------------------------
        # Save the dataset as a JSON file inside the chirp folder
        # ------------------------------------------------------------------------------------------------
        height_cloud_base.index.name = 'time'
        height_cloud_top.index.name  = 'time'
        height_cloud_mean.index.name = 'time'
        geometric_cloud_thickness.index.name = 'time'
        name_folders_json  = ["height_cloud_base", "height_cloud_top", "height_cloud_mean", "geometric_cloud_thickness"]
        
        for df, name in zip([height_cloud_base, height_cloud_top, height_cloud_mean, geometric_cloud_thickness],
                        name_folders_json):
            folder_name = f"../../../processed_data/chirp_{nchirp}/{name}/"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            
            try:
                # Convert columns to serializable format
                output_path = os.path.join(folder_name, f"{date.strftime('%Y%m%d')}_{name}.json")
                df_serializable = df.applymap(handle_serialization)
                df_serializable.to_json(output_path, orient='index')
            except Exception as e:
                print(f"Error while saving {name}: {e}")
    # ------------------------------------------------------------------------------------------------

class HMmodel:
    def __init__(self,
                 lwp,
                 h,
                 z,
                 v=8.7,
                 rw=1e6):
        
        self.cloud_lwp = lwp # liquid water path
        self.cloud_z   = h   # cloud thickeness
        self.radar_ze  = z
        self.nu        = v
        self.rho_w     = rw
        self.k_nv      = (v+3)*(v+4)*(v+5)/( v*(v+1)*(v+2) ) 
        self.k_rv      = (v+2)/( (v+3)*(v+4)*(v+5) )**(1./3.)
        
    def get_num(self):
        # droplet concentration in cm^-3
        #print(integrate.trapz(np.sqrt(self.radar_ze), self.cloud_z))
        # ( g m^-2 / ( g m^-3 mm^3 m^-3/2 * m )  )^2 
        # ( m^3/2 mm^-3 )^2
        #  m^3 (10^-3 m)^-6
        #  m^3 10^-18 m^-6 
        # 10-18 m^-3
        # um^-3
        # To convert from um^-3 to cm^-3: 10^12 * (um^-3) = cm^-3
        return 1e12*self.k_nv*( 6*self.cloud_lwp/( np.pi*self.rho_w*np.trapz(np.sqrt(self.radar_ze), self.cloud_z) ) )**2
        #return 1e12*self.k_nv*( 6*self.cloud_lwp/( np.pi*self.rho_w*np.nansum(np.sqrt(self.radar_ze))*30 ) )**2
    
    def get_re(self):
        # (  g m^-3 mm^3 m^-3/2 * m / ( g m^-2)  )^1/3 mm m^-1/3
        # (  mm^3 m^-3/2  )^1/3 mm m^-1/2
        # (  mm m^-1/2  ) mm m^-1/2
        # (  mm^2 m^-1
        # (  10^-6 m^2 m^-1 )
        # ( 10^-6 m ) = um
        #print(  ( np.pi*self.rho_w*np.trapz(np.sqrt(self.radar_ze), self.cloud_z)/(48*self.cloud_lwp) )**(1./3.) * self.radar_ze**(1./6.)  ) 
        return self.k_rv*( np.pi*self.rho_w*np.trapz(np.sqrt(self.radar_ze), self.cloud_z)/(48*self.cloud_lwp) )**(1./3.) * self.radar_ze**(1./6.)
        
class CloudProcess:
    def __init__(self,
                  df_class, 
                  cloud_value,
                  no_cloud_target_values,
                  consecutive_bins,
                  cloud_type="single_phase",
                  bins_cloud=0):
        
        self.classification   = df_class # Dataframe
        self.height           = df_class.columns
        self.time             = df_class.index
        self.cloud            = cloud_value # array of integers or integer
        self.cloud_type       = cloud_type 
        self.no_cloud_targ    = no_cloud_target_values
        self.consecutive_bins = consecutive_bins
        self.nbins_cloud      = bins_cloud

        self.cloud_indexes()
        self.cloud_boundaries()
    
    def cloud_indexes(self):
        
        # if np.size(self.cloud)==1:
        #     self.row_cloud, self.col_cloud = np.where(self.classification == self.cloud)
        # else:
        conditions = []
        for elem in self.cloud:
            conditions.append(self.classification == elem)
        merged_conditions = reduce(operator.or_, conditions)
        self.row_cloud, self.col_cloud = np.where(merged_conditions)
        # self.classification[~merged_conditions] = np.nan
        
        self.row_cloud_unique = np.unique(self.row_cloud)
        self.col_cloud_boundaries = self.time.shape[0]*[np.nan]
        if self.cloud_type == "mixed_phase":
            remove_unique = []
            for i, i_time in enumerate(self.row_cloud_unique):
                sequence_cloud  = groupSequence( self.col_cloud[self.row_cloud == i_time], self.nbins_cloud) 
                sequence_cloud  = self.new_sequence(i_time, self.no_cloud_targ, self.consecutive_bins, sequence_cloud)
                n_layer         = len(sequence_cloud)
                count           = 0
                remove_position = []
                for j, sublist in enumerate(sequence_cloud):
                    aux_classification = self.classification.iloc[i_time, sublist]
                    if self.are_all_elements_equal(aux_classification, ICE_PARTICLES) or\
                        self.are_all_elements_equal(aux_classification, CLOUD_LIQUID):
                        remove_position.append(j)
                        count += 1

                if count>0:
                    # print(sequence_cloud)
                    self.remove_elements_using_pop(sequence_cloud, remove_position)
                    # print(sequence_cloud, count, n_layer)
                if count < n_layer:
                    self.col_cloud_boundaries[i_time] = sequence_cloud
                    # tem_nan = self.col_cloud_boundaries[i_time] == np.nan
                    # if tem_nan==True:
                    #     break
                if count == n_layer:
                    remove_unique.append(i)
                    # print(count, n_layer)
            self.row_cloud_unique = np.delete(self.row_cloud_unique, remove_unique)
            # print(sum(self.col_cloud_boundaries != np.nan) == self.row_cloud_unique.shape[0])
        else:    
            for i, i_time in enumerate(self.row_cloud_unique):
                sequence_cloud = groupSequence( self.col_cloud[self.row_cloud == i_time], self.nbins_cloud) # liquid pixels sequancies
                self.col_cloud_boundaries[i_time] = self.new_sequence(i_time, self.no_cloud_targ, self.consecutive_bins, sequence_cloud)

    def new_sequence(self, time, targets, nbins, sequence_cloud):
        n_sequence = len(sequence_cloud)
        # print(sequence_cloud)
        i=0
        while i < n_sequence-1 and n_sequence > 1:
            ini                     = sequence_cloud[i][-1]
            end                     = sequence_cloud[i+1][0]
            hole                    = self.classification.iloc[time, ini+1:end]
            seq_bins, ind_i, ind_f  = self.count_consecutive(targets, hole) # array with classification values
            if seq_bins > nbins:
                i+=1
            else:
                new_layer      = sorted(sequence_cloud[i] + sequence_cloud[i+1])
                sequence_cloud.pop(i)
                sequence_cloud.pop(i)
                sequence_cloud.insert(i, new_layer)
                n_sequence -= 1

        return sequence_cloud
    
    def cloud_boundaries(self):
        self.cloud_base = []
        self.cloud_top  = []
        for i_time in self.row_cloud_unique: 
            self.cloud_base.append([sublist[0] for sublist in self.col_cloud_boundaries[i_time]])
            self.cloud_top.append([sublist[-1] for sublist in self.col_cloud_boundaries[i_time]])
        
        self.time_cbt = self.time[self.row_cloud_unique]
    
    def are_all_elements_equal(self, df, value):
        return (df == value).all().all()
    
    def remove_elements_using_pop(self, lst, positions):
        # Sort positions in reverse order to avoid index shifting during deletion
        positions.sort(reverse=True)
        for pos in positions:
            lst.pop(pos)

    def cloud_mask(self):
        mask = pd.DataFrame(False, index=self.classification.index,\
                             columns=self.classification.columns)
        for i_time in self.row_cloud_unique:
            for sublist in self.col_cloud_boundaries[i_time]:
                mask.iloc[i_time,sublist] = True
        return mask
    
    def calculate_cloud_properties(self,
                                   height_cloud_base: pd.DataFrame,
                                   height_cloud_top: pd.DataFrame,
                                   height_cloud_mean: pd.DataFrame,
                                   geometric_cloud_thickness: pd.DataFrame,
                                   cloud_type: str) -> None:
        """
        Calculates cloud properties based on provided data and updates the corresponding DataFrame objects.

        Parameters:
            height_cloud_base (DataFrame): A DataFrame containing cloud base height data.
            height_cloud_top (DataFrame): A DataFrame containing cloud top height data.
            height_cloud_mean (DataFrame): A DataFrame containing mean cloud height data.
            geometric_cloud_thickness (DataFrame): A DataFrame containing geometric cloud thickness data.

        Returns:
            None: This function does not return anything, as it updates the DataFrame objects in-place.
        """
        for i, time_cloud in enumerate(self.time_cbt):
            h_cb = self.height[self.cloud_base[i]]  # Get the cloud base height for the current time.
            h_ct = self.height[self.cloud_top[i]]  # Get the cloud top height for the current time.
            
            # Update the corresponding DataFrames for "Ice" at the current time.
            height_cloud_base.loc[time_cloud, cloud_type] = h_cb
            height_cloud_top.loc[time_cloud, cloud_type] = h_ct
            height_cloud_mean.loc[time_cloud, cloud_type] = (h_cb + h_ct) / 2
            geometric_cloud_thickness.loc[time_cloud, cloud_type] = (h_ct - h_cb)
    
    def contain_value(list1: List[int], list2: List[int]) -> List[int]:
        # Create a mask to mark positions in the first list where elements are present in the second list.
        mask = [1 if elem in list2 else 0 for elem in list1]
        return mask
                
    def quantile_nearest_pixel(self, specie):
        ''' description '''
        row_cloud, col_cloud    = np.where(self.classification == self.cloud)
        row_specie, col_specie  = np.where(self.classification == specie)
        intersection            = np.intersect1d(row_cloud, row_specie) # Temporal index with in Dataframe with coexistence of cloud type and especie
                                                                        # Specie could be ice, rain, aerosol etc....
        no_candidate            = -1.0*self.classification.shape[1]
        dz_bellow               = []
        dz_above                = []
        for i_time in intersection: 
            sequence_cloud  = groupSequence( col_cloud[row_cloud == i_time], self.nbins_cloud ) # liquid pixels sequancies  
            sequence_specie = groupSequence( col_specie[row_specie == i_time] ) 
            
            for cloud_thickness in sequence_cloud:
                
                dbl = no_candidate  # delta bellow the cloud
                ibl = np.nan        # pixel bellow the cloud
        
                dab = abs(no_candidate) # delta bellow the cloud
                iab = np.nan            # pixel bellow the cloud
                
                for specie_thickness in sequence_specie:
            
                    dnew = specie_thickness[-1] - cloud_thickness[0] # new delta bellow the cloud
                    if dnew < 0 and dnew > dbl:
                        dbl = dnew
                        ibl = specie_thickness[-1]
                        
                    dnew = specie_thickness[0] - cloud_thickness[-1] # new delta above the cloud
                    if dnew > 0 and dnew < dab:
                        dab = dnew
                        iab = specie_thickness[0]
         
                if not np.isnan(ibl) and not np.isnan(iab):
                    dz_bellow.append( self.height[cloud_thickness[0]] - self.height[ibl] )
                    dz_above.append( self.height[iab] - self.height[cloud_thickness[-1]] )
                elif not np.isnan(ibl):
                    dz_bellow.append( self.height[cloud_thickness[0]] - self.height[ibl] )
                elif not np.isnan(iab):
                    dz_above.append( self.height[iab] - self.height[cloud_thickness[-1]] )
                  
        return dz_bellow, dz_above
        
    def filter_species(self, specie, dzb_max, dzt_max):
        ''' description '''

        row_specie, col_specie  = np.where(self.classification == specie)
        intersection            = np.intersect1d(self.row_cloud_unique, row_specie) # Temporal index with in Dataframe with coexistence of cloud type and especie
        
        # for ind in intersection:
        #     print(self.col_cloud_boundaries[ind])
        #         # print("tem Nans")
        
        if intersection.any():                                                                # Specie could be ice, rain, aerosol etc....
            no_candidate        = -1.0*self.classification.shape[1]
            for i_time in intersection: 
                # print(i_time in intersection)
                sequence_cloud  = self.col_cloud_boundaries[i_time]
                sequence_specie = groupSequence( col_specie[row_specie == i_time] ) 
    
                # print(self.classification.iloc[i_time, self.col_cloud_boundaries[i_time]])
                for cloud_thickness in sequence_cloud:
                    dbl = no_candidate      # delta bellow the cloud
                    dab = abs(no_candidate) # delta bellow the cloud
                    ibl = np.nan            # pixel bellow the cloud
                    iab = np.nan            # pixel bellow the cloud
                    for specie_thickness in sequence_specie:
                        dnew = specie_thickness[-1] - cloud_thickness[0] # new delta bellow the cloud
                        if dnew < 0 and dnew > dbl:
                            dbl = dnew
                            ibl = specie_thickness[-1]
                        
                        dnew = specie_thickness[0] - cloud_thickness[-1] # new delta above the cloud
                        if dnew > 0 and dnew < dab:
                            dab = dnew
                            iab = specie_thickness[0]
                        
                    # look if there is soecie inside the cloud:
                    if sum(self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] == specie) > 0.0:
                        self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan
        
                    if not np.isnan(ibl):
                        # dzb = self.height[cloud_thickness[0]] - self.height[ibl]
                        #print(dzb)
                        # if dzb <= dzb_max:
                        delta_bin = cloud_thickness[0] - ibl
                        if delta_bin < dzb_max:
                            self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan
                            # layer_to_remove.append(j)
                        #self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan
                    if not np.isnan(iab):
                        # dzt = self.height[iab] - self.height[cloud_thickness[-1]]
                        delta_bin = iab - cloud_thickness[-1]
                        # if dzt <= dzt_max:
                        if delta_bin < dzt_max:
                            self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan
                            # layer_to_remove.append(j)
                        #self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan
        
        # Update cloud indexes and cloud boundaries
        self.cloud_indexes()
        self.cloud_boundaries()

    def filtering_dz(self, dz_min):
        for i_time in self.row_cloud_unique:
            sequence_z = self.col_cloud_boundaries[i_time]
            for group_z in sequence_z:
                # dz = self.height[group_z].max() - self.height[group_z].min()
                dz = self.height[group_z[-1]] - self.height[group_z[0]]
                if dz < dz_min:
                    self.classification.iloc[i_time,group_z] = np.nan
        
        self.cloud_indexes()
        self.cloud_boundaries()   

    def filtering_dt(self, dt_min):
        # """ description """
        # # filtered_dt = self.classification.copy()
        # cloud_ini = self.cloud_base[0]
        # i=0
        # n_base = len(self.cloud_base)
        # while i < n_base:
        #     nb_actual = len(self.cloud_base[i])
        #     nb_next   = len(self.cloud_base[i+1])
        #     if nb_actual == nb_next:
        #         dz = self.height[self.cloud_base[i]] - self.height[self.cloud_base[i+1]]
        #         if dz >= 100:
        #             #not in the same cloud layer
        #             dt = self.time[self.cloud_base[i]] - self.time[cloud_ini]
        #             if dt < datetime.timedelta(minutes=dt_min):
        #                 # exclui todos os dados dentro da base e topo da nuvem dentro desse intervalo
        #                 for j in range(cloud_ini,i+1):
        #                     self.classification.iloc[j,self.cloud_base[j]:self.cloud_top[j]] = np.nan
        #     else:
        #         # compara cada bin 

        if self.row_cloud_unique.any():
            sequence_time   = groupSequence(self.row_cloud_unique)
            for group_t in sequence_time:
                dt = self.time[group_t].max() - self.time[group_t].min()
                if dt < datetime.timedelta(minutes=dt_min):
                    for i_time in group_t:
                        sequence_z = self.col_cloud_boundaries[i_time]
                        for group_z in sequence_z:
                            self.classification.iloc[i_time,group_z] = np.nan
        
    def get_specie_below(self, specie, max_bins):
        ''' description '''
        if np.size(specie)==1:
            row_specie, col_specie = np.where(self.classification == specie[0])
        else:
            conditions = []
            for elem in specie:
                conditions.append(self.classification == elem)
            merged_conditions = reduce(operator.or_, conditions)
            row_specie, col_specie = np.where(merged_conditions) 
     
        # row_specie, col_specie  = np.where(self.classification == specie)
        intersection            = np.intersect1d(self.row_cloud_unique, row_specie) # Temporal index with in Dataframe with coexistence of cloud type and especie
        # mask = np.full(self.classification.shape, False) 
        mask                    = np.full(self.classification.shape[0], False)
        mask[intersection]      = True
        self.classification.loc[~mask,:]=np.nan
        
        if intersection.any():
            # no_candidate = -1.0*self.classification.shape[1]
            for i_time in intersection: 
                sequence_cloud  = self.col_cloud_boundaries[i_time] # liquid pixels sequancies  
                sequence_specie = groupSequence( col_specie[row_specie == i_time] )
                
                # max_layers      = len( sequence_cloud )
                threshold = 0 # index of the ground
                # print(sequence_cloud)
                for i, cloud_thickness in enumerate(sequence_cloud):
                    # dbl = no_candidate      # delta bellow the cloud
                    # ibl = np.nan            # pixel bellow the cloud
                    bin_cbase  = cloud_thickness[0]
                    bin_ctop   = cloud_thickness[-1]
                    closest_bin = 0
                    for specie_layer in sequence_specie:
                        if specie_layer[-1] < bin_cbase and specie_layer[-1] < threshold:
                            self.classification.iloc[i_time, threshold:bin_ctop+1] = np.nan
                        
                        # elif specie_layer[0] > bin_ctop: 
                        #     self.classification.iloc[i_time, threshold+1:specie_layer[0]] = np.nan
                        #     if i+1==n_layers:
                        #         self.classification.iloc[i_time, threshold+1:] = np.nan
                        elif specie_layer[-1] < bin_cbase:
                            closest_bin = specie_layer[-1]
                        else:
                            self.classification.iloc[i_time, threshold:bin_ctop+1] = np.nan
                    

                    if bin_cbase-closest_bin > max_bins:
                        self.classification.iloc[i_time, threshold+1:bin_ctop+1] = np.nan
                        
                    threshold = bin_ctop

        self.cloud_indexes()
        self.cloud_boundaries()
                    
    def count_consecutive(self, numbers_set, sample):
        mask = [1 if num in numbers_set else 0 for num in sample]

        current_count = 0
        max_count = 0
        start_index = 0
        end_index = 0

        for i, value in enumerate(mask):
            if value:
                if current_count == 0:
                    start_index = i
                current_count += 1
                if current_count > max_count:
                    max_count = current_count
                    end_index = i
            else:
                current_count = 0

        return max_count, start_index, end_index


def get_date_from_files_wo_ldr(path):
    """
    Reads all netCDF files in a given directory using xarray.
    
    Args:
    path (str): Path to directory containing netCDF files.
    
    Returns:
    list: List of xarray.Dataset objects, one for each netCDF file in the directory.
    
    The following code was copied from Cloudnet quality check github repository:
    https://github.com/actris-cloudnet/cloudnetpy-qc/blob/v1.13.6/cloudnetpy_qc/quality.py
    
    class TestLDR(Test):
    def run(self):
        has_ldr = "ldr" in self.nc.variables or "sldr" in self.nc.variables
        has_v = "v" in self.nc.variables
        if has_v and has_ldr:
            v = self.nc["v"][:]
            ldr = (
                self.nc["ldr"][:] if "ldr" in self.nc.variables else self.nc["sldr"][:]
            )
            v_count = ma.count(v)
            ldr_count = ma.count(ldr)
            if v_count > 0 and (ldr_count / v_count * 100) < 0.1:
                self._add_warning("LDR exists in less than 0.1 % of pixels.")

    """
    files = [f for f in os.listdir(path) if f.endswith('.nc')]
    date = []
    for file in files:
        
        dataset = nc.Dataset(os.path.join(path, file))
        ldr_count = np.ma.count(dataset['ldr']) 
        v_count   = np.ma.count(dataset['v'])
        if v_count > 0 and (ldr_count / v_count * 100) < 0.1:
            print(f"File: {file[:8]} - LDR exists in less than 0.1 % of pixels.")
            date.append(datetime.datetime.strptime(file[:8], '%Y%m%d'))
    return date


def break_into_sequences(timestamps: List[datetime.datetime]):
    """
    Breaks a list of timestamps into sequences of consecutive dates.
    
    Args:
        timestamps (List[datetime.datetime]): List of timestamps.
        
    Returns:
        List[List[datetime.datetime]]: List of sequences of consecutive dates.
    """
    sequences = []
    current_sequence = [timestamps[0]]

    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i-1] == timedelta(days=1):
            current_sequence.append(timestamps[i])
        else:
            sequences.append(current_sequence)
            current_sequence = [timestamps[i]]

    sequences.append(current_sequence)
    return sequences

def get_total_folder_size(path: str):
    """
    Calculates the total size of folders in a given path.
    
    Args:
        path (str): The path to the folder.
        
    Returns:
        None
    """
    # Get the dates without LDR from the files in the given path
    dates_without_ldr = sorted(get_date_from_files_wo_ldr(path))
    
    # Define the path to the radar NAS folder
    path_radar_nas = "/home/matheustolen/shared/NAS_raw_data/UGR/nephele"
    
    # Create a list of folder paths in the NAS folder corresponding to the dates without LDR
    folders_path_nephele_nas = [os.path.join(path_radar_nas, date.strftime('%Y/%m/%d')) for date in dates_without_ldr]

    # Initialize the total size variable
    total_size = 0
    
    # Iterate over each folder path
    for folder_path in folders_path_nephele_nas:
        try:
            # Calculate the size of each file in the folder and sum them up
            folder_size = sum(os.path.getsize(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)))
            
            # Convert the size to gigabytes
            folder_size_gb = folder_size / (1024**3)
            
            # Add the folder size to the total size
            total_size += folder_size_gb
            
            # Print the folder size
            print(f"Folder {folder_path} size: {folder_size_gb:.3f} GB")
        except FileNotFoundError as e:
            print(f"Folder {folder_path} does not exist")
    
    # Print the total size
    print(f"Total size: {total_size:.3f} GB")

def plot_chirp_intervals(intervals_dic: Dict[Any, Any], height_dic: Dict[Any, Any], PATH_FIG: str) -> None:
    """
    Plots chirp intervals based on input dictionaries of intervals and heights.

    Parameters:
    - intervals_dic (Dict[Any, Any]): Dictionary containing chirp intervals.
    - height_dic (Dict[Any, Any]): Dictionary containing height data.
    - PATH_FIG (str): Path where the generated plot will be saved.

    Returns:
    - None
    """

    # Initialize an empty DataFrame to store interval data
    df = pd.DataFrame(columns=['start_times', 'end_times', 'range_res'])

    # Iterate through range resolutions and corresponding intervals
    for range_res, interval in intervals_dic.items():
        height = height_dic[range_res].data[:]
        time_intervals = break_into_sequences(interval)
        start_times = [min(time_interval) for time_interval in time_intervals]
        end_times   = [max(time_interval) for time_interval in time_intervals]
        
        # Process height data to identify changes
        diff_range = np.round(np.diff(height), 1)
        change_indices   = np.where(np.diff(diff_range) != 0)[0] + 2
        height_intervals = [height[0], *height[change_indices], height[-1]]
        print(height_intervals)

        # Concatenate interval data to the DataFrame
        df = pd.concat([df, pd.DataFrame({'start_times': start_times, 'end_times': end_times, 'range_res': [range_res]*len(start_times)})], ignore_index=True)

    # Sort the DataFrame by start times
    df = df.sort_values(by='start_times')

    # Extract year, month, day, and width information
    df['year'] = df['start_times'].dt.year
    df['month'] = df['start_times'].dt.month
    df['day'] = df['start_times'].dt.day
    df['width'] = df['end_times'] - df['start_times'] + timedelta(days=1)
    df['width'] = pd.to_timedelta(df['width'])
    df['range_res'] = df['range_res'].apply(lambda x: tuple(round(val, 2) for val in x))
    
    # Group the DataFrame by range resolution
    grouped_df = df.groupby('range_res')

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    color_dict = {}
    
    decimals = 1
    # Iterate through each group in the grouped DataFrame
    for i, (range_res, group) in enumerate(grouped_df):
        color = cm.Set1(i / len(grouped_df))
        
        # Iterate through each row in the group
        for index, row in group.iterrows():
            start_date = row['start_times']
            end_date = start_date + row['width']

            # Plot the bar with horizontal width according to the 'width' column
            ax.barh(row['year'], width=row['width'].days, left=(start_date - pd.Timestamp(start_date.year, 1, 1)).days,
                color=color, alpha=0.9)
        
        # Convert each element to a string with the specified number of decimals
        formatted_elements = [f"{element:.{decimals}f}" for element in range_res]
        # Join the formatted elements with "-"
        result_string = "--".join(formatted_elements)
        # Add the range_res and color to the dictionary
        color_dict[result_string] = color

    # Set labels and title
    ax.set_xlabel('Day/Month')

    # Customize the x-axis to represent days of the year and format ticks as day/month
    days_in_year = (pd.Timestamp('2023-01-01') - pd.Timestamp('2022-01-01')).days
    ax.set_xlim(0, days_in_year)
    ax.set_xticks(range(0, days_in_year, 30))
    ax.set_xticklabels([(pd.Timestamp('2022-01-01') + timedelta(days=i)).strftime('%d/%m') for i in range(0, days_in_year, 30)])
    ax.grid()

    # Create custom legend handles with colored bars
    legend_handles = [Patch(color=color_dict[label], label=str(label)) for label in color_dict.keys()]

    # Add a single legend outside the loop with custom handles
    legend = ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)

    # Remove super title
    fig.suptitle('')

    # Remove legend border
    legend.get_frame().set_linewidth(0)

    # Adjust layout to make room for the legend
    plt.tight_layout()

    # Remove spines, save the figure, and show the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    fig.savefig(PATH_FIG + "chirp_intervals.png", bbox_inches='tight', dpi=300)
    plt.show()

#--------------------------------------------------------------------------------------------------------
# Define procedures to perform
#--------------------------------------------------------------------------------------------------------
check_files_without_ldr = False
plot_chirp_time_series  = False
process_database        = True
save_cloudnet_products  = False

process_for_specific_analysis = True
#-------------------------------------------------------------------------------------------------------
# Check the size day folder withot LDR for nephele in NAS
#-------------------------------------------------------------------------------------------------------
if check_files_without_ldr:
    get_total_folder_size(PATH_RADAR)
#--------------------------------------------------------------------------------------------------------
# Startind data processing
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# Define paths for microphysics comparison
#--------------------------------------------------------------------------------------------------------
if process_for_specific_analysis:
    PATH_RADAR_COMP = "/home/matheustolen/shared/NAS_raw_data/UGR/nephele"
    PATH_CATE_COMP  = "/home/matheustolen/shared/NAS_raw_data/CATE"
    PATH_CLASS_COMP = "/home/matheustolen/shared/NAS_raw_data/CLASS"
    paths           = [PATH_RADAR, PATH_CATE, PATH_CLASS]
else: #--------------------------------------------------------------------------------------------------------
    paths     = [PATH_RADAR, PATH_CATE, PATH_CLASS]

extension = '.nc'
database_intersection  = common_prefix_of_filenames(paths, extension)
start_date = min(database_intersection) # first date of database
end_date   = max(database_intersection) # last date of database

start_date = datetime.datetime(2022, 4, 22)
end_date   = datetime.datetime(2022, 4, 22, 23, 59, 59)

# start_date = datetime.datetime(2018, 6, 1)
# end_date   = datetime.datetime(2018, 11, 1)

#TODO: should create a loop to iterate over all chirp configurations
intervals_dic, height_dic = compare_radar_chirp_configurations(start_date, end_date, database_intersection, PATH_RADAR)

# print("\nRemoving all cloudnet files of LWC and Reff from its directory...")

# os.system("rm "+PATH_CLOUDNET_LWC+"*lwc.nc") # remove all lwc files from lwc path 
# os.system("rm "+PATH_CLOUDNET_DER+"*der.nc") # remove all der files from der path
# os.system("rm "+PATH_CLOUDNET_IWC+"*iwc.nc") # remove all der files from der path
# os.system("rm "+PATH_CLOUDNET_IER+"*ier.nc") # remove all der files from der path

# print("All file removed")

if plot_chirp_time_series:
    plot_chirp_intervals(intervals_dic, height_dic, PATH_FIG)

if process_database:
    # Create a list of arguments for parallel processing
    processing_args = []
    
    for nchirp, key_res in enumerate(intervals_dic):
        height     = height_dic[key_res]
        # time_complete     = pd.date_range(start=start_date+timedelta(seconds=15),
        #                                   end=end_date + pd.Timedelta(days=1),
        #                                   freq='30S')
        # ----------------------------------------------------------------------------------------------------
        # for ind_day, date in enumerate([database_intersection.date[18]]):
        # start_time = time_module.time()
        for date in intervals_dic[key_res]:
            #------------------------------------------------------------------------------------------------
            # generate_cloudnet_products(date, 
            #                            PATH_CATE, 
            #                            PATH_CLOUDNET_LWC, 
            #                            PATH_CLOUDNET_IWC, 
            #                            PATH_CLOUDNET_DER)
            #------------------------------------------------------------------------------------------------
            processing_args.append((height, nchirp, date, key_res))
            process_cloud_data_parallel((height, nchirp, date, key_res))

    # end_time = time_module.time()
    # #------------------------------------------------------------------------------------------------
    # # Parallel execution using multiprocessing.Pool
    # # ------------------------------------------------------------------------------------------------
    # num_processes = 4  # You can adjust this as needed
    # # Start the timer for parallel execution
    # print("Starting parallel execution...")
    # start_time = time_module.time()
    # # Parallel execution using multiprocessing.Pool
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     pool.map(process_cloud_data_parallel, processing_args)
    # end_time = time_module.time()
    # print("Parallel execution finished.")
    # # ------------------------------------------------------------------------------------------------
    # # Calculate and print the execution time
    # execution_time = (end_time - start_time) / 60
    # print(f"Execution time: {execution_time:.2f} minutes")

# if __name__ == "__main__":
#     main()