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
import datetime
import pandas as pd
from pandas import DataFrame
import os
from functools import reduce
import operator
from typing import List, Tuple
import itertools
import xarray as xr
import seaborn as sns
import colorcet as cc  # Import the colorcet library
# import seaborn as sns
#-------------------------------------------------------------------------------------------------------
from cloudnetpy.products import generate_lwc
from cloudnetpy.products import generate_iwc
from cloudnetpy.products import generate_der
from cloudnetpy.products.der import Parameters
#-------------------------------------------------------------------------------------------------------
plt.ion()
plt.close('all')

#from cloud_classes import Intersection_products, HMmodel, Cloud_filters
#-------------------------------------------------------------------------------------------------------
# paths
#-------------------------------------------------------------------------------------------------------
PATH_CLASS        = '../../../data/classification/'
PATH_CATE         = '../../../data/categorize/'
PATH_RADAR        = '../../../data/radar/'
PATH_FIG          = '../figures/'
PATH_CLOUDNET_LWC = '../../tests/output_retrievals/lwc/'
PATH_CLOUDNET_IWC = '../../tests/output_retrievals/iwc/'
PATH_CLOUDNET_DER = '../../tests/output_retrievals/der/'
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
NBINS_CLOUD                     = 3
THRESHOLD_BELLOW                = 100 # [ m ]
THRESHOLD_ABOVE                 = 100 # [ m ]
FONT                            = {'family': 'serif',
                                   'color':  'black',
                                   'weight': 'normal',
                                   'size': 17,
                                   }
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

def compare_radar_chirp_configurations(start_date: datetime,
                                       end_date: datetime,
                                       database_intersection: List[datetime.datetime],
                                       path_radar: str) -> Tuple[List[datetime.datetime], List[np.ndarray], List[np.ndarray]]:
    """
    Compare radar chirp configurations for multiple dates from NetCDF files.

    Parameters:
        start_date (datetime): The start date from which the processing should begin.
        database_intersection (List[datetime]): List of dates to process radar data for.
        path_radar (str): The path where the NetCDF radar data files are located.

    Returns:
        Tuple[List[datetime], List[np.ndarray], List[np.ndarray]]: A tuple containing:
            - start_chirp (List[datetime]): List of dates when the radar chirp configuration changed.
            - chirp_zres (List[np.ndarray]): List of arrays containing the range resolutions for each date.
            - height (List[np.ndarray]): List of arrays containing the height data for each date.
    """
    # Initialize lists to store results
    start_chirp = [start_date]
    end_chirp = []
    chirp_zres = []
    height = []
    
    # Read and store the initial range resolution and chirp configuration
    radar = nc.Dataset(path_radar + start_date.strftime('%Y%m%d') + "_granada_rpg-fmcw-94.nc")
    range_resolution = np.round(radar['range_resolution'][:], 1)
    chirp_zres.append(range_resolution)
    height.append(radar['range'][:])
    
    # Loop through the dates in the intersection and compare chirp configurations
    for date in database_intersection:
        radar = nc.Dataset(path_radar + date.strftime('%Y%m%d') + "_granada_rpg-fmcw-94.nc")
        new_range_res = np.round(radar['range_resolution'][:], 1)
        
        # Check if the new range resolution is the same as the previous one
        if not np.array_equal(new_range_res, range_resolution):
            print("Range resolution is not the same for all dates")
            
            # Store the new date, chirp configuration, and height data if different
            start_chirp.append(date)
            end_chirp.append(date - timedelta(days=1))
            chirp_zres.append(new_range_res)
            height.append(radar['range'][:])
            range_resolution = new_range_res
    
    end_chirp.append(end_date)  
    return start_chirp, end_chirp, chirp_zres, height

# Define a function to handle serialization of individual columns
def handle_serialization(column):
    if isinstance(column, np.ma.MaskedArray):
        return column.filled(np.nan).tolist()
    return column

def plot_cloud_type(df_complete, df_cloud, df_ze, cloud_filter, name_title, z_min, z_max, color_names):
    
    # List of manually specified colors (replace these with your desired colors)
    manual_colors = ["#FFFFFF","#2077D6", "#0A2658", "#FFFF00", "#4EF6C1",\
                      "#D05BAC", "#BFBD8D", "#118527","#8794B3", "#DA6F49", "#88183E"]
    ncolors = len(color_names)
    # Create a figure and an array of subplots
    fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(15, 10))

    # Create a ListedColormap using the manual colors
    manual_cmap = plt.cm.colors.ListedColormap(manual_colors)

    # Plot the classification heatmap
    f0 = axs[0].pcolormesh(df_complete.index, 
                            df_complete.columns/1000, 
                            np.transpose(df_complete),
                            cmap=manual_cmap,
                            vmin=0,
                            vmax=ncolors)

    axs[0].set_ylabel(r'Height [km]')
    axs[0].grid()

    # Plot the mixed phase categories heatmap
    f1 = axs[1].pcolormesh(df_cloud.index, 
                          df_cloud.columns/1000, 
                          np.transpose(df_cloud),
                          cmap=manual_cmap,
                          vmin=0,
                          vmax=ncolors)
    
    i = 0
    for sublist in cloud_filter.cloud_base:
        axs[1].plot(np.tile(cloud_filter.time_cbt[i], len(sublist)), df_classification.columns[sublist]/1000, "*", color='black', markersize=3)
        i += 1
    i = 0
    for sublist in cloud_filter.cloud_top:
        axs[1].plot(np.tile(cloud_filter.time_cbt[i], len(sublist)), df_classification.columns[sublist]/1000, "*", color='red', markersize=3)
        i += 1

    axs[1].set_ylabel(r'Height [km]')
    axs[1].grid()
    
    # Create a colorbar with custom color patches and labels (vertical)
    colorbar1 = fig.colorbar(f1, ax=axs[0:2], ticks=[], orientation='vertical')

    # Adjust the position of colorbar and add color patches with names
    for idx, (color, name) in enumerate(zip(manual_cmap.colors, color_names)):
        rect = plt.Rectangle((0, idx), 1, 1, color=color)
        colorbar1.ax.add_patch(rect)
        colorbar1.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=10)


    # Plot the mixed phase Ze heatmap
    f2 = axs[2].pcolormesh(df_ze.index, 
                          df_ze.columns/1000, 
                          np.transpose(df_ze),
                          cmap='viridis',
                          vmin=-40,
                          vmax=10)

    axs[2].set_ylabel(r'Height [km]')
    axs[2].set_xlabel(r'Time [UTC]')
    colorbar2 = fig.colorbar(f2, ax=axs[2:3])
    colorbar2.set_label('Ze label')  # Add a label to the colorbar
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Customize x-axis limits based on hour_s and hour_e
    # axs[2].set_xlim([cloud_filter.classification.index.date[0] + pd.DateOffset(hour=hour_s), 
    #             cloud_filter.classification.index.date[0] + pd.DateOffset(hour=hour_e)])
    
    # Set y-axis limits
    axs[2].set_ylim([z_min, z_max])
    
    axs[2].grid()
    plt.suptitle(name_title)
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
        
class Cloud_filters:
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
                                   height_cloud_base: DataFrame,
                                   height_cloud_top: DataFrame,
                                   height_cloud_mean: DataFrame,
                                   geometric_cloud_thickness: DataFrame,
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
                        dzb = self.height[cloud_thickness[0]] - self.height[ibl]
                        #print(dzb)
                        if dzb <= dzb_max:
                            self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan
                            # layer_to_remove.append(j)
                        #self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan
                    if not np.isnan(iab):
                        dzt = self.height[iab] - self.height[cloud_thickness[-1]]
                        if dzt <= dzt_max:
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
    
    def cloud_categorization(self, var, with_specie=[]):
        ''' description '''
        
        # if np.size(with_specie)>0:
        #     conditions = []
        #     for elem in with_specie:
        #         conditions.append(self.classification == elem)
        #     merged_conditions = reduce(operator.or_, conditions)
        #     row_specie, col_specie = np.where(merged_conditions) 
     
        #     intersection            = np.intersect1d(self.row_cloud, row_specie) # Temporal index with in Dataframe with coexistence of cloud type and especie
        #     mask                    = np.full(self.classification.shape[0], False)
        #     mask[intersection]      = True
        #     self.classification.loc[~mask,:]=np.nan

        # if np.size(self.cloud)==1:
        #     final_classification = self.classification.where(self.classification == self.cloud, np.nan)
        #     final_var            = var.where(self.classification == self.cloud, np.nan)
            
        # else:
        conditions = []
        for elem in self.cloud+with_specie:
            conditions.append( self.classification == elem)
        merged_mask = reduce(operator.or_, conditions)
        final_classification = self.classification.where(merged_mask, np.nan)
        final_var            = var.where(merged_mask, np.nan)
            
        return final_classification, final_var
        
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
        
#--------------------------------------------------------------------------------------------------------
paths     = [PATH_RADAR, PATH_CATE, PATH_CLASS]
extension = '.nc'
database_intersection = common_prefix_of_filenames(paths, extension)
start_date        = min(database_intersection) # first date of database
end_date          = max(database_intersection) # last date of database

#TODO: should create a loop to iterate over all chirp configurations

chirp_ini, chirp_final, chirp_zres, chirp_height = compare_radar_chirp_configurations(start_date, end_date, database_intersection, PATH_RADAR)

number_chirp_config = len(chirp_ini)
for nchirp in range(number_chirp_config):
    height     = chirp_height[nchirp]
    zres       = chirp_zres[nchirp]
    start_date = chirp_ini[nchirp]
    end_date = chirp_final[nchirp]
    
    date_complete     = pd.date_range( start=start_date, end=end_date, freq='D') # datetime with all days between
                                                                                # start and end dates
    time_complete             = pd.date_range(start=start_date+timedelta(seconds=15),
                                    end=end_date + pd.Timedelta(days=1),
                                    freq='30S')
    # Create an empty list to store DataFrames
    dataframes_list_ze = []
    dataframes_list_vd = []
    hydrometeor_count         = pd.DataFrame(index=time_complete,
                                             columns=["Hydrometeors"])
    height_cloud_base         = pd.DataFrame(index=time_complete, 
                                    columns=["Liquid", "Ice", "Mixed-phase", "Pre-liquid", "Pre-mixed-phase"])
    height_cloud_top          = pd.DataFrame(index=time_complete,
                                    columns=["Liquid", "Ice", "Mixed-phase", "Pre-liquid", "Pre-mixed-phase"])
    height_cloud_mean         = pd.DataFrame(index=time_complete,
                                    columns=["Liquid", "Ice", "Mixed-phase", "Pre-liquid", "Pre-mixed-phase"])
    geometric_cloud_thickness = pd.DataFrame(index=time_complete,
                                    columns=["Liquid", "Ice", "Mixed-phase", "Pre-liquid", "Pre-mixed-phase"])
    liquid_water_path         = pd.DataFrame(index=time_complete,
                                            columns=["value"])
    ice_water_path            = pd.DataFrame(index=time_complete,
                                            columns=["value"])
    number_of_layers          = pd.DataFrame(index=time_complete,
                                    columns=["Liquid", "Ice", "Mixed-phase", "Pre-liquid", "Pre-mixed-phase"])
    hydrometeor_mask = pd.DataFrame(index=time_complete , columns=height)
    liquid_count_profile      = pd.DataFrame(index=time_complete , columns=height)
    ice_count_profile         = pd.DataFrame(index=time_complete , columns=height)

    npr_per_day = 24*3600/30 
    print("Start date:", start_date.date())
    print("End date:", end_date.date())

    #----------------------------------------------------------------------------------------------------
    # verification
    #----------------------------------------------------------------------------------------------------

    # date_ex = cloudnet_example.dates[28] # 29/04/2021
    # i = 20 case with diferent resolution 
    # NOTE: If working in a server without adm permission the following two lines should 
    # be discomented
    print("\nRemoving all cloudnet files of LWC and Reff from its directory...")
    os.system("rm "+PATH_CLOUDNET_LWC+"*lwc.nc") # remove all lwc files from lwc path 
    os.system("rm "+PATH_CLOUDNET_DER+"*der.nc") # remove all der files from der path
    os.system("rm "+PATH_CLOUDNET_IWC+"*iwc.nc") # remove all der files from der path
    print("\nAll file removed")
    date_test = [database_intersection.date[20]]
    #----------------------------------------------------------------------------------------------------
    # uncomment the following line for a complet time series analysis
    print("\nComputating cloud microphysics for liquid clouds")
    #for i, date in enumerate(cloudnet_example.dates):
    # and of course, comment the next line :)
    n = len(database_intersection.date)
    for ind_day, date in enumerate(date_test):
    # for ind_day, date in enumerate(time_database_intersection):
    #----------------------------------------------------------------------------------------------------
        print("\nFiles = %d/%d"%(ind_day+1, n))
        #------------------------------------------------------------------------------------------------
        # reading categorize and classification files 
        #------------------------------------------------------------------------------------------------
        categorize     = nc.Dataset(PATH_CATE+date.strftime('%Y%m%d')+"_granada_categorize.nc")
        classification = nc.Dataset(PATH_CLASS+date.strftime('%Y%m%d')+"_granada_classification.nc")
        radar          = xr.open_dataset(PATH_RADAR+date.strftime('%Y%m%d')+"_granada_rpg-fmcw-94.nc")
        #------------------------------------------------------------------------------------------------
        # cloudnet algorithm to generate netcdf files with liquid water content (lwc) and droplet 
        # effective radius (der)
        #------------------------------------------------------------------------------------------------
        generate_lwc(PATH_CATE+date.strftime('%Y%m%d')+"_granada_categorize.nc", 
                        PATH_CLOUDNET_LWC+date.strftime('%Y%m%d')+"_granada_"+'lwc.nc')
        generate_iwc(PATH_CATE+date.strftime('%Y%m%d')+"_granada_categorize.nc",
                        PATH_CLOUDNET_IWC+date.strftime('%Y%m%d')+"_granada_"+'iwc.nc')
        params = Parameters(2.0, 100.0e6, 200.0e6, 0.25, 0.1, 5.0e-3)
        generate_der(PATH_CATE+date.strftime('%Y%m%d')+"_granada_categorize.nc",
                        PATH_CLOUDNET_DER+date.strftime('%Y%m%d')+"_granada_"+'der.nc', 
                        parameters=params)
        
        # check if categorize and classification files have the same time resolution 
        if classification.dimensions['time'].size == categorize.dimensions['time'].size\
        and sum(categorize['time'][:] == classification['time'][:]) == categorize.dimensions['time'].size:
            
            time_auxiliary  = []
            for h in categorize['time']:
                time_auxiliary.append( date.strftime('%Y%m%d')\
                                        + ' ' + str(timedelta(hours=float(h))) )
            time = round_datetimeindex_to_seconds( pd.to_datetime(time_auxiliary, format='mixed', dayfirst=True) )
        else:
            print("Red flag: classification and categorize files with diferent time resolution - ",
                    time.date())
        
        tot_profile_number = time.shape[0]
        #------------------------------------------------------------------------------------------------
        # verification: plot time interval
        #------------------------------------------------------------------------------------------------
        # a1 = np.diff(classification['time'][:30])*3600
        # a2 = np.diff(categorize['time'][:30])*3600
        # plt.figure()
        # plt.plot(a1, '-o',label='classification')
        # plt.plot(a2, '-o', label='categorize')
        # plt.ylabel(r'$\Delta T\ [s]$')
        # plt.legend()
        # plt.ylim([25, 40])
        # plt.show()
        #------------------------------------------------------------------------------------------------
        # verification: plot one profile of reflectivity (comment this block if nedeed)
        #------------------------------------------------------------------------------------------------
        # t = 500
        # fig = plt.figure(figsize=[5, 8])
        # ax1 = plt.subplot()

        # colormesh = plt.plot(categorize['Z'][t,:], categorize['height'][:]/1000, marker='o')

        # xlabx = ax1.xaxis.get_label()
        # xlaby = ax1.yaxis.get_label()

        # xlabx.set_size(12)
        # xlaby.set_size(12)
        # ax1.set_ylabel(r'Height [km]')
        # ax1.set_xlabel(r'Reflectivity [dBZ]')
        # plt.show()
        # #------------------------------------------------------------------------------------------------
        # # verification: plot reflectivity time serie, cloud base and cloud top 
        # #------------------------------------------------------------------------------------------------
        # fig = plt.figure(figsize=[12, 6])
        # axs = plt.subplot()

        # colormesh = plt.pcolormesh(time, categorize['height'][:],
        #                             np.transpose(categorize['Z'][:]),
        #                             cmap='viridis',
        #                             shading='nearest')
        # p1        = plt.scatter(time, classification['cloud_base_height_amsl'][:],
        #                         s=1, c='black', alpha=.7, marker='*')
        # p2        = plt.scatter(time, classification['cloud_top_height_amsl'][:],
        #                         s=1, c='red', alpha=.7, marker='*')

        # xlabx = axs.xaxis.get_label()
        # xlaby = axs.yaxis.get_label()
        # cbar = plt.colorbar(colormesh)
        # cbar.set_label(r'Z [dBz]')
        # xlabx.set_size(12)
        # xlaby.set_size(12)
        # axs.set_ylabel(r'Height [m]')
        # axs.set_xlabel(r'Time [UTC]')
        # axs.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # #plt.savefig(PATH_FIG+cloudnet_filename[0][:8]+'_radar.png', dpi=400)
        # plt.show()
        #------------------------------------------------------------------------------------------------
        # verification: plot classification time serie
        #------------------------------------------------------------------------------------------------
        # fig = plt.figure(figsize=[12, 6])
        # axs = plt.subplot()

        # colormesh = plt.pcolormesh(time, classification['height'][:],
        #                             np.transpose(classification['target_classification'][:]),
        #                             cmap='tab10',
        #                             shading='nearest',
        #                             vmin=0,
        #                             vmax=10)

        # fig.colorbar(colormesh, ax=axs, ticks=list(range(11)))
        # #cbar.set_ticks([mn,md,mx])
        # axs.set_ylabel(r'Height [m]')
        # axs.set_xlabel(r'Time [UTC]')
        # axs.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # #plt.savefig(PATH_FIG+cloudnet_filename[0][:8]+'_radar.png', dpi=400)
        # plt.show()
        #------------------------------------------------------------------------------------------------
        # creating variables as dataframe with time series as the index
        # and altitude as columns 
        #------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------
        # reading lwc and der files generated by the last code block
        #------------------------------------------------------------------------------------------------
        cloudnet_lwc = nc.Dataset(PATH_CLOUDNET_LWC+date.strftime('%Y%m%d')+"_granada_"+'lwc.nc')
        cloudnet_iwc = nc.Dataset(PATH_CLOUDNET_IWC+date.strftime('%Y%m%d')+"_granada_"+'iwc.nc')
        cloudnet_der = nc.Dataset(PATH_CLOUDNET_DER+date.strftime('%Y%m%d')+"_granada_"+'der.nc')
        
        dataframes_list_ze.append(radar.Zh)
        dataframes_list_vd.append(radar.v)
        df_cloud_base     = pd.DataFrame(data   =classification['cloud_base_height_amsl'][:], 
                                            index  =time )
        df_cloud_top      = pd.DataFrame(data   =classification['cloud_top_height_amsl'][:],
                                            index  =time )
        df_reflectivity   = pd.DataFrame(data   =categorize['Z'][:],
                                            index  =time,
                                            columns=height)
        df_lwp            = pd.DataFrame(data  =categorize['lwp'][:], 
                                        index =time, 
                                        columns=["value"])

        df_classification = pd.DataFrame(data=classification['target_classification'][:],
                                            index  =time,
                                            columns=height)

        df_cloudnet_lwc   = pd.DataFrame(data   =1.0e3*cloudnet_lwc['lwc'][:],
                                            index  =time,
                                            columns= height) # g m^-3
        df_cloudnet_iwc   = pd.DataFrame(data   =1.0e3*cloudnet_iwc['iwc'][:],
                                            index  =time,
                                            columns= height) # g m^-3
        df_cloudnet_der   = pd.DataFrame(data   =1.0e6*cloudnet_der['der'][:],
                                            index  =time,
                                            columns=height) # um
        #------------------------------------------------------------------------------------------------
        tick_labels = ['Clear  sky', 
               'Droplets', 
               'Drizzle or rain', 
               'Drizzle & droplets', 
               'Ice', 
               'Ice & droplets', 
               'Melting ice', 
               'Melting & droplets', 
               'Aerosol',
               'Insect', 
               'Aerosol & insect']
        
        cloud_types = { "Liquid"          : [CLOUD_LIQUID,DRIZZLE_OR_RAIN_LIQUID_DROPLETS],
                          "Ice"             : [ICE_PARTICLES],
                          "Mixed-phase"     : [CLOUD_LIQUID,ICE_PARTICLES,ICE_WITH_SUP_WATER,\
                                               MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS],
                          "Pre-liquid"      : [CLOUD_LIQUID,DRIZZLE_OR_RAIN_LIQUID_DROPLETS],
                          "Pre-mixed-phase" : [ICE_PARTICLES,ICE_WITH_SUP_WATER,MELTING_ICE,\
                                                MELTING_ICE_LIQUID_DROPLETS]
                        }  
        
        targ_between_cloud = {  "Liquid"          : [CLEAR_SKY, AERO_NO_CLOUD, INSECT_NO_CLOUD,\
                                                     AERO_WITH_INSECT_NO_CLOUD],
                                "Ice"             : [CLEAR_SKY, CLOUD_LIQUID, AERO_NO_CLOUD,\
                                                        INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD],
                                "Mixed-phase"     : [CLEAR_SKY, DRIZZLE_OR_RAIN, AERO_NO_CLOUD,\
                                                        INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD],
                                "Pre-liquid"      : [CLEAR_SKY, AERO_NO_CLOUD, INSECT_NO_CLOUD,\
                                                        AERO_WITH_INSECT_NO_CLOUD],
                                "Pre-mixed-phase" : [CLEAR_SKY, DRIZZLE_OR_RAIN, AERO_NO_CLOUD,\
                                                        INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD]}
        
        targ_to_filter = {"Liquid"          : [DRIZZLE_OR_RAIN, ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE,\
                                                MELTING_ICE_LIQUID_DROPLETS],
                          "Ice"             : [CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS,\
                                                ICE_WITH_SUP_WATER, MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS],
                          "Mixed-phase"     : [DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS],
                          "Pre-liquid"      : [ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS],
                          "Pre-mixed-phase" : []}
        
        targ_to_get_bellow = {"Pre-liquid"      : [DRIZZLE_OR_RAIN],
                              "Pre-mixed-phase" : [CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS]}
        
        cloud_phase = ["single_phase", "single_phase", "mixed_phase", "single_phase", "mixed_phase", "single_phase"]
        #------------------------------------------------------------------------------------------------
        # CLOUD filters for calculations of cloud properties 
        # ------------------------------------------------------------------------------------------------
        for i, cloud in enumerate(cloud_types):
            classification_filter = Cloud_filters(df_classification.copy(), 
                                                    cloud_types[cloud], 
                                                    targ_between_cloud[cloud], 
                                                    NBINS_BETWEEN_CLOUD,
                                                    cloud_phase[i],
                                                    NBINS_CLOUD)
            for target in targ_to_filter[cloud]:
                classification_filter.filter_species(target, 400, 200)
            
            if cloud == "Pre-liquid" or cloud == "Pre-mixed-phase":
                classification_filter.get_specie_below(targ_to_get_bellow[cloud], 10)
                cloud_cat, cloud_ze = classification_filter.cloud_categorization(df_reflectivity.copy(),
                                                                                 targ_to_get_bellow[cloud])
            else:
                cloud_cat, cloud_ze = classification_filter.cloud_categorization(df_reflectivity.copy())
            
            classification_filter.calculate_cloud_properties(height_cloud_base,
                                                                height_cloud_top,
                                                                height_cloud_mean,
                                                                geometric_cloud_thickness,
                                                                cloud)
            number_of_layers.loc[classification_filter.time_cbt, cloud] = sublist_lengths(classification_filter.cloud_base)
            with sns.axes_style("whitegrid"):
                plot_cloud_type(df_classification, 
                            cloud_cat, 
                            cloud_ze, 
                            classification_filter, 
                            cloud, .1, 12., tick_labels)
                
        #------------------------------------------------------------------------------------------------
        # CLOUD statistics
        # ------------------------------------------------------------------------------------------------
        cloud_types = { "Liquid"          : [CLOUD_LIQUID,DRIZZLE_OR_RAIN,DRIZZLE_OR_RAIN_LIQUID_DROPLETS],
                        "Ice"             : [ICE_PARTICLES],
                        "Mixed_phase"     : [ICE_WITH_SUP_WATER,\
                                                MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS],
                        "Total"    : [CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS,\
                                                ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE,\
                                                      MELTING_ICE_LIQUID_DROPLETS]
                        }

        targ_between_cloud = [CLEAR_SKY, AERO_NO_CLOUD, INSECT_NO_CLOUD,\
                                                     AERO_WITH_INSECT_NO_CLOUD]
        nbins_between_cloud = 1
        ds_hydrometeor = xr.Dataset(coords={'time': time_complete, 'range': height})
        for cloud in cloud_types:
            classification_filter = Cloud_filters(df_classification.copy(), 
                                                cloud_types[cloud], 
                                                targ_between_cloud, 
                                                nbins_between_cloud)
            
            cloud_mask = classification_filter.cloud_mask()
            ds_hydrometeor[cloud] = xr.DataArray(cloud_mask, dims=('time', 'range'), coords={'time': time, 'range': height})
            # number_of_layers.loc[classification_filter.time_cbt, cloud] = sublist_lengths(classification_filter.cloud_base)
            # plot_cloud_mask(df_classification, cloud_mask, cloud, 300, 12000)

        # verification:  
        try:
            total_hydromet_sum = (ds_hydrometeor.Ice + ds_hydrometeor.Liquid\
              + ds_hydrometeor.Mixed_phase).sum(dim='time')
            
            if np.all(total_hydromet_sum == ds_hydrometeor.Total.sum(dim='time')):
                print("All cloud types are mutually exclusive")
            else:
                print("Cloud types are not mutually exclusive")
        except Exception as e:
            print("An error occurred:", e)
            
        liquid_water_path.loc[time, "value"] = df_lwp["value"]
        #------------------------------------------------------------------------------------------------
        # Analisis hydrometeors 
        #------------------------------------------------------------------------------------------------ 
        # targets_hydro     = [CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS, ICE_PARTICLES,\
        #                     ICE_WITH_SUP_WATER, MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS]

        # hydrometeor_count_profile.loc[time, :] = mask_target(df_classification, targets_hydro)

        # #------------------------------------------------------------------------------------------------
        # # test: ICE_CLOUDS filter 
        # #------------------------------------------------------------------------------------------------ 
        # name_title          = "ICE"
        # targets             = [ICE_PARTICLES]
        # targs_between_cloud = [CLEAR_SKY, CLOUD_LIQUID, AERO_NO_CLOUD, INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD]
        # classification_filter = Cloud_filters(df_classification.copy(), targets, targs_between_cloud, NBINS_BETWEEN_CLOUD)
        # classification_filter.filter_species(CLOUD_LIQUID, 400, 200)
        # classification_filter.filter_species(DRIZZLE_OR_RAIN, 400, 200)
        # classification_filter.filter_species(DRIZZLE_OR_RAIN_LIQUID_DROPLETS, 400, 200)
        # classification_filter.filter_species(ICE_WITH_SUP_WATER, 400, 200)
        # classification_filter.filter_species(MELTING_ICE, 400, 200)
        # classification_filter.filter_species(MELTING_ICE_LIQUID_DROPLETS, 400, 200)
        # cloud_cat, cloud_ze = classification_filter.cloud_categorization(df_reflectivity.copy())
        # #------------------------------------------------------------------------------------------------
        # # test: Analisis
        # #------------------------------------------------------------------------------------------------
        # classification_filter.calculate_cloud_properties(height_cloud_base,
        #                                                 height_cloud_top,
        #                                                 height_cloud_mean,
        #                                                 geometric_cloud_thickness,
        #                                                 "Ice")
        
        # ice_count_profile.loc[time, :] = mask_target(df_classification, targets)
        # number_of_layers.loc[classification_filter.time_cbt,"Ice"] = sublist_lengths(classification_filter.cloud_base)
        # #------------------------------------------------------------------------------------------------
        # # fig, axs = plt.subplots()
        # # axs.plot(ice_profile_number, ice_profile_number.index)
        # # axs.set_ylabel("z [m]")
        # # axs.set_xlabel("Number ice particle")
        # # plt.show()
        
        # # cloud_base_hist = classification_filter.height[flatten_list_with_itertools(classification_filter.cloud_base)]
        # # cloud_top_hist = classification_filter.height[flatten_list_with_itertools(classification_filter.cloud_top)]
        # # plt.figure(); plt.hist(cloud_base_hist, bins=20);plt.title("ICE CB");plt.show()
        # # plt.figure(); plt.hist(cloud_top_hist, bins=20);plt.title("ICE CT");plt.show()

        # # fig, ax = plt.subplots()
        # # h_m = []
        # # i=0
        # # for j in range(len(classification_filter.cloud_base)):
        # #     n_p  = len(classification_filter.cloud_base[j])
        # #     h_cb = classification_filter.height[classification_filter.cloud_base[j]]
        # #     h_ct = classification_filter.height[classification_filter.cloud_top[j]]
        # #     h_m.append((h_cb+h_ct)/2)
        # #     ax.plot( np.tile(classification_filter.time_cbt[i], n_p), h_m[j], "*b")
        # #     # ax.plot( np.tile(classification_filter.time_cbt[i], n_p-1), np.diff(h_m[j]), "-*")
        # #     # ax.plot( np.tile(classification_filter.time_cbt[i], n_p), df_classification.columns[classification_filter.cloud_base[j]], "*", color='black', markersize=3)
        # #     # ax.plot( np.tile(classification_filter.time_cbt[i], n_p), df_classification.columns[classification_filter.cloud_top[j]], "*", color='red', markersize=3)
        # #     i+=1
        # # plt.show()
        
        # # fig, ax = plt.subplots()
        # # ax.hist(flatten_list_with_itertools(h_m), 25)
        # # plt.show()

        # #------------------------------------------------------------------------------------------------  
        # # test: plot the result for ICE_CLOUDS filter 
        # #------------------------------------------------------------------------------------------------
        # # plot_cloud_type(df_classification, cloud_cat, cloud_ze, classification_filter, name_title, 0, 23, 300, 12000)
        # #------------------------------------------------------------------------------------------------
        # # test: LIQUID_CLOUDS filter 
        # #------------------------------------------------------------------------------------------------
        # name_title = "LIQUID CLOUDS"
        # targets             = [CLOUD_LIQUID,DRIZZLE_OR_RAIN_LIQUID_DROPLETS]
        # targs_between_cloud = [CLEAR_SKY, AERO_NO_CLOUD, INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD]
        # classification_filter = Cloud_filters(df_classification.copy(), targets , targs_between_cloud, NBINS_BETWEEN_CLOUD)
        # classification_filter.filter_species(DRIZZLE_OR_RAIN, 400, 200)
        # classification_filter.filter_species(ICE_PARTICLES, 400, 200)
        # classification_filter.filter_species(ICE_WITH_SUP_WATER, 400, 200)
        # classification_filter.filter_species(MELTING_ICE, 400, 200)
        # classification_filter.filter_species(MELTING_ICE_LIQUID_DROPLETS, 400, 200)
        # cloud_cat, cloud_ze= classification_filter.cloud_categorization(df_reflectivity.copy())
        # #------------------------------------------------------------------------------------------------
        # # test: Analisis
        # #------------------------------------------------------------------------------------------------
        # classification_filter.calculate_cloud_properties(height_cloud_base,
        #                                                 height_cloud_top,
        #                                                 height_cloud_mean,
        #                                                 geometric_cloud_thickness,
        #                                                 "Liquid")
        # liquid_count_profile.loc[time, :] = mask_target(df_classification, targets)
        # number_of_layers.loc[classification_filter.time_cbt,"Liquid"] = sublist_lengths(classification_filter.cloud_base)
        # #------------------------------------------------------------------------------------------------
        # # fig, axs = plt.subplots()
        # # axs.plot(liquid_profile_number, liquid_profile_number.index)
        # # axs.set_ylabel("z [m]")
        # # axs.set_xlabel("Number Liquid Particles")
        # # plt.show()
        
        # # plot_cloud_type(df_classification, cloud_cat, cloud_ze, classification_filter, name_title, 0, 23, 300, 12000)
        # #------------------------------------------------------------------------------------------------
        # # test: precipitating LIQUID_CLOUDS filter 
        # #------------------------------------------------------------------------------------------------
        # name_title = "PRECIPITATING LIQUID CLOUDS"
        # targets             = [CLOUD_LIQUID,DRIZZLE_OR_RAIN_LIQUID_DROPLETS]
        # targs_between_cloud = [CLEAR_SKY, AERO_NO_CLOUD, INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD]
        # classification_filter = Cloud_filters(df_classification.copy(), targets , targs_between_cloud, NBINS_BETWEEN_CLOUD)
        # classification_filter.filter_species(ICE_PARTICLES, 400, 200)
        # classification_filter.filter_species(ICE_WITH_SUP_WATER, 400, 200)
        # classification_filter.filter_species(MELTING_ICE, 400, 200)
        # classification_filter.filter_species(MELTING_ICE_LIQUID_DROPLETS, 400, 200)
        # classification_filter.get_specie_below([2], 10)
        # cloud_cat, cloud_ze= classification_filter.cloud_categorization(df_reflectivity.copy(), [DRIZZLE_OR_RAIN])
        # #------------------------------------------------------------------------------------------------
        # # test: Abalisys
        # #------------------------------------------------------------------------------------------------
        # classification_filter.calculate_cloud_properties(height_cloud_base,
        #                                                 height_cloud_top,
        #                                                 height_cloud_mean,
        #                                                 geometric_cloud_thickness,
        #                                                 "Pre-liquid")
        # number_of_layers.loc[classification_filter.time_cbt,"Pre-liquid"] = sublist_lengths(classification_filter.cloud_base)
        # #------------------------------------------------------------------------------------------------
        # # test: plot the result for ICE_CLOUDS filter 
        # #------------------------------------------------------------------------------------------------
        # # plot_cloud_type(df_classification, cloud_cat, cloud_ze, classification_filter, name_title, 0, 23, 300, 12000)
        # #------------------------------------------------------------------------------------------------
        # # test: mixed phase clouds 
        # #------------------------------------------------------------------------------------------------
        # name_title = "MIXED PHASE CLOUDS"
        # targets               = [CLOUD_LIQUID,ICE_PARTICLES,ICE_WITH_SUP_WATER,MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS]
        # targs_between_cloud   = [CLEAR_SKY, DRIZZLE_OR_RAIN, AERO_NO_CLOUD, INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD]
        # classification_filter = Cloud_filters(df_classification.copy(), targets, targs_between_cloud, NBINS_BETWEEN_CLOUD, "mixed_phase")
        # classification_filter.filter_species(DRIZZLE_OR_RAIN, 400, 200)
        # classification_filter.filter_species(DRIZZLE_OR_RAIN_LIQUID_DROPLETS, 400, 200)
        # cloud_cat, cloud_ze = classification_filter.cloud_categorization(df_reflectivity.copy())
        # #------------------------------------------------------------------------------------------------
        # # test: Analisis
        # #------------------------------------------------------------------------------------------------
        # classification_filter.calculate_cloud_properties(height_cloud_base,
        #                                                 height_cloud_top,
        #                                                 height_cloud_mean,
        #                                                 geometric_cloud_thickness,
        #                                                 "Mixed-phase")
        # number_of_layers.loc[classification_filter.time_cbt,"Mixed-phase"] = sublist_lengths(classification_filter.cloud_base)
        # #------------------------------------------------------------------------------------------------
        # # test: plot the result of mixed phase clouds categorization
        # #------------------------------------------------------------------------------------------------
        # plot_cloud_type(df_classification, cloud_cat, cloud_ze, classification_filter, name_title, 0, 23, 300, 12000)
        # #------------------------------------------------------------------------------------------------
        # # test: precipitating mixed phase clouds 
        # #------------------------------------------------------------------------------------------------
        # name_title = "PRECIPITATING MIXED PHASE CLOUDS"
        # targets               = [ICE_PARTICLES,ICE_WITH_SUP_WATER,MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS]
        # targs_between_cloud   = [CLEAR_SKY, DRIZZLE_OR_RAIN, AERO_NO_CLOUD, INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD]
        # classification_filter = Cloud_filters(df_classification.copy(), targets , targs_between_cloud, NBINS_BETWEEN_CLOUD, "mixed_phase")
        # classification_filter.get_specie_below([CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS], 10)
        # cloud_cat, cloud_ze= classification_filter.cloud_categorization(df_reflectivity.copy(), [CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS])
        # #------------------------------------------------------------------------------------------------
        # # test: Analisis
        # #------------------------------------------------------------------------------------------------
        # classification_filter.calculate_cloud_properties(height_cloud_base,
        #                                                 height_cloud_top,
        #                                                 height_cloud_mean,
        #                                                 geometric_cloud_thickness,
        #                                                 "Pre-mixed-phase")
        # number_of_layers.loc[classification_filter.time_cbt,"Pre-mixed-phase"] = sublist_lengths(classification_filter.cloud_base)
        # #------------------------------------------------------------------------------------------------
        # # test: plot the result of mixed phase clouds categorization
        # #------------------------------------------------------------------------------------------------
        # # plot_cloud_type(df_classification, cloud_cat, cloud_ze, classification_filter, name_title, 0, 23, 300, 12000)
        # #------------------------------------------------------------------------------------------------
        # liquid_water_path.loc[time, "value"] = df_lwp["value"]
    #------------------------------------------------------------------------------------------------
    # Save the data as a NetCDF file
    #------------------------------------------------------------------------------------------------
    concatenated_ze = xr.concat(dataframes_list_ze, dim='time')
    concatenated_vd = xr.concat(dataframes_list_vd, dim='time')
    ds_radar = xr.merge([concatenated_ze, concatenated_vd])
    folder_name = f"../../../processed_data/chirp_{nchirp}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    number_of_layers.index.name = 'time'
    ds_layers                   = xr.Dataset.from_dataframe(number_of_layers)
    liquid_water_path.index.name= 'time'
    ds_lwp                      = xr.Dataset.from_dataframe(liquid_water_path)

    # Save the dataset as a NetCDF file inside the folder
    for ds, name in zip([ds_layers, ds_hydrometeor, ds_lwp, ds_radar], 
                        ["number_of_layers", "hydrometeor", "lwp", "radar"]): 
        print(f"Saving {name}...")
        output_path = os.path.join(folder_name, f"chirp_{nchirp}_{name}.nc")
        ds.to_netcdf(output_path)
    # ------------------------------------------------------------------------------------------------
    height_cloud_base.index.name = 'time'
    height_cloud_top.index.name = 'time'
    height_cloud_mean.index.name = 'time'
    geometric_cloud_thickness.index.name = 'time'
    
    for df, name in zip([height_cloud_base, height_cloud_top, height_cloud_mean, geometric_cloud_thickness],
                    ["height_cloud_base", "height_cloud_top", "height_cloud_mean", "geometric_cloud_thickness"]):
        print(f"Saving {name}...")
        try:
            # Convert columns to serializable format
            df_serializable = df.applymap(handle_serialization)
            df_serializable.to_json(f"{folder_name}/chirp_{nchirp}_{name}.json", orient='index')
        except Exception as e:
            print(f"Error while saving {name}: {e}")
    #------------------------------------------------------------------------------------------------