import xarray as xr
from pdb import set_trace
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
# from sklearn.cluster import DBSCAN
# from sklearn.cluster import AgglomerativeClustering
# from scipy.spatial import cKDTree
from functools import reduce
import operator
import pandas as pd
from typing import List, Tuple, Dict, Any
import datetime
from phd_clouds.utils import groupSequence
import glob
import netCDF4 as nc
import os

from phd_clouds.constants import CLEAR_SKY, CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS, ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS, AERO_NO_CLOUD, INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD, CLASSIFICATION_TICK_LABELS

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
        
        conditions = []
        for elem in self.cloud:
            conditions.append(self.classification == elem)
        merged_conditions = reduce(operator.or_, conditions)
        self.row_cloud, self.col_cloud = np.where(merged_conditions)
  
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
        intersection            = np.intersect1d(self.row_cloud_unique, row_specie)

        if intersection.any():  # Specie could be ice, rain, aerosol etc....
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
                        delta_bin = cloud_thickness[0] - ibl
                        if delta_bin < dzb_max:
                            self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan
                    if not np.isnan(iab):
                     
                        delta_bin = iab - cloud_thickness[-1]
                        if delta_bin < dzt_max:
                            self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan

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





