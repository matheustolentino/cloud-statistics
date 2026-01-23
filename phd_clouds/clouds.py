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
from phd_clouds.utils import download_cloudnet_products, groupSequence
import glob
import netCDF4 as nc
import os
from phd_clouds.constants import CLEAR_SKY, CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS, ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS, AERO_NO_CLOUD, INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD, CLASSIFICATION_TICK_LABELS, GRANADA_ALTITUDE, CLOUD_VALUES, CLOUD_CATEGORY, CLOUD_COLORS
from collections import Counter
from scipy import ndimage
import matplotlib as mpl
from scipy.optimize import curve_fit

PATH_FIG_TEST     = '../../tests/figures/'

HYDROMET_VALUES = {"CLOUD_LIQUID": CLOUD_LIQUID,
               "DRIZZLE_OR_RAIN": DRIZZLE_OR_RAIN,
               "DRIZZLE_OR_RAIN_LIQUID_DROPLETS": DRIZZLE_OR_RAIN_LIQUID_DROPLETS,
               "ICE_PARTICLES": ICE_PARTICLES,
               "ICE_WITH_SUP_WATER": ICE_WITH_SUP_WATER,
               "MELTING_ICE": MELTING_ICE,
               "MELTING_ICE_LIQUID_DROPLETS": MELTING_ICE_LIQUID_DROPLETS}

HYDROMET_INT = [CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS,\
                                        ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE,\
                                                MELTING_ICE_LIQUID_DROPLETS]

NO_HYDRO_INT = [CLEAR_SKY, AERO_NO_CLOUD, INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD]

cloud_cmap = plt.cm.colors.ListedColormap(CLOUD_COLORS)

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
                 z,
                 v=8.7,
                 rw=1e6):

        self.cloud_lwp = lwp # liquid water path
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
        # return self.k_rv*( np.pi*self.rho_w*np.trapz(np.sqrt(self.radar_ze), self.cloud_z)/(48*self.cloud_lwp) )**(1./3.) * self.radar_ze**(1./6.)
        # convert Ze from dBz to mm^6 m^-3
        # adapted to xarray
        z_linear = 10**(self.radar_ze/10.)
        return self.k_rv*( np.pi*self.rho_w*np.sqrt(z_linear).fillna(0).integrate('height')/(48*self.cloud_lwp) )**(1./3.) * z_linear**(1./6.)

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
    
# Define the linear function
def linear_func(x, a, b):
    return a * x + b

class CloudProcessing:
    def __init__(self, path_classification = None,
                  path_microphys = None, 
                  path_categorize = None, 
                  path_radar = None, 
                  path_mwr = None, 
                  site = 'granada',
                  min_cloud_pixels = 100,
                  min_rain_pixels = 10):
        
        self.path_classification = path_classification
        self.path_microphys = path_microphys
        self.path_categorize = path_categorize
        self.path_radar = path_radar
        self.path_mwr = path_mwr
        self.site = site
        self.min_cloud_pixels = min_cloud_pixels
        self.min_rain_pixels = min_rain_pixels

        self.layer_type            = ["single_layer", "multi_layer"]
        self.cloud_structure       = ["cloud_base", "cloud_top", "cloud_thickness"]

    def get_filenames(self, dic_patterns):
        dic_files = {'classification': None, 'lwc': None, 'iwc': None, 'der': None, 'ier': None, 'categorize': None, 'radar': None, 'mwr':
                        None}
        if self.path_classification is not None:
            dic_files['classification'] = glob.glob(os.path.join(self.path_classification, dic_patterns['classification']))
        if self.path_microphys is not None:
            dic_files['lwc'] = glob.glob(os.path.join(self.path_microphys, dic_patterns['lwc']))
            dic_files['iwc'] = glob.glob(os.path.join(self.path_microphys, dic_patterns['iwc']))
            dic_files['der'] = glob.glob(os.path.join(self.path_microphys, dic_patterns['der']))
            dic_files['ier'] = glob.glob(os.path.join(self.path_microphys, dic_patterns['ier']))
        if self.path_categorize is not None:
            dic_files['categorize'] = glob.glob(os.path.join(self.path_categorize, dic_patterns['categorize']))
        if self.path_radar is not None:
            dic_files['radar'] = glob.glob(os.path.join(self.path_radar, dic_patterns['radar']))
        if self.path_mwr is not None:
            dic_files['mwr'] = glob.glob(os.path.join(self.path_mwr, dic_patterns['mwr']))
        self.filenames = dic_files
        
    def get_filenames_from_datastr(self, datastr):
        
        dic_files = {'classification': None, 'lwc': None, 'iwc': None, 'der': None, 'ier': None, 'categorize': None, 'radar': None, 'mwr':
                        None}
        if self.path_classification is not None:
            dic_files['classification'] = glob.glob(os.path.join(self.path_classification, f"{datastr}_{self.site}_classification.nc"))

        if self.path_microphys is not None:
            dic_files['lwc'] = glob.glob(os.path.join(self.path_microphys, f"{datastr}_{self.site}_lwc-scaled-adiabatic.nc"))
            dic_files['iwc'] = glob.glob(os.path.join(self.path_microphys, f"{datastr}_{self.site}_iwc-Z-T-method.nc"))
            # dic_files['der'] = glob.glob(os.path.join(self.path_microphys, f"{datastr}_{self.site}_der.nc"))
            # dic_files['ier'] = glob.glob(os.path.join(self.path_microphys, f"{datastr}_{self.site}_ier.nc"))

        if self.path_categorize is not None:
            dic_files['categorize'] = glob.glob(os.path.join(self.path_categorize, f"{datastr}_{self.site}_'categorize'.nc"))

        if self.path_radar is not None:
            dic_files['radar'] = glob.glob(os.path.join(self.path_radar, f"{datastr}_{self.site}_rpg-fmcw-94*.nc"))

        if self.path_mwr is not None:
            dic_files['mwr'] = glob.glob(os.path.join(self.path_mwr, f"{datastr}_{self.site}_hatpro*.nc"))

        self.filenames = dic_files

    def download_products(self, date_ini, date_end, path_output, products):
        for product in products:
            if product == 'microphysics':
                download_cloudnet_products(date_ini, date_end, path_output[product], product='lwc', site=self.site)
                download_cloudnet_products(date_ini, date_end, path_output[product], product='iwc', site=self.site)
                download_cloudnet_products(date_ini, date_end, path_output[product], product='der', site=self.site)
                download_cloudnet_products(date_ini, date_end, path_output[product], product='ier', site=self.site)
            else:
                download_cloudnet_products(date_ini, date_end, path_output[product], product=product, site=self.site)

    def load_lwc(self, chunk_data=False):
        for file in self.filenames['lwc']:
            if chunk_data:
                self.lwc = xr.open_dataset(file, engine='netcdf4', chunks={'time': -1})
            else:
                self.lwc = xr.open_dataset(file)
        
    def load_iwc(self, chunk_data=False):
        for file in self.filenames['iwc']:
            if chunk_data:
                self.iwc = xr.open_dataset(file, engine='netcdf4', chunks={'time': -1})
            else:
                self.iwc = xr.open_dataset(file)
            
    def load_der(self, chunk_data=False):
        for file in self.filenames['der']:
            if chunk_data:
                self.der = xr.open_dataset(file, engine='netcdf4', chunks={'time': -1})
            else:
                self.der = xr.open_dataset(file)
            
    def load_ier(self, chunk_data=False):
        for file in self.filenames['ier']:
            if chunk_data:
                self.ier = xr.open_dataset(file, engine='netcdf4', chunks={'time': -1})
            else:
                self.ier = xr.open_dataset(file)
            
    def load_classification(self, chunk_data=False, get_var=False):
        try:
            if self.filenames['classification']:
                if get_var:
                    if chunk_data:
                        dataset_list = [xr.open_dataset(file, engine='netcdf4', chunks={'time': -1})[get_var] for file in self.filenames['classification']]
                        self.classification = xr.concat(dataset_list, dim='time').sortby('time').chunk({'time': -1})
                    else:
                        dataset_list = [xr.open_dataset(file)[get_var] for file in self.filenames['classification']]
                        self.classification = xr.concat(dataset_list, dim='time').sortby('time')
                else:
                    if chunk_data:
                        dataset_list = [xr.open_dataset(file, engine='netcdf4', chunks={'time': -1}) for file in self.filenames['classification']]
                        self.classification = xr.concat(dataset_list, dim='time').sortby('time').chunk({'time': -1})
                    else:
                        dataset_list = [xr.open_dataset(file) for file in self.filenames['classification']]
                        self.classification = xr.concat(dataset_list, dim='time').sortby('time')
        except Exception as e:
            print(e)
            self.classification = None
            
    def load_categorize(self, chunk_data=False, get_var=False):
        try:
            if self.filenames['categorize']:
                if get_var:
                    if chunk_data:
                        dataset_list = [xr.open_dataset(file, engine='netcdf4', chunks={'time': -1})[get_var] for file in self.filenames['categorize']]
                        self.categorize = xr.concat(dataset_list, dim='time').sortby('time').chunk({'time': -1})
                    else:
                        dataset_list = [xr.open_dataset(file)[get_var] for file in self.filenames['categorize']]
                        self.categorize = xr.concat(dataset_list, dim='time').sortby('time')
                else:
                    if chunk_data:
                        dataset_list = [xr.open_dataset(file, engine='netcdf4', chunks={'time': -1}) for file in self.filenames['categorize']]
                        self.categorize = xr.concat(dataset_list, dim='time').sortby('time').chunk({'time': -1})
                    else:
                        dataset_list = [xr.open_dataset(file) for file in self.filenames['categorize']]
                        self.categorize = xr.concat(dataset_list, dim='time').sortby('time')
        except Exception as e:
            print(e)
            self.categorize = None
    
    def load_lwc_scaled_adiabatic(self, chunk_data=False, get_var=False):
        try:
            if self.filenames['lwc']:
                dataset_list = [xr.open_dataset(file)['lwc'] for file in self.filenames['lwc']]
                self.lwc     = xr.concat(dataset_list, dim='time').sortby('time')
        except Exception as e:
            print(e)
            self.lwc = None

    def load_iwc_Z_T_method(self, chunk_data=False, get_var=False):
        try:
            if self.filenames['iwc']:
                dataset_list = [xr.open_dataset(file)[['iwc','iwc_retrieval_status']] for file in self.filenames['iwc']]
                ds = xr.concat(dataset_list, dim='time').sortby('time')
                ds = ds['iwc'].where(~(ds[f'iwc_retrieval_status']==2))
                self.iwc = ds.copy()
        except Exception as e:
            print(e)
            self.iwc = None
            
    def load_radar(self, chunk_data=False, get_var=False):
        try:
            if self.filenames['radar']:
                if get_var:
                    if chunk_data:
                        dataset_list = [xr.open_dataset(file, engine='netcdf4', chunks={'time': -1})[get_var] for file in self.filenames['radar']]
                        self.radar = xr.concat(dataset_list, dim='time').sortby('time').chunk({'time': -1})
                    else:
                        dataset_list = [xr.open_dataset(file)[get_var] for file in self.filenames['radar']]
                        self.radar = xr.concat(dataset_list, dim='time').sortby('time')
                else:
                    if chunk_data:
                        dataset_list = [xr.open_dataset(file, engine='netcdf4', chunks={'time': -1}) for file in self.filenames['radar']]
                        self.radar = xr.concat(dataset_list, dim='time').sortby('time').chunk({'time': -1})
                    else:
                        dataset_list = [xr.open_dataset(file) for file in self.filenames['radar']]
                        self.radar = xr.concat(dataset_list, dim='time').sortby('time')  
        except Exception as e:
            print(e)
            self.radar = None

    def load_mwr(self, chunk_data=False, get_var=False):
        try: 
            if self.filenames['mwr']:
                if get_var:
                    if chunk_data:
                        dataset_list = [xr.open_dataset(file, engine='netcdf4', chunks={'time': -1})[get_var] for file in self.filenames['mwr']]
                        self.mwr = xr.concat(dataset_list, dim='time').sortby('time').chunk({'time': -1})
                    else:
                        dataset_list = [xr.open_dataset(file)[get_var] for file in self.filenames['mwr']]
                        self.mwr = xr.concat(dataset_list, dim='time').sortby('time')
                else:
                    if chunk_data:
                        dataset_list = [xr.open_dataset(file, engine='netcdf4', chunks={'time': -1}) for file in self.filenames['mwr']]
                        self.mwr = xr.concat(dataset_list, dim='time').sortby('time').chunk({'time': -1})
                    else:
                        dataset_list = [xr.open_dataset(file) for file in self.filenames['mwr']]
                        self.mwr = xr.concat(dataset_list, dim='time').sortby('time')
        except Exception as e:
            print(e)
            self.mwr = None
        
    def load_all_files(self, chunk_data=False):
        return self.load_lwc(chunk_data), self.load_iwc(chunk_data), self.load_der(chunk_data), self.load_ier(chunk_data), self.load_classification(chunk_data), self.load_categorize(chunk_data), self.load_radar(chunk_data), self.load_mwr(chunk_data)

    def initialize_time(self):
        self.time = self.classification.time

    def generate_cloud_mask(self):

        df_data = pd.DataFrame(self.classification['target_classification'].values, columns=self.classification['target_classification']['height'].values, index=self.classification['target_classification']['time'].values)
        classification_filter = CloudProcess(df_data,
                                             HYDROMET_INT,
                                             NO_HYDRO_INT,
                                             1)
         
        self.cloud_mask = classification_filter.cloud_mask().to_numpy(dtype=int)
    
    def dilate_and_label_clouds(self, cloud_mask, distance=2):
        """
        Dilate the cloud mask and label connected components.
        """
        # Create a binary structure with specified connectivity
        s = ndimage.generate_binary_structure(cloud_mask.ndim, connectivity=distance)

        # Dilate the binary array with the structuring element
        dilated_array = ndimage.binary_dilation(cloud_mask, structure=s)

        # Label connected components
        labels, num_features = ndimage.label(dilated_array)

        # Initialize a dictionary to store indices for each label
        indices_dict = {}

        # Iterate over each unique label
        for label_val in range(1, num_features+1):
            # Get indices where mask equals the current label
            indices = np.argwhere(labels == label_val)
            # Add indices to the dictionary
            indices_dict[label_val] = indices

        return indices_dict

    def classify_clusters(self):
        cloud_indx = {}
        indx_inside_cloud = {}
        cloud_composition = {}
        ds_classification = {}
        valid_clouds = []
        time_idx_valid_clouds = []
        time_idx_noise = np.zeros(self.time.shape[0])
        
        cloud_type = xr.Dataset(coords={'time': self.time})
        for var in CLOUD_CATEGORY[1:]:
            cloud_type[var] = xr.DataArray(data=np.zeros(self.time.shape), coords={'time': self.time}, dims='time')
        
        indices_dict = self.dilate_and_label_clouds(self.cloud_mask, distance=2)
        indices_dict_sorted = dict(sorted(indices_dict.items()))

        # Iterate over each label and calculate cloud composition
        for cloud_number, indx in indices_dict_sorted.items():
            mask_with_only_hydrometeor = self.cloud_mask[indx[:, 0], indx[:, 1]] == 1  # remove values that are not hydrometeors due dilatation
            indx = indx[mask_with_only_hydrometeor]
            cloud_indx[cloud_number] = indx

            hydro_cloud = self.classification.target_classification.values[indx[:, 0], indx[:, 1]]
            mask_withou_rain = hydro_cloud != DRIZZLE_OR_RAIN
            indx_inside_cloud[cloud_number] = indx[mask_withou_rain]
            n_pixels_rain = len(indx[~mask_withou_rain])
            hydro_inside_cloud = hydro_cloud[mask_withou_rain]
            n_pixels_inside_cloud = len(indx_inside_cloud[cloud_number])

            if n_pixels_inside_cloud == 0 or n_pixels_inside_cloud < self.min_cloud_pixels:
                ds_classification[cloud_number] = "Not Classified"
                idx_noise = np.unique(indx[:, 0])
                time_idx_noise[idx_noise] = 1
                cloud_type["Not Classified"][idx_noise] = 1
            else:
                valid_clouds.append(cloud_number)
                time_idx_valid_clouds.append(np.unique(indx_inside_cloud[cloud_number][:, 0]))
                hydromet_freq = {}
                for hydromet_name, hydromet_val in HYDROMET_VALUES.items():
                    count = np.count_nonzero(hydro_inside_cloud == hydromet_val)
                    hydromet_freq[hydromet_name] = 100 * (count / n_pixels_inside_cloud)
                    cloud_composition[cloud_number] = hydromet_freq

                liquid_percentage = cloud_composition[cloud_number]["CLOUD_LIQUID"] + cloud_composition[cloud_number]["DRIZZLE_OR_RAIN_LIQUID_DROPLETS"]

                if liquid_percentage > 70:
                    if n_pixels_rain > self.min_rain_pixels:
                        ds_classification[cloud_number] = "Precipitating-Liquid"
                        cloud_type["Precipitating-Liquid"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1
                    else:
                        ds_classification[cloud_number] = "Liquid"
                        cloud_type["Liquid"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1

                elif cloud_composition[cloud_number]["ICE_PARTICLES"] > 90 or \
                        (cloud_composition[cloud_number]["CLOUD_LIQUID"] + cloud_composition[cloud_number]["ICE_WITH_SUP_WATER"]) < 10:

                    if cloud_composition[cloud_number]["DRIZZLE_OR_RAIN_LIQUID_DROPLETS"] > 0:
                        drizzle_index = np.where(hydro_inside_cloud == DRIZZLE_OR_RAIN_LIQUID_DROPLETS)
                        # remove drizzle icdx from indx_inside_cloud
                        indx_inside_cloud[cloud_number] = np.delete(indx_inside_cloud[cloud_number], drizzle_index, axis=0)

                    if n_pixels_rain > self.min_rain_pixels:
                        ds_classification[cloud_number] = "Precipitating-Ice"
                        cloud_type["Precipitating-Ice"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1
                    else:
                        ds_classification[cloud_number] = "Ice"
                        cloud_type["Ice"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1

                elif n_pixels_rain > self.min_rain_pixels:
                    ds_classification[cloud_number] = "Precipitating-Mixed-Phase"
                    cloud_type["Precipitating-Mixed-Phase"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1

                    drizzle_index = np.where(hydro_inside_cloud == DRIZZLE_OR_RAIN_LIQUID_DROPLETS)
                    indx_inside_cloud[cloud_number] = np.delete(indx_inside_cloud[cloud_number], drizzle_index, axis=0)

                else:
                    ds_classification[cloud_number] = "Mixed-Phase"
                    cloud_type["Mixed-Phase"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1

        self.valid_cloud_number = valid_clouds
        self.time_idx_noise     = time_idx_noise
        self.cloud_type         = cloud_type
        self.cloud_classification = ds_classification
        self.indx_inside_cloud  = indx_inside_cloud
        self.time_idx_valid_clouds = time_idx_valid_clouds
        self.cloud_indx = cloud_indx # used by cluster classification product method

        # return cloud_indx, cloud_composition

    def classify_clusters_user_defined_thresholds(self, 
                                                  min_cloudp=100, 
                                                  min_rainp=10, 
                                                  min_liquid_percentage=70, 
                                                  min_ice_percentage=90):
        # print threshold values
        # print(f"min_cloudp={min_cloudp},\nmin_rainp={min_rainp},\nmin_liquid_percentage={min_liquid_percentage},\nmin_ice_percentage={min_ice_percentage}")
        cloud_indx = {}
        indx_inside_cloud = {}
        cloud_composition = {}
        ds_classification = {}
        valid_clouds = []
        time_idx_valid_clouds = []
        time_idx_noise = np.zeros(self.time.shape[0])
        
        cloud_type = xr.Dataset(coords={'time': self.time})
        for var in CLOUD_CATEGORY[1:]:
            cloud_type[var] = xr.DataArray(data=np.zeros(self.time.shape), coords={'time': self.time}, dims='time')
        
        indices_dict = self.dilate_and_label_clouds(self.cloud_mask, distance=2)
        indices_dict_sorted = dict(sorted(indices_dict.items()))

        # Iterate over each label and calculate cloud composition
        for cloud_number, indx in indices_dict_sorted.items():
            mask_with_only_hydrometeor = self.cloud_mask[indx[:, 0], indx[:, 1]] == 1  # remove values that are not hydrometeors due dilatation
            indx = indx[mask_with_only_hydrometeor]
            cloud_indx[cloud_number] = indx

            hydro_cloud = self.classification.target_classification.values[indx[:, 0], indx[:, 1]]
            mask_withou_rain = hydro_cloud != DRIZZLE_OR_RAIN
            indx_inside_cloud[cloud_number] = indx[mask_withou_rain]
            n_pixels_rain = len(indx[~mask_withou_rain])
            hydro_inside_cloud = hydro_cloud[mask_withou_rain]
            n_pixels_inside_cloud = len(indx_inside_cloud[cloud_number])

            if n_pixels_inside_cloud == 0 or n_pixels_inside_cloud < min_cloudp:
                ds_classification[cloud_number] = "Not Classified"
                idx_noise = np.unique(indx[:, 0])
                time_idx_noise[idx_noise] = 1
                cloud_type["Not Classified"][idx_noise] = 1
            else:
                valid_clouds.append(cloud_number)
                time_idx_valid_clouds.append(np.unique(indx_inside_cloud[cloud_number][:, 0]))
                hydromet_freq = {}
                for hydromet_name, hydromet_val in HYDROMET_VALUES.items():
                    count = np.count_nonzero(hydro_inside_cloud == hydromet_val)
                    hydromet_freq[hydromet_name] = 100 * (count / n_pixels_inside_cloud)
                    cloud_composition[cloud_number] = hydromet_freq

                liquid_percentage = cloud_composition[cloud_number]["CLOUD_LIQUID"] + cloud_composition[cloud_number]["DRIZZLE_OR_RAIN_LIQUID_DROPLETS"]

                if liquid_percentage > min_liquid_percentage:
                    if n_pixels_rain > min_rainp:
                        ds_classification[cloud_number] = "Precipitating-Liquid"
                        cloud_type["Precipitating-Liquid"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1
                    else:
                        ds_classification[cloud_number] = "Liquid"
                        cloud_type["Liquid"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1

                elif cloud_composition[cloud_number]["ICE_PARTICLES"] > min_ice_percentage or \
                        (cloud_composition[cloud_number]["CLOUD_LIQUID"] + cloud_composition[cloud_number]["ICE_WITH_SUP_WATER"]) < 10:

                    if cloud_composition[cloud_number]["DRIZZLE_OR_RAIN_LIQUID_DROPLETS"] > 0:
                        drizzle_index = np.where(hydro_inside_cloud == DRIZZLE_OR_RAIN_LIQUID_DROPLETS)
                        # remove drizzle icdx from indx_inside_cloud
                        indx_inside_cloud[cloud_number] = np.delete(indx_inside_cloud[cloud_number], drizzle_index, axis=0)

                    if n_pixels_rain > min_rainp:
                        ds_classification[cloud_number] = "Precipitating-Ice"
                        cloud_type["Precipitating-Ice"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1
                    else:
                        ds_classification[cloud_number] = "Ice"
                        cloud_type["Ice"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1

                elif n_pixels_rain > min_rainp:
                    ds_classification[cloud_number] = "Precipitating-Mixed-Phase"
                    cloud_type["Precipitating-Mixed-Phase"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1

                    drizzle_index = np.where(hydro_inside_cloud == DRIZZLE_OR_RAIN_LIQUID_DROPLETS)
                    indx_inside_cloud[cloud_number] = np.delete(indx_inside_cloud[cloud_number], drizzle_index, axis=0)

                else:
                    ds_classification[cloud_number] = "Mixed-Phase"
                    cloud_type["Mixed-Phase"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1

        self.valid_cloud_number = valid_clouds
        self.time_idx_noise     = time_idx_noise
        self.cloud_type         = cloud_type
        self.cloud_classification = ds_classification
        self.indx_inside_cloud  = indx_inside_cloud
        self.time_idx_valid_clouds = time_idx_valid_clouds
        self.cloud_indx = cloud_indx # used by cluster classification product method

    def analyze_clouds(self, filter_abl_ice_clouds=True, thick_threshold=700, base_threshold=4000):
        
        self.filter_abl_ice_clouds = filter_abl_ice_clouds
        self.time_idx_noise        = np.zeros(self.time.shape[0])

        for key in self.valid_cloud_number:
            indx = self.indx_inside_cloud[key]
            time_idx = np.unique(indx[:, 0])
            aux_time_single_layer = []

            for i in time_idx:
                mask_time = indx[:, 0] == i
                height_idx = indx[mask_time, :][:, 1]
                layers = groupSequence(height_idx, 1)
                n_layers = len(layers)

                if n_layers == 1:  # single layer
                    rain_above = self.israinabove(self.classification['target_classification'][i, layers[0][-1] + 1:].values)
                    if rain_above:
                        self.cloud_occurrence["multi_layer"][i] = 1
                        self.cloud_props["cloud_base"][i] = np.nan
                        self.cloud_props["cloud_top"][i] = np.nan
                        self.cloud_props["cloud_thickness"][i] = np.nan
                    else:
                        self.cloud_occurrence["single_layer"][i] = 1
                        aux_time_single_layer.append(i)
                        self.cloud_props["cloud_base"][i] = self.classification.height[layers[0][0]]
                        self.cloud_props["cloud_top"][i] = self.classification.height[layers[0][-1]]
                        self.cloud_props["cloud_thickness"][i] = self.cloud_props["cloud_top"][i] - self.cloud_props["cloud_base"][i]
                else:
                    self.cloud_occurrence["multi_layer"][i] = 1

            if self.filter_abl_ice_clouds:
                self.filter_ice_clouds(key, aux_time_single_layer, thick_threshold, base_threshold)

        self.correct_overlapping_clouds()

    def initialize_datasets(self):

        self.cloud_occurrence = xr.Dataset(coords={'time': self.time})
        self.cloud_props = xr.Dataset(coords={'time': self.time})

        for var in self.layer_type:
            self.cloud_occurrence[var] = xr.DataArray(data=np.zeros(self.time.shape),
                                                      coords={'time': self.time},
                                                      dims='time')
        for var in self.cloud_structure:
            self.cloud_props[var] = xr.DataArray(data=np.full(self.time.shape, np.nan),
                                                 coords={'time': self.time},
                                                 dims='time')
            
    def israinabove(self, hydromet_column: np.ndarray) -> bool:
        """
        Check if there is rain above the cloud column
        """
        rain_above = False
        for i in range(hydromet_column.shape[0]):
            if hydromet_column[i] == DRIZZLE_OR_RAIN:
                rain_above = True
                return rain_above

    def filter_ice_clouds(self, key, aux_time_single_layer, thick_threshold, base_threshold):
        if len(aux_time_single_layer) > 0:
            if self.cloud_classification[key] in ['Precipitating-Ice', 'Ice']:
                mean_cloud_base = self.cloud_props["cloud_base"][aux_time_single_layer].mean()
                mean_cloud_thickness = self.cloud_props["cloud_thickness"][aux_time_single_layer].mean()

                is_thick_enough = mean_cloud_thickness < thick_threshold
                is_base_enough = mean_cloud_base < base_threshold
                if is_thick_enough and is_base_enough:
                    self.cloud_occurrence["single_layer"][aux_time_single_layer] = 0
                    self.time_idx_noise[aux_time_single_layer] = 1
                    self.cloud_type["Not Classified"][aux_time_single_layer] = 1
                    self.cloud_props["cloud_base"][aux_time_single_layer] = np.nan
                    self.cloud_props["cloud_top"][aux_time_single_layer] = np.nan
                    self.cloud_props["cloud_thickness"][aux_time_single_layer] = np.nan

    def correct_overlapping_clouds(self):
        flatenned_time_idx_cloud = [item for sublist in self.time_idx_valid_clouds for item in sublist]
        count = Counter(flatenned_time_idx_cloud)
        repeated_values = [value for value, frequency in count.items() if frequency > 1]

        for index in repeated_values:
            self.cloud_occurrence["multi_layer"][index] = 1
            self.cloud_occurrence["single_layer"][index] = 0
            self.cloud_props["cloud_base"][index] = np.nan
            self.cloud_props["cloud_top"][index] = np.nan
            self.cloud_props["cloud_thickness"][index] = np.nan

        mask_clear_sky = self.cloud_occurrence.to_dataframe().sum(axis=1) == 0
        mask_clear_sky = mask_clear_sky.astype(float)
        self.cloud_occurrence["clear_sky"] = xr.DataArray(data=mask_clear_sky.values, coords={'time': self.time}, dims='time')
        self.cloud_occurrence["noise"] = xr.DataArray(data=self.time_idx_noise, coords={'time': self.time}, dims='time')

    def create_cluster_classification_product(self):
        array_cloud = np.zeros((self.time.shape[0], self.classification.height.shape[0]))

        for cloud_number, indx in self.cloud_indx.items():
            cloud_name = self.cloud_classification[cloud_number]
            integer = CLOUD_VALUES[cloud_name]
            if cloud_number in self.valid_cloud_number:
                array_cloud[indx[:, 0], indx[:, 1]] = integer
            else:
                array_cloud[indx[:, 0], indx[:, 1]] = CLOUD_VALUES["Not Classified"]

        ds_cloud = xr.Dataset(data_vars={'cloud_classification': (['time', 'height'], array_cloud)},
                              coords={'time': self.time, 'height': self.classification.height})
        
        ds_cloud.cloud_classification.attrs['long_name'] = 'Cloud classification'
        ds_cloud.cloud_classification.attrs['description'] = \
            'Cloud classification based on cloudnet target classification.\nHydrometeor cluster classification algorithm. \n0: No Cloud, \n1: Liquid, \n2: Liquid-Precipitable, \n3: Ice, \n4: Ice-Precipitable, \n5: Mixed-Phase, \n6: Mixed-Phase-Precipitable, \n7: Not Classified'
        
        self.cluster_classification  = ds_cloud

    def add_radar_lwp(self):
        self.cloud_props['lwp_radar'] = self.radar['lwp'].resample(time='30S', skipna=True).mean().interp(time=self.cloud_props.time, method='nearest')

    def add_mwr_lwp(self):
        if self.filenames['mwr']:
            self.cloud_props['lwp'] = self.mwr['lwp'].resample(time='30S', skipna=True).mean().interp(time=self.cloud_props.time, method='nearest')
            self.cloud_props['lwp'].attrs['source'] = self.mwr.attrs['source']
        else:
            self.cloud_props['lwp'] = None
            # self.cloud_props['lwp'] = self.categorize['lwp']  # post-process LWP
            # self.cloud_props['lwp'].attrs['source'] = self.categorize['lwp'].attrs['source'] + " (categorize)"

    def create_attenuation_mask(self, cloud_cmap, time_roll="10T", lwp_treshold=0.8, corr_treshold=-0.5, lwp2_treshold=1, make_plot=False):
        """
        Apply attenuation filter to cloud properties and categorize data.
        """
        # ---------------------------------------------------------------------------------------------
        # get data in cloud_props and categorize at each 5 minutes (moving mean of 5 min)
        # and take the correlation between cloud thickness and lwp:
        # https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.corr.html
        # ---------------------------------------------------------------------------------------------
        # Add cloud thickness and lwp in a pandas dataframe:
        df = self.cloud_props.cloud_thickness.to_dataframe()
        df['lwp_radar'] = self.cloud_props.lwp_radar.to_dataframe()
        #df.dropna(inplace=True)
        # Calculate the correlation between cloud thickness and lwp at each 5 minutes:
        corr_pd = df['cloud_thickness'].rolling(window=time_roll, min_periods=10, center=False).corr(df['lwp_radar'])
        # corr_pd.plot()
        # transform the correlation to xarray dataset:
        self.cloud_props['corr'] = xr.DataArray(corr_pd.values, coords={'time': self.cloud_props.time}, dims='time')
        self.cloud_props['corr'].attrs['long_name'] = f'Correlation (thickness & lwp)'
        self.cloud_props['corr'].attrs['description'] = 'Correlation between cloud thickness and lwp at each 5 minutes'
        # self.cloud_props['corr'].attrs['units'] = '1'
        # ---------------------------------------------------------------------------------------------
        # Coarsen the data
        # ---------------------------------------------------------------------------------------------
        mask_lwp     = self.cloud_props['lwp_radar'] > lwp_treshold
        mask_corr    = self.cloud_props['corr'] < corr_treshold
        self.mask_att = (mask_lwp & mask_corr) | (self.cloud_props['lwp_radar'] > lwp2_treshold)
        
        if make_plot:

            # -----------------------------------------------------------------------------
            # Plot Target Classification Product (T.C.P) time serie
            # -----------------------------------------------------------------------------
            manual_colors = ["#FFFFFF","#007CFF", "#0A2658", "#FFFF00", "#4EF6C1",\
                    "#D05BAC", "#BFBD8D", "#118527","#8794B3", "#DA6F49", "#88183E", "#DDDEDA"]
            ncolors = len(manual_colors)
            manual_cmap   = plt.cm.colors.ListedColormap(manual_colors)

            color_names = ['Clear  sky',
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

            fig = plt.figure(figsize=(12, 5))
            gs  = fig.add_gridspec(1, 2, width_ratios=[1, .02], wspace=0.05, hspace=0.08)
            ax2 = fig.add_subplot(gs[0, 0])
            pc = ax2.pcolormesh(self.classification['time'], 
                                self.classification['height']/1e3, 
                                self.classification['target_classification'].T, 
                                cmap=manual_cmap, 
                                vmin=0, 
                                vmax=ncolors)
            # countour = ax2.contour(categorize['model_time'], categorize['model_height'][:]/1e3,
            #                         categorize['temperature'][:].T - 273.15,
            #                         levels=[-40, -25, -10, 0, 5], colors='black', linewidths=0.5)
            # countour = ax1.contour(categorize['model_time'], categorize['model_height'][:]/1e3,
            #                 categorize['temperature'][:].T - 273.15,
            #                 levels=[-40, -25, -10, 0, 5], colors='black', linewidths=0.5)
            # countour.clabel(inline=True, fmt='%2.1f'+r'$^{\circ}$C', fontsize=12)
            ax2.set_ylabel('Height (km) a.s.l')
            ax2.set_xlabel('Time (UTC)')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            # ax2.set_ylim([categorize['height'][0]/1e3, categorize['height'][-1]/1e3])

            # ax2.set_title(f"Cloud Classification {date_str} - {site}")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            # ax2.set_xlim(data.time.values[0], data.time.values[-1])
            ax2.grid()
            cax = fig.add_subplot(gs[0, 1])
            cbar = plt.colorbar(pc, cax=cax, ticks=[], orientation='vertical')

            for idx, (color, name) in enumerate(zip(manual_cmap.colors, color_names)):
                rect = plt.Rectangle((0, idx), 1, 1, color=color)
                cbar.ax.add_patch(rect)
                cbar.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=14)
            ax2.set_facecolor('white')
            # ax2.set_ylim([0, ds_cloud['height'][-1]/1e3])
            # fig.savefig(PATH_FIG + f"{date_str}_cloud_classification.png", dpi=700, bbox_inches='tight')
            plt.show()
            # -----------------------------------------------------------------------------

            # fig = plt.figure(figsize=(15, 7))
            # gs = fig.add_gridspec(2, 2, width_ratios=[1, .02], height_ratios=[1, .6], wspace=0.02, hspace=0.2)

            # ax2 = fig.add_subplot(gs[1, 0])
            # self.cloud_props.cloud_thickness.plot(ax=ax2, color='b')

            # ax3 = ax2.twinx()
            # self.cloud_props.lwp_radar.plot(ax=ax3, color='r')
            # ax3.spines['right'].set_color('red')
            # ax3.axhline(y=lwp_treshold, color='r', linestyle='--')
            # ax3.axhline(y=lwp2_treshold, color='k', linestyle='--')

            # ax5 = ax2.twinx()
            # self.cloud_props.corr.plot(ax=ax5, color='g')
            # ax5.spines['right'].set_position(('outward', 50))  # Move the y-axis outward
            # ax5.spines['right'].set_color('green')
            # # plot horizontal line at zero correlation
            # ax5.axhline(y=corr_treshold, color='g', linestyle='--')
            
            # # self.mask_att = (mask_lwp & mask_corr)
            # ax1 = fig.add_subplot(gs[0, 0], sharex=ax2)

            # self.cluster_classification.cloud_classification.T.plot(ax=ax1, cmap=cloud_cmap, vmin=0, vmax=7, add_colorbar=False)
            # ax1.fill_between(self.cluster_classification.time.values, 0, 12000, where=self.mask_att, color='red', alpha=0.2) # removed areas due to attenuation
            # # self.cloud_props['corr'] = self.cloud_props['corr'].where(smask)
            # # self.cloud_props.cloud_base.plot(ax=ax1, color='r', linestyle='-')
            # # self.cloud_props.cloud_top.plot(ax=ax1, color='k', linestyle='-')

            # # Set up colorbar with ticks centered in each color
            # cbar_ax = fig.add_subplot(gs[0, 1])
            # n_colors = len(CLOUD_VALUES)
            # # Ticks at centers: 0.5, 1.5, ..., n-0.5
            # ticks = np.arange(n_colors) + 0.5
            # # Set boundaries so each color is centered
            # boundaries = np.arange(n_colors + 1)
            # norm = mpl.colors.BoundaryNorm(boundaries, cloud_cmap.N)
            # cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cloud_cmap, norm=norm, boundaries=boundaries, ticks=ticks, orientation='vertical')
            # cb.ax.set_yticklabels(list(CLOUD_VALUES.keys()))
            # cb.ax.tick_params(length=0)  # Remove tick lines for clarity
            # plt.show()
            
            fig, ax = plt.subplots(figsize=(12, 5))

            im = self.cluster_classification.cloud_classification.T.plot(ax=ax, cmap=cloud_cmap, vmin=0, vmax=7, add_colorbar=False)
            ax.fill_between(self.cluster_classification.time.values, 0, 12000, where=self.mask_att, color='red', alpha=0.2) # removed areas due to attenuation
            
            # plot top and base cloud properties
            self.cloud_props.cloud_base.plot(ax=ax, color='r', linestyle='-')
            self.cloud_props.cloud_top.plot(ax=ax, color='k', linestyle='-')

            ax.grid(True, linestyle=':')
            ax.set_ylim(0, 12000)
            ax.set_ylabel('Height (km) a.s.l')
            n_colors = len(CLOUD_VALUES)
            ticks = np.arange(n_colors) + 0.5
            boundaries = np.arange(n_colors + 1)
            norm = mpl.colors.BoundaryNorm(boundaries, cloud_cmap.N)
            cb = plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cloud_cmap),
                ax=ax,
                boundaries=boundaries,
                ticks=ticks,
                orientation='vertical',
                pad=0.02
            )
            cb.ax.set_yticklabels(list(CLOUD_VALUES.keys()))
            cb.ax.tick_params(length=0)
            plt.show()

    def calculate_fit_parameters(self, small_than=0.2, larger_than=0.8):
        """
        Calculate linear fit parameters for LWP (Radar x MWR) and save them in an xarray.
        """
        # Extract LWP and LWP radar values
        if not self.cloud_props['lwp'].any():
            # print("MWR LWP is not available.")
            self.fit_params = None
            self.fit_mask_small_radar_lwp = None
            self.fit_mask_larger_radar_lwp = None
            self.count_verification = None
            return
        
        lwp_values = self.cloud_props["lwp"].values
        lwp_radar_values = self.cloud_props["lwp_radar"].values

        # Remove NaN values
        mask = ~np.isnan(lwp_values) & ~np.isnan(lwp_radar_values)
        lwp_values = lwp_values[mask]
        lwp_radar_values = lwp_radar_values[mask]

        mask_small  = lwp_radar_values < small_than
        mask_larger = lwp_radar_values > larger_than
        
        count_small       = np.count_nonzero(mask_small)
        count_larger      = np.count_nonzero(mask_larger)

        if count_small < 20 or count_larger < 20:
            # print("Not enough data to perform linear fits")
            self.fit_params = None
            self.fit_mask_small_radar_lwp = None
            self.fit_mask_larger_radar_lwp = None
            return

        # relative_difference = (count_huge - count_in_between) / count_huge

        # Perform the linear fit for small values of LWP
        params_small, _ = curve_fit(linear_func, lwp_values[mask_small], lwp_radar_values[mask_small])
        
        # Calculate the Chi quadrado value:
        chi2_small = np.sum((lwp_radar_values[mask_small] - linear_func(lwp_values[mask_small], *params_small))**2 / lwp_radar_values[mask_small])
        
        # Perform the linear fit for larger values of LWP
        params_larger, _ = curve_fit(linear_func, lwp_values[mask_larger], lwp_radar_values[mask_larger])
        
        # Calculate the Chi quadrado value:
        chi2_larger = np.sum((lwp_radar_values[mask_larger] - linear_func(lwp_values[mask_larger], *params_larger))**2 / lwp_radar_values[mask_larger])

        # Save parameters in an xarray Dataset
        self.fit_params = xr.Dataset(
            {
            "params_small": (["param"], params_small),
            "params_larger": (["param"], params_larger),
            "chi2_small": (["date"], [chi2_small]),
            "chi2_larger": (["date"], [chi2_larger]),
            },
            coords={"param": ["slope", "intercept"], "date": [pd.to_datetime(self.time[0].values).normalize()]},
        )

        # Add descriptions
        self.fit_params["params_small"].attrs['description'] = f'Linear fit parameters (slope and intercept) for LWP < {small_than}'
        self.fit_params["params_larger"].attrs['description'] = f'Linear fit parameters (slope and intercept) for LWP > {larger_than}'
        self.fit_params["chi2_small"].attrs['description'] = f'Chi-squared value for the linear fit of small LWP < {small_than}'
        self.fit_params["chi2_larger"].attrs['description'] = f'Chi-squared value for the linear fit of larger LWP > {larger_than}'
        
        # Save mask values
        self.fit_mask_small_radar_lwp  = mask_small
        self.fit_mask_larger_radar_lwp = mask_larger

    def create_count_dataset_for_verification(self):
        """
        Create a dataset for count verification.
        """

        mask_huge = self.cloud_props["lwp_radar"].values > 1
        mask_in_between = (self.cloud_props["lwp_radar"].values > 0.8) & (self.cloud_props["lwp_radar"].values < 1)

        count_huge = np.count_nonzero(mask_huge)
        count_in_between = np.count_nonzero(mask_in_between)

        if count_huge < 20 or count_in_between < 20:
            # print("Not enough data to perform linear fits")
            self.count_verification = None
            return
        
        self.count_verification = xr.Dataset(
            {
                "count_high_lwp": count_huge,
                "count_in_between_lwp": count_in_between,
            },
            coords={"date": [pd.to_datetime(self.time[0].values).normalize()]},
        )
        # Add descriptions:
        self.count_verification["count_high_lwp"].attrs['description'] = 'Count of high LWP values (> 1 kg/m^2)'
        self.count_verification["count_in_between_lwp"].attrs['description'] = 'Count of LWP values between 0.8 and 1 kg/m^2'

    def plot_linear_fit_lwp(self, make_plot=False):
        """
        Plot linear fitting of LWP (Radar x MWR) and histogram of relative difference using saved fit parameters.
        """
        
        if self.fit_params is None:
            print("Fit parameters are not available. It does not interfere in the attenuation mask, where we use the radar LWP.")
            return
        
        # Extract LWP and LWP radar values
        lwp_values = self.cloud_props["lwp"].values
        lwp_radar_values = self.cloud_props["lwp_radar"].values

        # Remove NaN values
        mask = ~np.isnan(lwp_values) & ~np.isnan(lwp_radar_values)
        lwp_values = lwp_values[mask]
        lwp_radar_values = lwp_radar_values[mask]

        mask_small = self.fit_mask_small_radar_lwp
        mask_larger = self.fit_mask_larger_radar_lwp

        params_small = self.fit_params["params_small"].values
        params_larger = self.fit_params["params_larger"].values
        if make_plot:
            fig = plt.figure(figsize=(15, 9))
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 0.5], wspace=0.15, hspace=0.25)

            ax = fig.add_subplot(gs[0, 0])
            ax.scatter(lwp_values[mask_small], lwp_radar_values[mask_small], color='b', label='Data')
            ax.plot(lwp_values[mask_small], linear_func(lwp_values[mask_small], *params_small), color='r', label=f'Fit: y = {params_small[0]:.2f}x + {params_small[1]:.2f}, $\chi^2$ = {self.fit_params["chi2_small"].values[0]:.2f}')
            ax.set_xlabel("LWP (kg/m^2) - MWR")
            ax.set_ylabel("LWP (kg/m^2) - Radar")
            ax.grid()
            ax.legend()

            ax1 = fig.add_subplot(gs[0, 1])
            ax1.scatter(lwp_values[mask_larger], lwp_radar_values[mask_larger], color='b', label='Data')
            ax1.plot(lwp_values[mask_larger], linear_func(lwp_values[mask_larger], *params_larger), color='r', label=f'Fit: y = {params_larger[0]:.2f}x + {params_larger[1]:.2f}, $\chi^2$ = {self.fit_params["chi2_larger"].values[0]:.2f}')
            ax1.set_xlabel(f"LWP (kg/m^2) {self.cloud_props['lwp'].source}")
            ax1.set_ylabel("LWP (kg/m^2) - Radar")
            ax1.grid()
            ax1.legend()

            ax2 = fig.add_subplot(gs[1, :])
            self.cloud_props["lwp"].plot(ax=ax2, color='b', label=self.cloud_props["lwp"].attrs['source'])
            self.cloud_props["lwp_radar"].plot(ax=ax2, color='r', label='W-Band Radar')
            ax2.set_xlabel("Time (UTC)")
            ax2.set_ylabel("LWP (kg/m^2)")
            ax2.grid()
            ax2.legend()
            plt.show()

    def add_attenuation_to_clouds(self):
        self.cloud_type["Attenuation"] = xr.DataArray(data=self.mask_att.values, coords={'time': self.cloud_props.time}, dims='time')
        self.cloud_type["Attenuation"].attrs['long_name'] = 'Liquid attenuation QA flag'
        self.cloud_type["Attenuation"].attrs['description'] = 'Liquid attenuation QA flag. 1: High attenuation, 0: No attenuation'

        self.cloud_occurrence["attenuation"] = xr.DataArray(data=self.mask_att.values, coords={'time': self.cloud_props.time}, dims='time')
        self.cloud_occurrence["attenuation"].attrs['long_name'] = 'Liquid attenuation QA flag'
        self.cloud_occurrence["attenuation"].attrs['description'] = 'Liquid attenuation QA flag. 1: High attenuation, 0: No attenuation'
    
    def compute_cloud_lwp_iwp(self, make_plot=False):
        """
        Compute LWP and IWP from cloud base and top.
        """
        # 
        # integrate lwc and iwc between cloud base and cloud top
        #

        mask_boundaries = (self.lwc.height >= self.cloud_props["cloud_base"]) & (self.lwc.height <= self.cloud_props["cloud_top"])
        
        filtered_lwc    = self.lwc.where(mask_boundaries)
        filtered_iwc    = self.iwc.where(mask_boundaries)

        lwp_values = filtered_lwc.fillna(0.).integrate('height')
        iwp_values = filtered_iwc.fillna(0.).integrate('height')
        self.cloud_props['cloud_lwp'] = lwp_values
        self.cloud_props['cloud_iwp'] = iwp_values

        if make_plot:
            fig, ax = plt.subplots(figsize=(12, 5))
            im = self.cluster_classification.cloud_classification.T.plot(ax=ax, cmap=cloud_cmap, vmin=0, vmax=7, add_colorbar=False)
            ax.fill_between(self.cluster_classification.time.values, 0, 12000, where=self.mask_att, color='red', alpha=0.2) # removed areas due to attenuation
            # plot the base and top of the clouds
            self.cloud_props.cloud_base.plot(ax=ax, color='r', linestyle='-')
            self.cloud_props.cloud_top.plot(ax=ax, color='k', linestyle='-')

            ax.grid(True, linestyle=':')
            ax.set_ylim(0, 12000)
            ax.set_ylabel('Height (km) a.s.l')
            n_colors = len(CLOUD_VALUES)
            ticks = np.arange(n_colors) + 0.5
            boundaries = np.arange(n_colors + 1)
            norm = mpl.colors.BoundaryNorm(boundaries, cloud_cmap.N)
            cb = plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cloud_cmap),
                ax=ax,
                boundaries=boundaries,
                ticks=ticks,
                orientation='vertical',
                pad=0.02
            )
            cb.ax.set_yticklabels(list(CLOUD_VALUES.keys()))
            cb.ax.tick_params(length=0)
            plt.show()

            # Make the same plot but applying mask boundaries
            fig, ax = plt.subplots(figsize=(12, 5))
            im = self.cluster_classification.cloud_classification.where(mask_boundaries).T.plot(ax=ax, cmap=cloud_cmap, vmin=0, vmax=7, add_colorbar=False)
            ax.fill_between(self.cluster_classification.time.values, 0, 12000, where=self.mask_att, color='red', alpha=0.2) # removed areas due to attenuation
            
            # plot the base and top of the clouds
            self.cloud_props.cloud_base.plot(ax=ax, color='r', linestyle='-')
            self.cloud_props.cloud_top.plot(ax=ax, color='k', linestyle='-')
            ax.grid(True, linestyle=':')
            ax.set_ylim(0, 12000)
            ax.set_ylabel('Height (km) a.s.l')
            n_colors = len(CLOUD_VALUES)
            ticks = np.arange(n_colors) + 0.5
            boundaries = np.arange(n_colors + 1)
            norm = mpl.colors.BoundaryNorm(boundaries, cloud_cmap.N)
            cb = plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cloud_cmap),
                ax=ax,
                boundaries=boundaries,
                ticks=ticks,
                orientation='vertical',
                pad=0.02
            )
            cb.ax.set_yticklabels(list(CLOUD_VALUES.keys()))
            cb.ax.tick_params(length=0)
            plt.show()
        
            # Make the same plot above, but include a subplot with lwp and iwp below:
            fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            im = self.cluster_classification.cloud_classification.where(mask_boundaries).T.plot(ax=ax[0], cmap=cloud_cmap, vmin=0, vmax=7, add_colorbar=False)
            ax[0].fill_between(self.cluster_classification.time.values, 0, 12000, where=self.mask_att, color='red', alpha=0.2) # removed areas due to attenuation
            # plot the base and top of the clouds
            self.cloud_props.cloud_base.plot(ax=ax[0], color='r', linestyle='-')
            self.cloud_props.cloud_top.plot(ax=ax[0], color='k', linestyle='-')
            ax[0].grid(True, linestyle=':')
            ax[0].set_ylim(0, 12000)
            ax[0].set_ylabel('Height (km) a.s.l')
            n_colors = len(CLOUD_VALUES)
            ticks = np.arange(n_colors) + 0.5
            boundaries = np.arange(n_colors + 1)
            norm = mpl.colors.BoundaryNorm(boundaries, cloud_cmap.N)
            cb = plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cloud_cmap),
                ax=ax[0],
                boundaries=boundaries,
                ticks=ticks,    
                orientation='vertical',
                pad=0.02
            )
            cb.ax.set_yticklabels(list(CLOUD_VALUES.keys()))
            cb.ax.tick_params(length=0)
            # plot lwp and iwp in the second subplot
            self.cloud_props['cloud_lwp'].plot(ax=ax[1], color='b', label='Cloud LWP')
            self.cloud_props['cloud_iwp'].plot(ax=ax[1], color='g', label='Cloud IWP')
            ax[1].set_ylabel('kg/m²')
            ax[1].set_xlabel('Time (UTC)')
            ax[1].grid()
            # set log scale
            # ax[1].set_yscale('log')
            ax[1].legend()
            plt.show()
           
    def save_processed_data(self, path_to_save, products_to_store):
        date_str = self.time[0].dt.strftime('%Y%m%d').values.item()

        # Helper to create directory if it does not exist
        def ensure_dir(path):
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

        if products_to_store['cloud_occurrence']:
            dir_path = os.path.join(path_to_save, "cloud_occurrence")
            ensure_dir(dir_path)
            self.cloud_occurrence.to_netcdf(os.path.join(dir_path, f"{date_str}_cloud_occurrence_new.nc"))

        if products_to_store['cloud_props']:
            dir_path = os.path.join(path_to_save, "cloud_properties")
            ensure_dir(dir_path)
            self.cloud_props.to_netcdf(os.path.join(dir_path, f"{date_str}_cloud_props_new.nc"))

        if products_to_store['cloud_type']:
            dir_path = os.path.join(path_to_save, "cloud_type")
            ensure_dir(dir_path)
            self.cloud_type.to_netcdf(os.path.join(dir_path, f"{date_str}_cloud_type_new.nc"))

        # Save fit parameters:
        if products_to_store['fit_params'] and self.fit_params is not None:
            dir_path = os.path.join(path_to_save, "fit_parameters")
            ensure_dir(dir_path)
            self.fit_params.to_netcdf(os.path.join(dir_path, f"{date_str}_fit_params.nc"))

        if products_to_store['count_verification'] and self.count_verification is not None:
            dir_path = os.path.join(path_to_save, "fit_parameters")
            ensure_dir(dir_path)
            self.count_verification.to_netcdf(os.path.join(dir_path, f"{date_str}_count_verification.nc"))



