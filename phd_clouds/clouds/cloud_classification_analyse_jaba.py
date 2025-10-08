import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from pdb import set_trace
import pandas as pd
from phd_clouds.constants import GRANADA_ALTITUDE, SEASONS # in meters
from phd_clouds.utils import get_complete_time, assign_season, reading_dataset_chunking
import matplotlib.dates as mdates
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from IPython import get_ipython
import seaborn as sns
import matplotlib as mpl
# ipython = get_ipython()
# if ipython is not None:
#     ipython.run_line_magic('matplotlib', 'inline')
    
PATH_FIG          = '../../../cloud-statistics/figures/'
fontsize = 14
# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize

# plt.ion()
plt.close('all')
mpl.use('qtagg')

# Define a function to calculate skewness
def calculate_skewness(group):
    return xr.apply_ufunc(
        stats.skew, 
        group,
        input_core_dims=[['time']],
        kwargs={'axis': -1},
        vectorize=True
    )

load_data = True # Set to True if you want to load the data from the netCDF files
use_old_method = False # Set to True if you want to use the old method of cloud classification


filepath_cloud_occurence  = "/media/matheustolen/Seagate Basic/data_to_jaba/cloudnet/cloud_classification/cloud_occurrence" 
filepath_cloud_cloud_type = "/media/matheustolen/Seagate Basic/data_to_jaba/cloudnet/cloud_classification/cloud_type"
filepath_cloud_prop       = "/media/matheustolen/Seagate Basic/data_to_jaba/cloudnet/cloud_classification/cloud_properties"
filepath_categorize      = "/media/matheustolen/Seagate Basic/data_to_jaba/cloudnet/categorize"
filepath_microphys        = "/media/matheustolen/Seagate Basic/data_to_jaba/cloudnet/microphysics"

path_to_save_post_processed = "/media/matheustolen/Seagate Basic/data_to_jaba/cloudnet/cloud_classification/microphysics_stats" 

if load_data:
    # Get the list of netCDF files in the specified directory
    file_paths_occurence  = [os.path.join(filepath_cloud_occurence, file) for file in os.listdir(filepath_cloud_occurence) if file.endswith('new.nc')]
    file_paths_cloud_type = [os.path.join(filepath_cloud_cloud_type, file) for file in os.listdir(filepath_cloud_cloud_type) if file.endswith('new.nc')]
    file_paths_cloud_prop = [os.path.join(filepath_cloud_prop, file) for file in os.listdir(filepath_cloud_prop) if file.endswith('new.nc')]
    file_path_categorize  = [os.path.join(filepath_categorize, file) for file in os.listdir(filepath_categorize) if file.endswith('.nc')]
    file_paths_lwc        = [os.path.join(filepath_microphys, file) for file in os.listdir(filepath_microphys) if file.endswith("lwc-scaled-adiabatic.nc")]
    file_paths_iwc        = [os.path.join(filepath_microphys, file) for file in os.listdir(filepath_microphys) if file.endswith("iwc-Z-T-method.nc")]

    # Read the netCDF files into a list of xarray datasets
    print("Reading the netCDF files into a list of xarray datasets of new method...")
    datasets_occurence  = [xr.open_dataset(file_path) for file_path in file_paths_occurence]
    datasets_cloud_type = [xr.open_dataset(file_path) for file_path in file_paths_cloud_type]
    datasets_cloud_prop = [xr.open_dataset(file_path) for file_path in file_paths_cloud_prop]
    datasets_categorize = [xr.open_dataset(file_path)['lwp'] for file_path in file_path_categorize]
    
    print("Reading microphysics lwc files")
    datasets_microphys_lwc = [xr.open_dataset(file_path, engine='netcdf4', chunks={'time': -1})["lwc"].assign_coords(height=lambda ds: ds.height - GRANADA_ALTITUDE) for file_path in file_paths_lwc]

    print("Reading microphysics iwc files")
    datasets_microphys_iwc = [xr.open_dataset(file_path, engine='netcdf4', chunks={'time': -1})["iwc"].assign_coords(height=lambda ds: ds.height - GRANADA_ALTITUDE) for file_path in file_paths_iwc]
    
    #integrating the microphysics variable
    print("Integrating the microphysics variables...")
    datasets_lwp = [ds.fillna(0.).integrate('height') for ds in datasets_microphys_lwc]
    datasets_iwp = [ds.fillna(0.).integrate('height') for ds in datasets_microphys_iwc]
    
    chunked_lwp = xr.concat(datasets_lwp, dim='time').sortby('time')*1e3
    chunked_iwp = xr.concat(datasets_iwp, dim='time').sortby('time')*1e3
    chunked_lwp.chunk({'time': 'auto'})
    chunked_iwp.chunk({'time': 'auto'})

    print("Concatenating the datasets along the time coordinate...")
    # Concatenate the datasets_occurence along the time coordinate
    ds_cloud_occurence = xr.concat(datasets_occurence, dim='time').sortby('time')
    print("cloud occurence finished")
    ds_cloud_type = xr.concat(datasets_cloud_type, dim='time').sortby('time')
    ds_cloud_prop = xr.concat(datasets_cloud_prop, dim='time').sortby('time')
    ds_categorize = xr.concat(datasets_categorize, dim='time').sortby('time')
    

list_cloud_colors = ["#FFFFFF", "#007CFF", "blue", "cyan", "grey", "yellow", "orange", "magenta"]

# check if the variables are mutually exclusive
time_size           = ds_cloud_occurence['time'].size
sum_cloud_occurence = ds_cloud_occurence.sum(dim='time').to_array().values

# Check if the sum of the cloud occurence is equal to the size of the time dimension
if time_size == np.sum(sum_cloud_occurence[:-1]):
    print("The cloud occurence is mutually exclusive.")

monthly_cloud_occurence = ds_cloud_occurence.groupby('time.month').mean(dim='time')

# Datasets to save at the end of the code:
dict_datasets = {
    'atmosphere_classification': ds_cloud_occurence,
}

ds_cloud_prop = assign_season(ds_cloud_prop, SEASONS)


# -----------------------------------------------------------------------

ds_cloud_occurence   = ds_cloud_occurence.assign_coords(year=ds_cloud_occurence['time'].dt.year, month=ds_cloud_occurence['time'].dt.month)
count_available_data = ds_cloud_occurence['single_layer'].groupby('time.month').count(dim='time')
time_complete            = get_complete_time(ds_cloud_occurence, start_month=True)


single_layer_cloud_type = ds_cloud_type.where(ds_cloud_occurence['single_layer'], other=0)
monthly_single_layer_frequency = single_layer_cloud_type.groupby('time.month').mean(dim='time')
list_var_names = list(monthly_single_layer_frequency.data_vars)

dict_datasets['cloud_occurence_per_cloud_type'] = single_layer_cloud_type

# -----------------------------------------------------------------------
# Removing the attenuation from the data
# -----------------------------------------------------------------------
ds_cloud_prop = ds_cloud_prop.where(~ds_cloud_occurence['attenuation'].astype(bool)) # Remove the attenuation from the data
dict_datasets['cloud_macrophysics_corrected_by_attenuation'] = ds_cloud_prop.drop_vars(['lwp_radar', 'lwp', 'corr'])
# saving post processed data to the netcdf file
for names, datasets in dict_datasets.items():
    datasets.to_netcdf(f"{path_to_save_post_processed}/{names}.nc",
                        encoding={var: {'zlib': True, 'complevel': 5} for var in datasets.data_vars})
# -----------------------------------------------------------------------