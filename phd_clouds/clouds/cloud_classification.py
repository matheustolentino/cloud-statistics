import datetime
from phd_clouds.constants import CLEAR_SKY, CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS, ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS, AERO_NO_CLOUD, INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD, CLASSIFICATION_TICK_LABELS, GRANADA_ALTITUDE
from phd_clouds.constants import TARG_BET_HYDRO
from phd_clouds.clouds import CloudProcess
import xarray as xr
from phd_clouds.mwr_class import Mwr
from phd_clouds.utils import download_cloudnet_products, groupSequence
import matplotlib.pyplot as plt
from pdb import set_trace
import os
from collections import Counter
from scipy import ndimage
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.colors as colors
from phd_clouds.clouds import HMmodel
from pathlib import Path
from gfatpy.radar.rpg_nc import rpg
from gfatpy.radar.retrieve import retrieve
from rpgpy import rpg2nc
import glob
from cloudnetpy.products import generate_der
from cloudnetpy.products.der import Parameters
import matplotlib as mpl

fontsize = 14
# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize

plt.close('all')
mpl.use('qtagg')

PATH_FIG          = '../../../cloud-statistics/figures/'
PATH_FIG_TEST     = '../../tests/figures/'

def israinabove(hydromet_only_column: np.ndarray, rain_value: int) -> bool:
    """
    Check if there is rain above the cloud column
    """
    rain_above = False
    for i in range(hydromet_only_column.shape[0]):
        if hydromet_only_column[i] == rain_value:
            rain_above = True
            return rain_above

# hydromet colors
manual_colors = ["#FFFFFF","#007CFF", "#0A2658", "#FFFF00", "#4EF6C1",\
                    "#D05BAC", "#BFBD8D", "#118527","#8794B3", "#DA6F49", "#88183E", "#DDDEDA"]

ncolors = len(manual_colors)

manual_cmap   = plt.cm.colors.ListedColormap(manual_colors)

HYDROMET_VALUES = {"CLOUD_LIQUID": CLOUD_LIQUID,
               "DRIZZLE_OR_RAIN": DRIZZLE_OR_RAIN,
               "DRIZZLE_OR_RAIN_LIQUID_DROPLETS": DRIZZLE_OR_RAIN_LIQUID_DROPLETS,
               "ICE_PARTICLES": ICE_PARTICLES,
               "ICE_WITH_SUP_WATER": ICE_WITH_SUP_WATER,
               "MELTING_ICE": MELTING_ICE,
               "MELTING_ICE_LIQUID_DROPLETS": MELTING_ICE_LIQUID_DROPLETS}

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

cloud_category = ["No Cloud", "Liquid", "Liquid-Precipitable", "Ice", "Ice-Precipitable", "Mixed-Phase", "Mixed-Phase-Precipitable", "Noise"]
list_cloud_colors = ["#FFFFFF", "#007CFF", "blue", "cyan", "grey", "yellow", "orange", "magenta"]
cloud_cmap = plt.cm.colors.ListedColormap(list_cloud_colors)

# cloud_colors = {
#     "Liquid": "blue",
#     "Ice": "cyan",
#     "Liquid-Precipitable": "green",
#     "Ice-Precipitable": "yellow",
#     "Mixed-Phase": "orange",
#     "Mixed-Phase-Precipitable": "magenta"}

list_cloud_colors = ["blue", "cyan", "green", "yellow", "orange", "magenta"]

all_hydromet_values = [CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS,\
                                        ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE,\
                                                MELTING_ICE_LIQUID_DROPLETS]

path_save = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification" # Path to save the cloud classification files
path_microphys = "/home/matheustolen/Documentos/matheus_doctorado/output_retrievals" # Path with files already downloaded
path_categorize = "/media/matheustolen/Seagate Basic/cloudnet/categorize" # Path with files already downloaded
site = 'granada'
product_1 = 'classification'
product_2 = 'categorize'
product_3 = 'mwr'

process_all = False
save_files = False

make_plot = True
show_figure = False

filter_abl_ice_clouds = True

if filter_abl_ice_clouds:
    print("Filtering ice clouds in the ABL")
else:
    path_save = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification_without_filtering_ice_ABL"
    print("*****************Warning**********************\nNot filtering ice clouds in the ABL")


case_to_only_save = False
if not process_all:
    path_to_data = "../../tests/data" # Path to save cloudnet downloaded files

    date_ini = "2023-05-27"
    date_end = "2023-05-27"
    date_end_new = date_end.replace("-", "")

    download_cloudnet_products(date_ini, date_end, path_to_data, product=product_1, site=site)
    download_cloudnet_products(date_ini, date_end, path_to_data, product=product_2, site=site)
    download_cloudnet_products(date_ini, date_end, path_to_data, product=product_3, site=site)

    # Load the NetCDF file
    filenames = [f"{date_end_new}_{site}_{product_1}.nc"]
    lwc_filepaths = [f"{date_end_new}_{site}_lwc-scaled-adiabatic.nc"]
    iwc_filepaths = [f"{date_end_new}_{site}_iwc-Z-T-method.nc"]
    der_filepaths = [f"{date_end_new}_{site}_der.nc"]
    ier_filepaths = [f"{date_end_new}_{site}_ier.nc"]
else:
    path_to_data    = "/media/matheustolen/Seagate Basic/cloudnet/classification" # Path with files already downloaded
    lwc_filepaths = [file for file in os.listdir(path_microphys) if file.endswith("lwc-scaled-adiabatic.nc")]
    iwc_filepaths = [file for file in os.listdir(path_microphys) if file.endswith("iwc-Z-T-method")]
    der_filepaths = [file for file in os.listdir(path_microphys) if file.endswith("der.nc")] 
    ier_filepaths = [file for file in os.listdir(path_microphys) if file.endswith("ier.nc")]

    # Get a linst of filenames of products
    filenames = os.listdir(path_to_data)

for idx_file, file in enumerate(filenames):
    # Load the downloaded with xarray pandas:
    data         = xr.open_dataset(os.path.join(path_to_data, file))
    date_end_new = pd.to_datetime(data.time.values[-1]).strftime("%Y%m%d")
    
    if process_all:
        categorize   = xr.open_dataset(os.path.join(path_categorize, f"{date_end_new}_{site}_{product_2}.nc"))
    else: 
        categorize   = xr.open_dataset(os.path.join(path_to_data, f"{date_end_new}_{site}_{product_2}.nc"))
        mwr_files = glob.glob(os.path.join(path_to_data, f"{date_end_new}_{site}_hatpro*.nc"))
        if mwr_files:
            mwr = xr.open_mfdataset(mwr_files, combine='by_coords')
    # # Load the microphysical data
    # lwc = xr.open_dataset(os.path.join(path_microphys, lwc_filepaths[idx_file]))
    # iwc = xr.open_dataset(os.path.join(path_microphys, iwc_filepaths[idx_file]))
    # der = xr.open_dataset(os.path.join(path_microphys, der_filepaths[idx_file]))
    # ier = xr.open_dataset(os.path.join(path_microphys, ier_filepaths[idx_file]))

    # # der_new = der['der'].where(data['target_classification'] == CLOUD_LIQUID)
    # # ier_new = ier['ier'].where(data['target_classification'] == ICE_PARTICLES)
    # new_der = der.copy()
    # new_ier = ier.copy()
    # ----------------------------------------------------------------------------------------------------------------------
    # generating the cloud mask
    # ----------------------------------------------------------------------------------------------------------------------
    # Transform the target classification into a pandas DataFrame to cloud mask generation
    df_data = pd.DataFrame(data['target_classification'].values, columns=data['target_classification']['height'].values, index=data['target_classification']['time'].values)

    classification_filter = CloudProcess(df_data,
                                         all_hydromet_values,
                                         TARG_BET_HYDRO,
                                         1)

    # cloud_mask = xr.open_dataset("./ds_mask_reindexed.nc")
    cloud_mask          = classification_filter.cloud_mask().to_numpy(dtype=int)
    cloud_classification = data['target_classification']
    # # Dilatation example:
    # a = np.zeros((10, 10))
    # a[4, 4] = 1
    # print(a)
    # d=3
    # # s = ndimage.generate_binary_structure(a.ndim, connectivity=d)
    # s = np.zeros((5, 5))
    # s[2, 2:] = 1
    # s[2, :2] = 1
    # s[2:, 2] = 1
    # s[:2, 2] = 1

    # dilated_array = ndimage.binary_dilation(a, structure=s).astype(a.dtype)
    # print(dilated_array)
    # Set the desired connectivity distance
    distance = 2
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


    cloud_indx        = {}
    indx_inside_cloud = {}
    cloud_composition = {}
    ds_classification = {}
    valid_clouds      = []
    time_idx_valid_clouds = []
    time_idx_noise        = np.zeros(data.time.shape[0])
    cloud_min_pixels  = 100
    min_rain_pixels   = 10

    indices_dict_sorted = dict(sorted(indices_dict.items()))

    cloud_type= xr.Dataset(coords={'time': data.time})
    for var in cloud_category[1:]:
        cloud_type[var] = xr.DataArray(data=np.zeros(data.time.shape), coords={'time': data.time}, dims='time')

    # Iterate over each label and calculate cloud composition
    for cloud_number, indx in indices_dict_sorted.items():

        mask_with_only_hydrometeor = cloud_mask[indx[:, 0], indx[:, 1]] == 1 # remove values that are not hydrometeors due dilatation
        indx = indx[mask_with_only_hydrometeor]
        cloud_indx[cloud_number] = indx

        # set_trace()
        hydro_cloud       = cloud_classification.values[indx[:, 0], indx[:, 1]]
        mask_withou_rain  = hydro_cloud != DRIZZLE_OR_RAIN
        indx_inside_cloud[cloud_number] = indx[mask_withou_rain]
        n_pixels_rain     = len(indx[~mask_withou_rain])
        hydro_inside_cloud = hydro_cloud[mask_withou_rain]
        n_pixels_inside_cloud = len(indx_inside_cloud[cloud_number])

        if n_pixels_inside_cloud == 0 or n_pixels_inside_cloud < cloud_min_pixels:  # Only rain
            ds_classification[cloud_number] = "Noise"
            idx_noise = np.unique(indx[:, 0])
            time_idx_noise[idx_noise] = 1
            cloud_type["Noise"][idx_noise] = 1
        
        # if n_pixels_inside_cloud > cloud_min_pixels:
        #     valid_clouds.append(cloud_number)
        #     time_idx_valid_clouds.append(np.unique(indx_inside_cloud[cloud_number][:, 0]))
        # else:
        #     ds_classification[cloud_number] = "Noise"
        #     idx_noise = np.unique(indx_inside_cloud[cloud_number][:, 0])
        #     time_idx_noise[idx_noise]      = 1
        #     cloud_type["Noise"][idx_noise] = 1
        
        else:
            valid_clouds.append(cloud_number)
            time_idx_valid_clouds.append(np.unique(indx_inside_cloud[cloud_number][:, 0]))
            hydromet_freq = {}
            for hydromet_name, hydromet_val in HYDROMET_VALUES.items():
                count = np.count_nonzero(hydro_inside_cloud == hydromet_val)
                hydromet_freq[hydromet_name]  = 100*(count/n_pixels_inside_cloud)
                cloud_composition[cloud_number] = hydromet_freq

            liquid_percentage = cloud_composition[cloud_number]["CLOUD_LIQUID"]+cloud_composition[cloud_number]["DRIZZLE_OR_RAIN_LIQUID_DROPLETS"]

            if liquid_percentage > 70:
                if n_pixels_rain > min_rain_pixels:
                    ds_classification[cloud_number] = "Liquid-Precipitable"
                    cloud_type["Liquid-Precipitable"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1
                else:
                    ds_classification[cloud_number] = "Liquid"
                    cloud_type["Liquid"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1

            elif cloud_composition[cloud_number]["ICE_PARTICLES"] > 90 or \
                (cloud_composition[cloud_number]["CLOUD_LIQUID"] + cloud_composition[cloud_number]["ICE_WITH_SUP_WATER"]) < 10:

                if cloud_composition[cloud_number]["DRIZZLE_OR_RAIN_LIQUID_DROPLETS"] > 0:
                    drizzle_index = np.where(hydro_inside_cloud == DRIZZLE_OR_RAIN_LIQUID_DROPLETS)
                    # remove drizzle icdx from indx_inside_cloud
                    indx_inside_cloud[cloud_number] = np.delete(indx_inside_cloud[cloud_number], drizzle_index, axis=0)

                if n_pixels_rain > min_rain_pixels:
                    # set_trace()
                    ds_classification[cloud_number] = "Ice-Precipitable"
                    cloud_type["Ice-Precipitable"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1
                else:
                    ds_classification[cloud_number] = "Ice"
                    cloud_type["Ice"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1

            # elif cloud_composition[cloud_number]["MELTING_ICE"]  + cloud_composition[cloud_number]["MELTING_ICE_LIQUID_DROPLETS"] > 0:

            #     if cloud_composition[cloud_number]["ICE_PARTICLES"] > cloud_composition[cloud_number]["DRIZZLE_OR_RAIN_LIQUID_DROPLETS"]:
            #         ds_classification[cloud_number] = "Ice-Precipitable"

            elif n_pixels_rain > min_rain_pixels:
                ds_classification[cloud_number] = "Mixed-Phase-Precipitable"
                cloud_type["Mixed-Phase-Precipitable"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1

                drizzle_index = np.where(hydro_inside_cloud == DRIZZLE_OR_RAIN_LIQUID_DROPLETS)
                indx_inside_cloud[cloud_number] = np.delete(indx_inside_cloud[cloud_number], drizzle_index, axis=0)

            else:
                ds_classification[cloud_number] = "Mixed-Phase"
                cloud_type["Mixed-Phase"][np.unique(indx_inside_cloud[cloud_number][:, 0])] = 1



    # Print number of valid clouds
    # print(f"Valid clouds: {cloud_composition.keys()}")

    cloud_status = ["single_layer", "multi_layer"]
    cloud_occurrence = xr.Dataset(coords={'time': data.time})
    # Assign values to cloud occurrence for each cloud variable
    for var in cloud_status:
        cloud_occurrence[var] = xr.DataArray(data=np.zeros(data.time.shape), coords={'time': data.time}, dims='time')

    cloud_var = ["cloud_base", "cloud_top", "cloud_thickness"]
    cloud_props = xr.Dataset(coords={'time': data.time})
    # Assign values to cloud properties for each cloud variable
    for var in cloud_var:
        cloud_props[var] = xr.DataArray(data=np.full(data.time.shape, np.nan), coords={'time': data.time}, dims='time')
    
    time_range_start = "06:00"
    time_range_end   = "18:00"
    thick_threshold = 700
    base_threshold = 4000
    previous_single_layer = 0
    # Iterating inside each valid cloud group:
    for key in valid_clouds:
        indx     =  indx_inside_cloud[key]
        time_idx = np.unique(indx[:, 0])
        aux_time_single_layer = []
        # base_cloud_indx = []
        # top_cloud_indx  = []
        # for i in time_idx:
        #     mask_time = indx[:, 0] == i
        #     first_column_indx = indx[mask_time, 1].min()
        #     last_column_indx = indx[mask_time, 1].max()
        #     base_cloud_indx.append(first_column_indx)
        #     top_cloud_indx.append(last_column_indx)
        
        for i in time_idx:
            mask_time             = indx[:, 0] == i
            height_idx            = indx[mask_time, :][:, 1]
            layers                = groupSequence(height_idx, 1)
            n_layers              = len(layers)

            if n_layers == 1: #single layer

                rain_above = israinabove(data['target_classification'][i, layers[0][-1]+1:].values, DRIZZLE_OR_RAIN)
                if rain_above:
                    cloud_occurrence["multi_layer"][i] = 1
                    cloud_props["cloud_base"][i] = np.nan
                    cloud_props["cloud_top"][i]  = np.nan
                    cloud_props["cloud_thickness"][i] = np.nan
                else:
                    cloud_occurrence["single_layer"][i] = 1
                    aux_time_single_layer.append(i)
                    cloud_props["cloud_base"][i] = data.height[layers[0][0]]
                    cloud_props["cloud_top"][i]  = data.height[layers[0][-1]]
                    # cloud_props["cloud_base"][i] = data['cloud_base_height_amsl'][i]
                    # cloud_props["cloud_top"][i]  = data['cloud_top_height_amsl'][i]
                    cloud_props["cloud_thickness"][i] = cloud_props["cloud_top"][i] - cloud_props["cloud_base"][i]  
            else:
                cloud_occurrence["multi_layer"][i] = 1
                # cloud_type[i] = "ML"
        if filter_abl_ice_clouds:
            if len(aux_time_single_layer)>0:
                if ds_classification[key] == 'Ice-Precipitable' or ds_classification[key] == 'Ice':
                    # breakpoint()
                    mean_cloud_base = cloud_props["cloud_base"][aux_time_single_layer].mean()
                    mean_cloud_thickness = cloud_props["cloud_thickness"][aux_time_single_layer].mean()

                    # time_component = data.time[time_idx].dt.strftime('%H:%M')
                    # is_within_range = (time_component >= time_range_start) & (time_component <= time_range_end)
                    is_thick_enough = mean_cloud_thickness < thick_threshold
                    is_base_enough  = mean_cloud_base < base_threshold
                    if is_thick_enough and is_base_enough:
                        cloud_occurrence["single_layer"][aux_time_single_layer] = 0
                        time_idx_noise[aux_time_single_layer] = 1
                        cloud_props["cloud_base"][aux_time_single_layer] = np.nan
                        cloud_props["cloud_top"][aux_time_single_layer]  = np.nan
                        cloud_props["cloud_thickness"][aux_time_single_layer] = np.nan
                        case_to_only_save = True 

                        # Correct classification for single layer
        # if n_layers == 1:

        #     if previous_single_layer == 0:
        #         # Initialize intersection and time for single layer
        #         previous_time         = i
        #         intersection          = layers[0]
        #         previous_single_layer = 1
        #     else:
        #         current_time = time_idx[0]
        #         diff_time    = current_time - previous_time
        #         intersection = np.intersect1d(intersection, layers[0])
        #         previous_time = time_idx[-1]

    # Overlap between clouds
    # 1) find time index of noise
    # 2) find time index of clouds -> identify if there is overlap (multilayer)
    flatenned_time_idx_cloud = [item for sublist in time_idx_valid_clouds for item in sublist]
    count = Counter(flatenned_time_idx_cloud)
    repeated_values = [value for value, frequency in count.items() if frequency > 1]

    # # Correction due to overlapping clouds
    # time_idx_clouds = [np.unique(indx[:, 0]) for indx in cloud_indx.values() if indx.shape[0] > 1]
    # # Flatten the list of sublists
    # flattened_list  = [item for sublist in time_idx_clouds for item in sublist]

    # # Count occurrences of each value
    # count = Counter(flattened_list)
    # # Get values that appear more than once
    # repeated_values = [value for value, frequency in count.items() if frequency > 1]

    # Correction due to overlapping clouds
    # Update cloud occurrence for repeated indices
    for index in repeated_values:
        cloud_occurrence["multi_layer"][index] = 1
        cloud_occurrence["single_layer"][index] = 0
        cloud_props["cloud_base"][index] = np.nan
        cloud_props["cloud_top"][index]  = np.nan
        cloud_props["cloud_thickness"][index] = np.nan
        # cloud_type[index] = "ML"

    mask_clear_sky = cloud_occurrence.to_dataframe().sum(axis=1) == 0
    mask_clear_sky = mask_clear_sky.astype(float)
    cloud_occurrence["clear_sky"] = xr.DataArray(data=mask_clear_sky.values, coords={'time': data.time}, dims='time')
    cloud_occurrence["noise"]     = xr.DataArray(data=time_idx_noise, coords={'time': data.time}, dims='time')  # not mutually exclusive with clear sky

    new_height = np.arange(0, 14000+100, 100)
    # Identify single layer and non-noise clouds in time
    single_layer_mask = (cloud_occurrence["single_layer"] == 1) & (cloud_occurrence["noise"] == 0)
    # new_der = new_der.where(single_layer_mask, np.nan)['der'].groupby_bins('height', new_height, labels=new_height[1:]).mean()
    # cloud_occurrence.coords['cloud_type'] = ('time', cloud_type)
    # cloud_props.coords['cloud_type']      = ('time', cloud_type)
    
    date_str = cloud_occurrence.time[0].dt.strftime('%Y%m%d').values.item()
    # Save cloud occurrence as netCDF
    if save_files and process_all:
        cloud_occurrence.to_netcdf(path_save + f"/cloud_occurence/{date_str}_cloud_occurrence.nc")
        cloud_props.to_netcdf(path_save + f"/cloud_properties/{date_str}_cloud_props.nc")
        cloud_type.to_netcdf(path_save + f"/cloud_type/{date_str}_cloud_type.nc")
    # new_der.to_netcdf(f"/media/matheustolen/Seagate Basic/cloudnet/der/{date_str}_new_der.nc")
    
    # Add lwp in categorize in cloud_props(base, top, width):
    cloud_props["lwp"] = categorize["lwp"]
    cloud_props["rainfall_rate"] = categorize["rainfall_rate"]
    # ---------------------------------------------------------------------------------------------
    # Data sliced for certain time interval 
    # ---------------------------------------------------------------------------------------------
    start_time = pd.to_datetime(cloud_props.time.values[0]).strftime('%Y-%m-%d')
    ini= "00:00"
    end= "23:59"
    scloud_props = cloud_props.sel(time=slice(pd.to_datetime(f"{start_time} {ini}"), pd.to_datetime(f"{start_time} {end}")))
    scategorize  = categorize.sel(time=slice(pd.to_datetime(f"{start_time} {ini}"), pd.to_datetime(f"{start_time} {end}")))
    smask        = (cloud_occurrence["single_layer"] == 1).sel(time=slice(pd.to_datetime(f"{start_time} {ini}"), pd.to_datetime(f"{start_time} {end}")))
    # ---------------------------------------------------------------------------------------------
    # get data in cloud_props and categorize at each 5 minutes (moving mean of 5 min)
    # and take the correlation between cloud thickness and lwp:
    # https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.corr.html
    # ---------------------------------------------------------------------------------------------
    # Add cloud thickness and lwp in a pandas dataframe:
    df = scloud_props.cloud_thickness.to_dataframe()
    df['lwp'] = scloud_props.lwp.to_dataframe()
    #df.dropna(inplace=True)
    # Calculate the correlation between cloud thickness and lwp at each 5 minutes:
    time_roll = "30T"
    corr_pd = df['cloud_thickness'].rolling(window=time_roll).corr(df['lwp'])
    # corr_pd.plot()
    # transform the correlation to xarray dataset:
    scloud_props['corr'] = xr.DataArray(corr_pd.values, coords={'time': scloud_props.time}, dims='time')
    scloud_props['corr'].attrs['long_name'] = f'Correlation at each {time_roll} min'
    scloud_props['corr'].attrs['description'] = 'Correlation between cloud thickness and lwp at each 5 minutes'
    # scloud_props['corr'].attrs['units'] = '1'
    # ---------------------------------------------------------------------------------------------
    # Coarsen the data
    # ---------------------------------------------------------------------------------------------
    time_coarsen = "10min"
    coarsen_props      = scloud_props.copy()
    coarsen_categorize = scategorize.copy()

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, .02, .2, .6], wspace=0.1, hspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    scategorize['Z'].T.plot(ax=ax1, cmap='viridis', vmin=-40, vmax=20, add_colorbar=False)
    ax1.fill_between(scategorize.time.values, 0, 12000, where=smask, color='blue', alpha=0.2)
    # put NAN in the correalation where the cloud is not single layer
    # scloud_props['corr'] = scloud_props['corr'].where(smask)
    scloud_props.cloud_base.plot(ax=ax1, color='r', linestyle='-')
    scloud_props.cloud_top.plot(ax=ax1, color='k', linestyle='-')

    # plot a blue area for intervals with single layer clouds
    cbar = fig.add_subplot(gs[0, 1])
    cbar = plt.colorbar(ax1.collections[0], cax=cbar, label=categorize['Z'].units)
    
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    coarsen_props.cloud_thickness.plot(ax=ax2, color='b', label=time_coarsen)
    ax2.legend()

    ax3 = ax2.twinx()
    coarsen_props.lwp.plot(ax=ax3, color='r')
    ax3.spines['right'].set_color('red')

    ax4 = fig.add_subplot(gs[0, 3])
    scloud_props.plot.scatter(x='lwp', y='cloud_thickness', ax=ax4)

    ax5 = ax2.twinx()
    scloud_props['corr'] = scloud_props['corr'].where(scloud_props['lwp'] > 0.4)
    scloud_props.corr.plot(ax=ax5, color='g')
    ax5.spines['right'].set_position(('outward', 60))  # Move the y-axis outward
    ax5.spines['right'].set_color('green')
    # plot horizontal line at zero correlation
    ax5.axhline(y=0, color='g', linestyle='--')

    plt.show()
    # ---------------------------------------------------------------------------------------------

    array_cloud = np.zeros((data.time.shape[0], data.height.shape[0]))
    cloud_int= {
        "No Cloud": 0,
        "Liquid": 1,
        "Liquid-Precipitable": 2,
        "Ice": 3,
        "Ice-Precipitable": 4,
        "Mixed-Phase": 5,
        "Mixed-Phase-Precipitable": 6,
        "Noise": 7,
    }
    for cloud_number, indx in cloud_indx.items():
        cloud_name = ds_classification[cloud_number]
        integer    = cloud_int[cloud_name]
        if cloud_number in valid_clouds:
            array_cloud[indx[:, 0], indx[:, 1]] = integer
        else:
            array_cloud[indx[:, 0], indx[:, 1]] = cloud_int["Noise"]

    ds_cloud = xr.Dataset(data_vars={'cloud_classification': (['time', 'height'], array_cloud)},
                          coords={'time': data.time, 'height': data.height})
    ds_cloud.cloud_classification.attrs['long_name'] = 'Cloud classification'
    ds_cloud.cloud_classification.attrs['description'] = \
    'Cloud classification based on cloudnet target classification.\nHydrometeor cluster classification algorithm. \n0: No Cloud, \n1: Liquid, \n2: Liquid-Precipitable, \n3: Ice, \n4: Ice-Precipitable, \n5: Mixed-Phase, \n6: Mixed-Phase-Precipitable, \n7: Noise'
    
    letters = iter('abcdefghijklmnopqrstuvwxyz')
    show_category = False
    if not process_all or make_plot:
        # # -----------------------------------------------------------------------------
        # # Plot reflectivity time serie
        # # -----------------------------------------------------------------------------
        # # Do the same as above but plot categorize["Z"] insteado of target_classification
        # fig = plt.figure(figsize=(12, 5))
        # gs  = fig.add_gridspec(1, 2, width_ratios=[1, .02], wspace=0.05, hspace=0.08)
        # ax2 = fig.add_subplot(gs[0, 0])
        # pc = ax2.pcolormesh(categorize['time'], categorize['height']/1e3, categorize['Z'].T, cmap='viridis', vmin=-40, vmax=20) 
        # ax2.set_ylabel('Altitude (km) a.s.l')
        # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # ax2.set_ylim([0, 12])
        # # ax2.set_title(f"Reflectivity {date_str} - {site}")
        # ax2.grid()

        # cax = fig.add_subplot(gs[0, 1])
        # cbar = plt.colorbar(pc, cax=cax, orientation='vertical', label='Reflectivity (dBZ)')

        # # fig.savefig(PATH_FIG + f"{date_str}_reflectivity.png", dpi=700, bbox_inches='tight')
        # plt.show()
        # # -----------------------------------------------------------------------------

        # # -----------------------------------------------------------------------------
        # # Plot Target Classification Product (T.C.P) time serie
        # # -----------------------------------------------------------------------------
        # fig = plt.figure(figsize=(12, 5))
        # gs  = fig.add_gridspec(1, 2, width_ratios=[1, .02], wspace=0.05, hspace=0.08)
        # ax2 = fig.add_subplot(gs[0, 0])
        # pc = ax2.pcolormesh(cloud_classification['time'], cloud_classification['height']/1e3, cloud_classification.T, cmap=manual_cmap, vmin=0, vmax=ncolors)
        # # countour = ax2.contour(categorize['model_time'], categorize['model_height'][:]/1e3,
        # #                         categorize['temperature'][:].T - 273.15,
        # #                         levels=[-40, -25, -10, 0, 5], colors='black', linewidths=0.5)
        # # countour = ax1.contour(categorize['model_time'], categorize['model_height'][:]/1e3,
        # #                 categorize['temperature'][:].T - 273.15,
        # #                 levels=[-40, -25, -10, 0, 5], colors='black', linewidths=0.5)
        # # countour.clabel(inline=True, fmt='%2.1f'+r'$^{\circ}$C', fontsize=12)
        # ax2.set_ylabel('Height (km) a.s.l')
        # ax2.set_xlabel('Time (UTC)')
        # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # # ax2.set_ylim([categorize['height'][0]/1e3, categorize['height'][-1]/1e3])

        # # ax2.set_title(f"Cloud Classification {date_str} - {site}")
        # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # # ax2.set_xlim(data.time.values[0], data.time.values[-1])
        # ax2.grid()
        # cax = fig.add_subplot(gs[0, 1])
        # cbar = plt.colorbar(pc, cax=cax, ticks=[], orientation='vertical')

        # for idx, (color, name) in enumerate(zip(manual_cmap.colors, color_names)):
        #     rect = plt.Rectangle((0, idx), 1, 1, color=color)
        #     cbar.ax.add_patch(rect)
        #     cbar.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=14)
        # ax2.set_facecolor('white')
        # ax2.set_ylim([0, ds_cloud['height'][-1]/1e3])
        # fig.savefig(PATH_FIG + f"{date_str}_cloud_classification.png", dpi=700, bbox_inches='tight')
        # plt.show()
        # # -----------------------------------------------------------------------------

        # # -----------------------------------------------------------------------------
        # # Plot Cloud Clusters (C.C) time serie
        # # -----------------------------------------------------------------------------
        # fig, ax = plt.subplots(figsize=(12, 5))
        # for cloud_number, indx in cloud_indx.items():
        # # for cloud_number in cloud_composition.keys:
        # #     indx= cloud_indx[cloud_number]
        #     x = data.time.values[indx[:, 0]]
        #     y = data.height.values[indx[:, 1]] / 1e3
        #     sc = ax.scatter(x, y, color=np.random.rand(3,), s=1)
        # ax.set_ylabel('Altitude (km) a.s.l')
        # ax.set_xlabel('Time (UTC)')
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # # ax.set_title(f"Cloud Classification {date_str} - {site}")
        # ax.set_ylim([0, 12])
        # # show untill the 1:00 UTC of the next day
        # ax.set_xlim(data.time.values[0], data.time.values[-1])
        # ax.grid()
        # fig.savefig(PATH_FIG + f"{date_str}_cloud_identification.png", dpi=700, bbox_inches='tight')
        # plt.show()
        # # -----------------------------------------------------------------------------
        
        # # -----------------------------------------------------------------------------
        # # Plot Cloud Classification Product (C.C.P) time serie
        # # -----------------------------------------------------------------------------
        # fig = plt.figure(figsize=(12, 5))
        # gs = fig.add_gridspec(1, 2, width_ratios=[1, .02], wspace=0.05)
        # ax = fig.add_subplot(gs[0, 0])
        # pc0 = ax.pcolormesh(ds_cloud['time'], ds_cloud['height']/1e3, ds_cloud['cloud_classification'].T, cmap=cloud_cmap, vmin=0, vmax=len(cloud_category))

        # # sc1 = ax.scatter(cloud_props['time'], cloud_props['cloud_base']/1e3, s=1, color='r', label='Cloud base')
        # # sc2 = ax.scatter(cloud_props['time'], cloud_props['cloud_top']/1e3, s=1, color='k', label='Cloud top')
        # ax.set_ylabel('Height (km) a.s.l')
        # ax.set_xlabel('Time (UTC)')
        # ax.grid()
        # # set x limitis from 17 to 18 UTC
        # # ax.set_xlim(datetime.datetime.strptime(date_str, '%Y%m%d') + datetime.timedelta(hours=1), datetime.datetime.strptime(date_str, '%Y%m%d') + datetime.timedelta(hours=15))
        # ax.set_ylim([0, ds_cloud['height'][-1]/1e3])

        # ax.set_xlim(data.time.values[0], data.time.values[-1])
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # ax.set_facecolor('white')

        # cax_scat = fig.add_subplot(gs[0, 1])
        # cbar_scat = plt.colorbar(pc0, cax=cax_scat, ticks=[], orientation='vertical')

        # for idx, (color, name) in enumerate(zip(cloud_cmap.colors, cloud_category)):
        #     rect = plt.Rectangle((0, idx), 1, 1, color=color)
        #     cbar_scat.ax.add_patch(rect)
        #     cbar_scat.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=14)
        # fig.savefig(PATH_FIG + f"{date_str}_cloud_classification_cluster.png", dpi=1000, bbox_inches='tight')
        # plt.show()
        # # -----------------------------------------------------------------------------

        # -----------------------------------------------------------------------------
        # Plot C.T.P, C.C.P and LWP
        # -----------------------------------------------------------------------------
        if show_category:
            fig = plt.figure(figsize=(14, 11))
            gs = fig.add_gridspec(3, 2, width_ratios=[1, .02], height_ratios=[1, 1, .3], wspace=0.05, hspace=0.08)
        else:
            fig = plt.figure(figsize=(13, 9))
            gs = fig.add_gridspec(3, 2, width_ratios=[1, .02], height_ratios=[1, 1, .5], wspace=0.05, hspace=0.05)

        ax1 = fig.add_subplot(gs[1, 0])
        pc0 = ax1.pcolormesh(ds_cloud['time'], ds_cloud['height']/1e3, ds_cloud['cloud_classification'].T, cmap=cloud_cmap, vmin=0, vmax=len(cloud_category))

        sc1 = ax1.scatter(cloud_props['time'], cloud_props['cloud_base']/1e3, s=1, color='r', label='Cloud base')
        sc2 = ax1.scatter(cloud_props['time'], cloud_props['cloud_top']/1e3, s=1, color='k', label='Cloud top')
        ax1.xaxis.set_tick_params(labelbottom=False)
        ax1.set_ylabel('Height (km) a.s.l')
        ax1.grid()
        # set x limitis from 17 to 18 UTC
        # ax1.set_xlim(datetime.datetime.strptime(date_str, '%Y%m%d') + datetime.timedelta(hours=1), datetime.datetime.strptime(date_str, '%Y%m%d') + datetime.timedelta(hours=15))
        # ax1.set_ylim([0, 4])

        ax1.set_xlim(data.time.values[0], data.time.values[-1])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax2 = fig.add_subplot(gs[0, 0], sharex=ax1, sharey=ax1)
        pc = ax2.pcolormesh(cloud_classification['time'], cloud_classification['height']/1e3, cloud_classification.T, cmap=manual_cmap, vmin=0, vmax=ncolors)
        countour = ax2.contour(categorize['model_time'], categorize['model_height'][:]/1e3,
                                categorize['temperature'][:].T - 273.15,
                                levels=[-40, -25, -10, 0, 5], colors='black', linewidths=0.5)
        countour = ax1.contour(categorize['model_time'], categorize['model_height'][:]/1e3,
                        categorize['temperature'][:].T - 273.15,
                        levels=[-40, -25, -10, 0, 5], colors='black', linewidths=0.5)
        countour.clabel(inline=True, fmt='%2.1f'+r'$^{\circ}$C', fontsize=12)
        ax2.set_ylabel('Height (km) a.s.l')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # ax2.set_ylim([categorize['height'][0]/1e3, categorize['height'][-1]/1e3])
        ax2.set_ylim([0, cloud_classification['height'][-1]/1e3])
        ax2.xaxis.set_tick_params(labelbottom=False)
        ax2.set_title(f"Cloud Classification {date_str} - {site}")
        ax2.grid()

        cax = fig.add_subplot(gs[0, 1])
        cbar = plt.colorbar(pc, cax=cax, ticks=[], orientation='vertical')

        for idx, (color, name) in enumerate(zip(manual_cmap.colors, color_names)):
            rect = plt.Rectangle((0, idx), 1, 1, color=color)
            cbar.ax.add_patch(rect)
            cbar.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=14)

        cax_scat = fig.add_subplot(gs[1, 1])
        cbar_scat = plt.colorbar(pc0, cax=cax_scat, ticks=[], orientation='vertical')

        for idx, (color, name) in enumerate(zip(cloud_cmap.colors, cloud_category)):
            rect = plt.Rectangle((0, idx), 1, 1, color=color)
            cbar_scat.ax.add_patch(rect)
            cbar_scat.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=14)

        if show_category:
            ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
            ax3.plot(data.time, cloud_occurrence['single_layer'], label='Single Layer')
            ax3.plot(data.time, cloud_occurrence['multi_layer'], label='Multi Layer')
            ax3.plot(data.time, cloud_occurrence['clear_sky'], label='Clear Sky')
            ax3.plot(data.time, cloud_occurrence['noise'], label='Noise')
            ax3.set_xlabel('Time')
            ax3.legend()
        else:
            ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
            ax3.plot(categorize['time'], categorize['lwp']*1000, 'm')
            ax3.set_ylabel('LWP (g/m$^2$)')
            ax3.set_xlabel('Time (UTC) HH:MM')
            ax3.grid()

        # fig.savefig(PATH_FIG_TEST + f"{date_str}_cloud_classification_zoom.png", dpi=300, bbox_inches='tight')
        case_to_only_save = False
        plt.show()
        # -----------------------------------------------------------------------------

        # # -----------------------------------------------------------------------------
        # # Plot C.T.P, C.C.P, ceilo att. backscatter and radar reflectivity (Z) 
        # # -----------------------------------------------------------------------------
        # letters = iter('abcdefghijklmnopqrstuvwxyz')
        # fig = plt.figure(figsize=(17, 7))
        # gs = fig.add_gridspec(2, 5, width_ratios=[1, .02, .5, 1, 0.02], hspace=0.08, wspace=0.05)
         
        # ax = fig.add_subplot(gs[0, 0])
        # pc0 = ax.pcolormesh(cloud_classification['time'], cloud_classification['height']/1e3, cloud_classification.T, cmap=manual_cmap, vmin=0, vmax=ncolors)
        # countour = ax.contour(categorize['model_time'], categorize['model_height'][:]/1e3,
        #                         categorize['temperature'][:].T - 273.15,
        #                         levels=[-40, -25, -10, 0, 5], colors='black', linewidths=0.5)
        # countour.clabel(inline=True, fmt='%2.1f'+r'$^{\circ}$C', fontsize=12)
        # ax.set_ylabel('Height (km) a.s.l')
        # ax.grid()
        # ax.xaxis.set_tick_params(labelbottom=False)
        # ax.set_ylim([0, 12])
        # ax.text(0.02, 0.95, next(letters)+")", transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

        # cax = fig.add_subplot(gs[0, 1])
        # cbar = plt.colorbar(pc, cax=cax, ticks=[], orientation='vertical')

        # for idx, (color, name) in enumerate(zip(manual_cmap.colors, color_names)):
        #     rect = plt.Rectangle((0, idx), 1, 1, color=color)
        #     cbar.ax.add_patch(rect)
        #     cbar.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=14)

        # ax1 = fig.add_subplot(gs[1, 0], sharex=ax)
        # pc1 = ax1.pcolormesh(ds_cloud['time'], ds_cloud['height']/1e3, ds_cloud['cloud_classification'].T, cmap=cloud_cmap, vmin=0, vmax=len(cloud_category))

        # sc1 = ax1.scatter(cloud_props['time'], cloud_props['cloud_base']/1e3, s=10, color='r', label='Cloud base', marker='x')
        # sc2 = ax1.scatter(cloud_props['time'], cloud_props['cloud_top']/1e3, s=10, color='k', label='Cloud top', marker='x')

        # ax1.set_ylabel('Height (km) a.s.l')
        # ax1.set_xlabel('Time (UTC)')
        # ax1.grid()
        # # ax1.legend()
        # ax1.text(0.02, 0.95, next(letters)+")", transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')

        # ax1.set_xlim(data.time.values[0], data.time.values[-1])
        # ax1.set_ylim([0, data.height[-1]/1e3])
        # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # cax_scat = fig.add_subplot(gs[1, 1])
        # cbar_scat = plt.colorbar(pc1, cax=cax_scat, ticks=[], orientation='vertical')

        # for idx, (color, name) in enumerate(zip(cloud_cmap.colors, cloud_category)):
        #     rect = plt.Rectangle((0, idx), 1, 1, color=color)
        #     cbar_scat.ax.add_patch(rect)
        #     cbar_scat.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=14)
        
        # ax2 = fig.add_subplot(gs[0, 3], sharex=ax, sharey=ax)  
        # pc2 = ax2.pcolormesh(categorize['time'], categorize['height']/1e3, categorize['Z'].T, cmap='viridis', vmin=-40, vmax=15)
        # ax2.yaxis.set_tick_params(labelleft=False)
        # ax2.xaxis.set_tick_params(labelbottom=False)
        # ax2.grid()
        # ax2.text(0.02, 0.95, next(letters)+")", transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')


        # cax2 = fig.add_subplot(gs[0, 4])
        # cbar2 = plt.colorbar(pc2, cax=cax2, orientation='vertical', label='Reflectivity (dBZ)')

        # ax3 = fig.add_subplot(gs[1, 3], sharex=ax, sharey=ax)
        # pc3 = ax3.pcolormesh(categorize['time'], categorize['height']/1e3, categorize['beta'].T, cmap='jet',
        #                      norm=colors.LogNorm(vmin=1e-7, vmax=1e-4))
        
        # ax3.yaxis.set_tick_params(labelleft=False)
        # ax3.set_xlabel('Time (UTC)')
        # ax3.grid()
        # ax3.text(0.02, 0.95, next(letters)+") ", transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top')

        # cax3 = fig.add_subplot(gs[1, 4])
        # cbar3 = plt.colorbar(pc3, cax=cax3, orientation='vertical',
        #                       label = r"$\beta$ (m$^{-1}$ sr$^{-1}$)")
        
        # # fig.savefig(PATH_FIG + f"{date_str}_cloudnet_misclassification.png", dpi=300, bbox_inches='tight')
        # plt.show()
        # -----------------------------------------------------------------------------

        # -----------------------------------------------------------------------------
        # Plot C.T.P and ceilo att. backscatter
        # -----------------------------------------------------------------------------
        # letters = iter('abcdefghijklmnopqrstuvwxyz')
        # fig = plt.figure(figsize=(20, 5))
        # gs = fig.add_gridspec(1, 5, width_ratios=[1, .02, .5, 1, 0.02], wspace=0.03)
         
        # ax = fig.add_subplot(gs[0, 0])
        # pc0 = ax.pcolormesh(cloud_classification['time'], cloud_classification['height']/1e3, cloud_classification.T, cmap=manual_cmap, vmin=0, vmax=ncolors)
        # countour = ax.contour(categorize['model_time'], categorize['model_height'][:]/1e3,
        #                         categorize['temperature'][:].T - 273.15,
        #                         levels=[-40, -25, -10, 0, 5], colors='black', linewidths=0.5)
        # countour.clabel(inline=True, fmt='%2.1f'+r'$^{\circ}$C', fontsize=12)
        # ax.set_ylabel('Height (km) a.s.l')
        # ax.set_xlabel('Time (UTC)')
        # ax.grid()
        # ax.set_ylim([0, 12])
        # ax.set_xlim(cloud_classification.time.values[0], cloud_classification.time.values[-1])
        # ax.set_ylim([0, cloud_classification.height[-1]/1e3])
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # ax.text(0.02, 0.95, next(letters)+")", transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

        # cax = fig.add_subplot(gs[0, 1])
        # cbar = plt.colorbar(pc, cax=cax, ticks=[], orientation='vertical')

        # for idx, (color, name) in enumerate(zip(manual_cmap.colors, color_names)):
        #     rect = plt.Rectangle((0, idx), 1, 1, color=color)
        #     cbar.ax.add_patch(rect)
        #     cbar.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=14)


        # ax3 = fig.add_subplot(gs[0, 3], sharex=ax, sharey=ax)
        # pc3 = ax3.pcolormesh(categorize['time'], categorize['height']/1e3, categorize['beta'].T, cmap='jet',
        #                      norm=colors.LogNorm(vmin=1e-7, vmax=1e-4))
        
        # ax3.yaxis.set_tick_params(labelleft=False)
        # ax3.set_xlabel('Time (UTC)')
        # ax3.grid()
        # ax3.text(0.02, 0.95, next(letters)+") ", transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top')

        # cax3 = fig.add_subplot(gs[0, 4])
        # cbar3 = plt.colorbar(pc3, cax=cax3, orientation='vertical',
        #                       label = r"$\beta$ (m$^{-1}$ sr$^{-1}$)")
        
        # fig.savefig(PATH_FIG + f"{date_str}_cloudnet_misclassification.png", dpi=300, bbox_inches='tight')
        # plt.show()
        # -----------------------------------------------------------------------------
# set_trace()
    # # Coarsen der and ier
    # target_height = np.arange(0, 14000+100, 100)
    # target_time   = data.time.values[::2]
    # coarsened_array = np.full((target_time.shape[0], target_height.shape[0]), np.nan)
    # test_data = new_der['der'].values
    # # Loop over each cell in the target grid
    # for i in range(target_time.shape[0]-1):
    #     for j in range(target_height.shape[0]-1):
    #         t_min = target_time[i]
    #         t_max = target_time[i+1]
    #         h_min = target_height[j]
    #         h_max = target_height[j+1]

    #         # Find the data points that fall within the current cell
    #         cell_data = test_data.sel(time=slice(t_min, t_max-1),
    #                                 height=slice(h_min, h_max-1)
    #                                 )
    #         # Calculate the mean of the data points within the current cell
    #         coarsened_array[i, j] = cell_data.mean()

    # test_data.groupby_bins('height', target_height, labels=target_height[1:]).mean()
   # Create a new DataArray with the manually coarsened data
    
    # coarsened_der = xr.DataArray(
    # data=coarsened_array,
    # dims=['time', 'height'],
    # coords={'time': target_time,
    #         'height': (target_height[0:] + target_height[:-1]) / 2},
    # name='coarsened_der'
    # )

    # number_of_layers = pd.DataFrame(0, index=cloud_classification['time'],
    #                                 columns=["Liquid", "Ice", "Mixed_phase", "Pre_liquid", "Pre_mixed_phase"])

    # height_cloud_base         = pd.DataFrame(index=cloud_classification['time'],
    #                                     columns=["Liquid", "Ice", "Mixed_phase", "Pre_liquid", "Pre_mixed_phase"])
    # height_cloud_top          = pd.DataFrame(index=cloud_classification['time'],
    #                                     columns=["Liquid", "Ice", "Mixed_phase", "Pre_liquid", "Pre_mixed_phase"])


    # groups_height_idx = groupSequence([indx[:,1] for indx in cloud_indx.values()])

# Microphysical properties inside specific clouds
# for single layer mixed phase clouds


# --------------------------------------------------------------------------------
# ----------------------------- Doppler spectra analysis -------------------------
# --------------------------------------------------------------------------------
# set_trace()
# cloud_identifier = 5
# mask_cloud_type = ds_cloud['cloud_classification'] == cloud_identifier
# # mask for single layer without noie
# mask_time_single_layer = (cloud_occurrence['single_layer'] == 1) & (cloud_occurrence['noise'] == 0)

# # for mixed phase clouds
# filtered_cloud = ds_cloud.where(mask_cloud_type & mask_time_single_layer)
# filtered_targ_clas = cloud_classification.where(mask_cloud_type & mask_time_single_layer)

# # initialize an xarray dataset to store the microphysical properties
# cloud_properties = xr.Dataset(coords={'time': ds_cloud.time, 'height': ds_cloud.height})
# cloud_properties["lwp"] = categorize["lwp"]*1e3
# cloud_properties["lwp"].attrs['units'] = r"$g/m^2$"
# cloud_properties["lwp"].attrs['long_name'] = "Liquid Water Path"
# cloud_properties["width"] = categorize["width"]
# cloud_properties["Ze"] = categorize["Z"]
# # cloud_properties["sldr"] = categorize["sldr"]

# # Path to microphysics retrieval from cloudnet
# path_cloudnet_retrieval = "../../tests/data/microphysics/"
# download_cloudnet_products(date_ini, date_end, path_cloudnet_retrieval, product="der", site=site)
# download_cloudnet_products(date_ini, date_end, path_cloudnet_retrieval, product="categorize", site=site)
# # Read data with xarray
# cloudnet_data = xr.open_dataset(path_cloudnet_retrieval + f"{date_str}_{site}_der.nc")
# cloudnet_data = cloudnet_data.where(mask_cloud_type & mask_time_single_layer)

# cloud_properties["der_cloudnet"] = cloudnet_data['der']*1e6
# cloud_properties["der_cloudnet"].attrs['units'] = r"$\mu m$"
# cloud_properties["der_cloudnet"].attrs['long_name'] = "Effective radius by Cloudnet"
# cloud_properties["der_cloudnet_scaled"] = cloudnet_data['der_scaled']*1e6
# cloud_properties["der_cloudnet_scaled"].attrs['units'] = r"$\mu m$"
# cloud_properties["der_cloudnet_scaled"].attrs['long_name'] = "Effective radius by Cloudnet scaled"

# mask = cloud_properties["der_cloudnet"].isnull().to_numpy()
# filtered_z = categorize['Z'].where(~mask)
# filtered_lwp = categorize['lwp'].where(~np.sum(mask, axis=1))

# ini = '2024-04-02T00:30:00'
# end = '2024-04-02T23:30:00'
# ds_cloud.cloud_classification.T.plot(cmap=cloud_cmap, figsize=(12, 5), vmin=0, vmax=len(cloud_category))
# # put colornames in the colorbar
# filtered_cloud.cloud_classification.T.plot(cmap=cloud_cmap, figsize=(12, 5), vmin=0, vmax=len(cloud_category))

# cloud_properties["der_knist"] = HMmodel(filtered_lwp*1e3,
#                                         filtered_z,
#                                         v=13.51).get_re()
# cloud_properties["der_knist"].attrs['units'] = r"$\mu m$"
# cloud_properties["der_knist"].attrs['long_name'] = "Effective radius by Knist"
# cloud_properties["der_knist"].attrs['description'] = "Effective radius calculated by Knist model"
# # Calculate the Droplets effective radius

# der_output_path = os.path.join(path_cloudnet_retrieval, date_ini.replace("-","") + "_granada_" + 'der_corrected.nc')
# path_categorize = os.path.join(path_cloudnet_retrieval, date_ini.replace("-","") + "_granada_" + 'categorize.nc')
# # params = Parameters(2.0, 200.0e6, 200.0e6, 0.35, 0.1, 5.0e-3)
# params_der = Parameters(2.0, 150.0e6, 150.0e6, 0.28, 0.1, 5.0e-3)
# generate_der(path_categorize, der_output_path, parameters=params_der)
# cloudnet_der_corrected = xr.open_dataset(der_output_path)
# cloudnet_der_corrected = cloudnet_der_corrected.where(mask_cloud_type & mask_time_single_layer)
# cloud_properties["der_cloudnet_corrected"] = cloudnet_der_corrected['der']*1e6
# cloud_properties["der_cloudnet_corrected"].attrs['units'] = r"$\mu m$"
# cloud_properties["der_cloudnet_corrected"].attrs['long_name'] = "Effective radius by Cloudnet - corrected"

# cloud_properties["der_cloudnet_scaled_corrected"] = cloudnet_der_corrected['der_scaled']*1e6
# cloud_properties["der_cloudnet_scaled_corrected"].attrs['units'] = r"$\mu m$"
# cloud_properties["der_cloudnet_scaled_corrected"].attrs['long_name'] = "Effective radius by Cloudnet scaled - corrected"

# # Plot the effective radius
# cloud_properties.der_knist.T.plot(cmap='jet', figsize=(12, 5))
# cloud_properties.der_cloudnet.T.plot(cmap='jet', figsize=(12, 5))
# cloud_properties.der_cloudnet_scaled.T.plot(cmap='jet', figsize=(12, 5), vmin=0, vmax=50)

# filtered_targ_clas["height"] = data.height - GRANADA_ALTITUDE
# cloud_properties["height"] = data.height - GRANADA_ALTITUDE

# filtered_targ_clas.T.plot(cmap=manual_cmap, figsize=(12, 5), vmin=0, vmax=ncolors)
# plt.xlim(datetime.datetime.strptime(date_str, '%Y%m%d') + datetime.timedelta(hours=17), datetime.datetime.strptime(date_str, '%Y%m%d') + datetime.timedelta(hours=18))

# # Plot the histograms of the effective radius
# binwidth = .5
# rmax = 200
# bins = np.arange(0, rmax + binwidth, binwidth)
# fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# cloud_properties.der_cloudnet_scaled.plot.hist(ax=ax, bins=bins, alpha=0.5, label='Cloudnet scaled', color="b")
# cloud_properties.der_cloudnet_scaled_corrected.plot.hist(ax=ax, bins=bins, alpha=0.5, label='Cloudnet scaled corrected', color="y")
# # change from bin to delta reff
# ax.set_xlabel(r"R$_{eff}$, $\mu m$")
# ax.set_ylabel("Counts (#)")
# ax.legend()
# ax.set_xlim(0, rmax)
# # fig.savefig(PATH_FIG + f"{date_str}_der_scaled_cloudnet_comparison_histogram_{cloud_identifier}.png", dpi=300, bbox_inches='tight')
# plt.show()

# hydrometeor_type = ICE_WITH_SUP_WATER

# # Plot the histograms of the effective radius
# bins = np.arange(0, rmax+ binwidth, binwidth)
# fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# cloud_properties.where(filtered_targ_clas == hydrometeor_type).der_cloudnet.plot.hist(ax=ax, bins=bins, alpha=0.5, label='Cloudnet', color="m")
# cloud_properties.where(filtered_targ_clas == hydrometeor_type).der_cloudnet_corrected.plot.hist(ax=ax, bins=bins, alpha=0.5, label='Cloudnet corrected', color="g")
# # change from bin to delta reff
# ax.set_xlabel(r"R$_{eff}$, $\mu m$")
# ax.set_ylabel("Counts (#)")
# ax.legend()
# ax.set_xlim(0, rmax)
# # fig.savefig(PATH_FIG + f"{date_str}_der_cloudnet_comparison_histogram_{cloud_identifier}.png", dpi=300, bbox_inches='tight')
# plt.show()

# # Plot the histograms of the effective radius
# bins = np.arange(0, rmax + binwidth, binwidth)
# fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# cloud_properties.der_knist.plot.hist(ax=ax, bins=bins, alpha=0.5, label='Knist', color="r")
# cloud_properties.der_cloudnet_scaled_corrected.plot.hist(ax=ax, bins=bins, alpha=0.5, label='Cloudnet scaled corrected', color="y")
# cloud_properties.der_cloudnet_corrected.plot.hist(ax=ax, bins=bins, alpha=0.5, label='Cloudnet corrected', color="g")
# # change from bin to delta reff
# ax.set_xlabel(r"R$_{eff}$, $\mu m$")
# ax.set_ylabel("Counts (#)")
# ax.legend()
# ax.set_xlim(0, rmax)
# # fig.savefig(PATH_FIG + f"{date_str}_der_comparison_histogram_{cloud_identifier}.png", dpi=300, bbox_inches='tight')
# plt.show()

# print(f" Median: {cloud_properties.der_knist.mean().values}, {cloud_properties.der_knist.median().values}, {cloud_properties.der_knist.std().values}")
# print(f" Median: {cloud_properties.der_cloudnet_scaled.mean().values}, {cloud_properties.der_cloudnet_scaled.median().values}, {cloud_properties.der_cloudnet_scaled.std().values}")
# print(f" Median: {cloud_properties.der_cloudnet_scaled_corrected.mean().values}, {cloud_properties.der_cloudnet_scaled_corrected.median().values}, {cloud_properties.der_cloudnet_scaled_corrected.std().values}")
# print(f" Median: {cloud_properties.der_cloudnet.mean().values}, {cloud_properties.der_cloudnet.median().values}, {cloud_properties.der_cloudnet.std().values}")
# print(f" Median: {cloud_properties.der_cloudnet_corrected.mean().values}, {cloud_properties.der_cloudnet_corrected.median().values}, {cloud_properties.der_cloudnet_corrected.std().values}")
# # cloud_properties.der_knist.T.sel(time=slice(ini, end)).plot(cmap='jet', figsize=(12, 5))

# cloud_properties["mean_der_cloudnet_scaled"] = cloud_properties.der_cloudnet_scaled.mean(dim='height')
# fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# cloud_properties.where(filtered_targ_clas == hydrometeor_type).plot.scatter(x="lwp", y="mean_der_cloudnet_scaled")
# # ax.set_ylim(0, 100)

# fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# cloud_properties.where(filtered_targ_clas == hydrometeor_type).plot.scatter(x="Ze", y="der_cloudnet_scaled")
# # ax.set_ylim(0, 100)

# # colorplot of the effective radius from cloudnet scaled in function of Ze and width variables
# fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# sc = ax.scatter(cloud_properties.where(filtered_targ_clas == hydrometeor_type).Ze,
#                 cloud_properties.where(filtered_targ_clas == hydrometeor_type).width,
#                 c=cloud_properties.where(filtered_targ_clas == hydrometeor_type).der_cloudnet_scaled_corrected,
#                 cmap='gist_ncar',
#                 vmin=0,
#                 vmax=20)
# cbar = plt.colorbar(sc, ax=ax, extend='both')
# cbar.set_label(r"$\mu m$")
# ax.set_xlabel("Ze (dBZ)")
# ax.set_ylabel("Width (m/s)")
# # fig.savefig(PATH_FIG + f"{date_str}_der_cloudnet_scaled_ice_with_sup_water.png", dpi=300, bbox_inches='tight')
# plt.show()

# #--------------------------------------------------------------------------------
# # Start RAW data analysis
# #--------------------------------------------------------------------------------

# start_time = pd.DatetimeIndex(cloud_properties.time.values)[0]
# instrument_name   = 'nebula_ka'
# pattern           = f"{start_time.strftime('%y%m%d')}_18*ZEN.*"
# filepath_raw      = f"/home/matheustolen/shared/RAW/UGR/{instrument_name}/{start_time.strftime('%Y/%m/%d')}"

# # create the same folder structure in the product folder
# product_folder = f"../../../radar_data/PRODUCTS/{instrument_name}/"
# date_list      = filepath_raw.split("/")[-3:]
# filepath_nc    = os.path.join(product_folder, *date_list)
# os.makedirs(filepath_nc, exist_ok=True) # create the folder if it does not exist

# filepaths = glob.glob(os.path.join(filepath_raw, pattern))
# for file in filepaths:
#     if os.path.basename(file).endswith('.LV0'):
#         raw_filename  = os.path.basename(file).replace('LV0', 'LV0.nc')
#         filename_nc   = os.path.join(filepath_nc, os.path.basename(file).replace('LV0', 'LV0.nc'))
#         # check if raw netcdf file exists
#         if not os.path.exists(filename_nc):
#             rpg2nc(file, output_file=filename_nc)
#             # rpg2nc_rpgpy(filepath_rawdata[0]+"/*LV0", output_file=filepath_output+"/huge_day_file.nc") # If we want to concatenate one day file
#             print(f"File created at {filename_nc}")
#         else:
#             print(f"File {filename_nc} already exists")
        
#         # Read the raw data with gfatpy
#         radar_lv0 = rpg(filename_nc)
#         # radar_lv0._data = radar_lv0._data.interp(time=cloud_properties.time.values, method="nearest")
#         # do the interpolation to the cloud properties and resample for 30 seconds
        
#     else:
#         raw_filename = os.path.basename(file).replace('LV1', 'LV1.nc')
#         filename_nc  = os.path.join(filepath_nc, os.path.basename(file).replace('LV1', 'LV1.nc'))
#                 # check if raw netcdf file exists
#         if not os.path.exists(filename_nc):
#             rpg2nc(file, output_file=filename_nc)
#             # rpg2nc_rpgpy(filepath_rawdata[0]+"/*LV0", output_file=filepath_output+"/huge_day_file.nc") # If we want to concatenate one day file
#             print(f"File created at {filename_nc}")
#         else:
#             print(f"File {filename_nc} already exists")
#         radar_lv1 = rpg(filename_nc)
        
# radar_lv0._data = radar_lv0.data.resample(time="30S").mean()
# radar_lv1._data = radar_lv1.data.resample(time="30S").mean()

# scloud_properties = cloud_properties.sel(time=radar_lv0.data.time, method="nearest")
# scloud_prop  = cloud_properties.sel(time=radar_lv0.data.time, method="nearest")
# starg_clas   = filtered_targ_clas.sel(time=radar_lv0.data.time, method="nearest")
# scategorize  = categorize.sel(time=radar_lv0.data.time, method="nearest")

# starg_clas.T.plot(cmap=manual_cmap, figsize=(12, 5), vmin=0, vmax=ncolors)
# # # Use radar_lv0.data.time.values[0] and radar_lv0.data.time.values[-1] to select the time slice to get cloud properties
# # scloud_prop    = cloud_properties.sel(time=slice(radar_lv0.data.time.values[0], radar_lv0.data.time.values[-1]))
# # starg_clas     = filtered_targ_clas.sel(time=slice(radar_lv0.data.time.values[0], radar_lv0.data.time.values[-1]))
# # scategorize    = categorize.sel(time=slice(radar_lv0.data.time.values[0], radar_lv0.data.time.values[-1]))

# # # --------------------------------------------------------------------------------------------
# # mask pixels with ice with supercooled water
# mask_ice_sup_water = starg_clas == ICE_WITH_SUP_WATER
# mask_liquid        = starg_clas == CLOUD_LIQUID
# mask_ice          = starg_clas == ICE_PARTICLES

# starg_clas.T.plot(cmap=manual_cmap, figsize=(12, 5), vmin=0, vmax=ncolors)
# scloud_prop.der_cloudnet_scaled.T.plot(cmap='jet', figsize=(12, 5), vmin=0, vmax=100)
# scloud_prop.lwp.T.plot(figsize=(12, 5))
# # radar_lv1.data.dBZe.T.plot(cmap='jet', figsize=(12, 5))
# # radar_lv1.data.width.resample(time="30S").mean().T.plot(cmap='jet', 
# #                                                         figsize=(12, 5))
# # radar_lv1.data.skewness.resample(time="30S").mean().T.plot(cmap='jet', 
# #                                                         figsize=(12, 5),
# #                                                         vmin=-1,
# #                                                         vmax=1)
# # radar_lv1.data.kurtosis.resample(time="30S").mean().T.plot(cmap='jet', 
# #                                                         figsize=(12, 5), 
# #                                                         vmin=0,
# #                                                         vmax=4)
# # radar_lv1.data.differential_phase.resample(time="30S").mean().T.plot(cmap='jet', 
# #                                                                      figsize=(12, 5), 
# #                                                                      vmin=2, 
# #                                                                      vmax=-2.5)
# # # scloud_prop["differential_phase"] = radar_lv1.data.differential_phase.resample(time="30S").mean().interp(method="nearest", time=scloud_prop.time.values)
# # radar_lv1.data.ldr_slanted.resample(time="30S").mean().T.plot(cmap='jet',
# #                                                                      figsize=(12, 5), 
# #                                                                      vmin=-70, 
# #                                                                      vmax=-30)
# # radar_lv1.data.correlation_coefficient.resample(time="30S").mean().T.plot(cmap='jet',
# #                                                                      figsize=(12, 5), 
# #                                                                      vmin=0, 
# #                                                                      vmax=1)
# # radar_lv1.data.differential_phase_shift.resample(time="30S").mean().T.plot(cmap='jet',
# #                                                                      figsize=(12, 5),
# #                                                                         vmin=-.2,
# #                                                                         vmax=.2)
# scloud_prop.der_cloudnet_scaled.T.plot(cmap='jet', figsize=(12, 5), vmin=0, vmax=100)
# scloud_prop.der_cloudnet.T.plot(cmap='jet', figsize=(12, 5))

# # fig = plt.figure(figsize=(6, 5))
# # scloud_prop.where(mask_ice_sup_water).plot.scatter(x="width", y="der_cloudnet_scaled")
# # fig = plt.figure(figsize=(6, 5))
# # scloud_prop.where(mask_ice_sup_water).plot.scatter(x="Ze", y="der_cloudnet_scaled")
# # fig = plt.figure(figsize=(6, 5))
# # scloud_prop.where(mask_ice_sup_water).plot.scatter(x="sldr", y="der_cloudnet_scaled")
# # plt.xlim(-55, -15)
# # fig = plt.figure(figsize=(6, 5))
# # scloud_prop.where(mask_ice_sup_water).plot.scatter(x="differential_phase", y="der_cloudnet_scaled")

# # starg_clas.T.where(mask_ice).plot(cmap=manual_cmap, figsize=(12, 5), vmin=0, vmax=ncolors)
# # starg_clas.T.where(mask_ice_sup_water).plot(cmap=manual_cmap, figsize=(12, 5), vmin=0, vmax=ncolors)
# # starg_clas.T.where(mask_liquid).plot(cmap=manual_cmap, figsize=(12, 5), vmin=0, vmax=ncolors)
# # # --------------------------------------------------------------------------------------------
# mask_to_ice_droplets = True
# mask_to_only_ice     = False
# mask_to_only_liquid  = False

# if mask_to_ice_droplets:
#     mask_hydometeor = mask_ice_sup_water
#     name_hydrometeor = color_names[ICE_WITH_SUP_WATER]
# elif mask_to_only_ice:
#     mask_hydometeor = mask_ice
#     name_hydrometeor = color_names[ICE_PARTICLES]
# elif mask_to_only_liquid:
#     mask_hydometeor = mask_liquid
#     name_hydrometeor = color_names[CLOUD_LIQUID]

# # Have to store the index of the pixels that are ice with supercooled water
# dic_index = {}
# for i in range(mask_hydometeor.shape[0]):
#     # if there is any true value in the row, store the index
#     # Otherwise do not store anything
#     # the dic key should be the time in datetime format
#     if np.any(mask_hydometeor[i, :]):
#         index = np.where(mask_hydometeor[i, :])[0]
#         dic_index[starg_clas.time.values[i]] = index

# list_mask_time = list(dic_index.keys())
# # # --------------------------------------------------------------------------------------------
# i=1
# rini_absl = starg_clas.height[dic_index[list_mask_time[i]]].min()
# rend_absl = starg_clas.height[dic_index[list_mask_time[i]]].max()
# height_slice = slice(rini_absl, rend_absl)
# # height_slice = slice(4000, 6500)
# # get slice time 1 min before and after list_mask_time[i]
# # slice_time = slice(list_mask_time[i] - pd.Timedelta(minutes=1), list_mask_time[i] + pd.Timedelta(minutes=1)) 
# # slice_time = slice(datetime.datetime(2024, 4, 2, 17,40,45), datetime.datetime(2024, 4, 2, 17,42,0))
# # starg_clas.T.sel(time=slice_time, height=height_slice).plot(cmap=manual_cmap, figsize=(12, 5), vmin=0, vmax=ncolors)

# # starg_clas.where(mask_hydometeor).T.sel(time=slice_time, height=height_slice).plot(cmap=manual_cmap, figsize=(12, 5), vmin=0, vmax=ncolors)

# # range_targets = radar_lv0.data.range.values[dic_index[list_mask_time[i]]].tolist()
# # range_slice = (range_targets[0], range_targets[-1])
# # now the range slide 100m below and above the above range slide
# # range_slice = (range_slice[0]-100, range_slice[1]+100)
# # fig, filepath = radar_lv0.plot_2D_spectrum(
# #         variable='doppler_spectrum_dBZe',
# #         target_time=list_mask_time[i],
# #         range_limits=range_slice,
# #         vmin=-10,
# #         vmax=10,
# #         **{"savefig": False}
# #     )

# i=1
# cloud_targets_by_time = starg_clas.sel(time=list_mask_time[i]).dropna(dim='height')
# range_cloud_by_time = cloud_targets_by_time.height.values.tolist()
# fig, filepath = radar_lv0.plot_spectra_by_range_and_hydromet(
#                                                 target_time=list_mask_time[i],
#                                                 range_slice=range_cloud_by_time,
#                                                 targ_class=cloud_targets_by_time,
#                                                 output_dir = Path("./"),
#                                                 **{"savefig": True, 
#                                                    "velocity_limits": (-3, 3)}
#                                                 )
# fig, filepath = radar_lv0.plot_spectra_by_range(
#                                                 target_time=list_mask_time[i],
#                                                 range_slice=range_cloud_by_time,
#                                                 output_dir = Path("./"),
#                                                 **{"savefig": False, 
#                                                    "velocity_limits": (-3, 3)}
#                                                 )

# fig, filepath = radar_lv0.plot_2D_spectrum(
#         variable='doppler_spectrum_dBZe',
#         target_time= list_mask_time[i],
#         range_limits=(4000, 12000),
#         vmin=-50,
#         vmax=-1,
#         **{"savefig": False}
#     )
# # change axis x-limits using fig handle
# fig.axes[0].set_xlim(-3, 3)

# fig = plt.figure(figsize=(12, 5))
# starg_clas.T.plot(cmap=manual_cmap, figsize=(12, 5), vmin=0, vmax=ncolors)
# # plot an vertical line for list_mask_time[i]:
# plt.axvline(x=list_mask_time[i] + pd.Timedelta(seconds=15), color='b', linestyle='--')
# plt.ylim(4000, 12000)
# plt.show()

# # fig, filepath =  radar_lv0.plot_spectra_by_range(target_time=list_mask_time[i], 
# #                                                 range_slice=range_targets,
# #                                                 **{"savefig": False, 
# #                                                    "velocity_limits": (-1.5, 1)}
# #                                                 )

# # # Save all spectrums in a folder
# # folder_to_save = Path( os.path.join(f"../figures/dvs_comparison/{instrument_name}/{name_hydrometeor}", *date_list) )
# # os.makedirs(folder_to_save, exist_ok=True)

# # # for i in range(len(list_mask_time)):
# # #     range_targets = radar_lv0.data.range.values[dic_index[list_mask_time[i]]].tolist()
# # #     fig, filepath =  radar_lv0.plot_spectra_by_range(target_time=list_mask_time[i], 
# # #                                                     range_slice=range_targets,
# # #                                                     **{"savefig": True, 
# # #                                                     "velocity_limits": (-4, .5),
# # #                                                     "output_dir": folder_to_save}
# # #                                                     )
    
