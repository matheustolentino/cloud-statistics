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
import matplotlib
plot_inline = False

fontsize = 14
# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize

PATH_FIG          = '../figures/'
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
site= 'granada'
product_1 = 'classification'
product_2 = 'categorize'

process_all  = False
save_files = False

make_plot = True
show_figure = False

filter_abl_ice_clouds = True

if filter_abl_ice_clouds:
    print("Filtering ice clouds in the ABL")
else:
    path_save = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification_without_filtering_ice_ABL"
    print("*****************Warning**********************\nNot filtering ice clouds in the ABL")

if plot_inline and show_figure:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.run_line_magic('matplotlib', 'notebook')
else:
    plt.close('all')
    matplotlib.use('TkAgg')

case_to_only_save = False
if not process_all:
    path_to_data = "../../tests/data" # Path to save cloudnet downloaded files

    date_ini = "2023-05-06"
    date_end = "2023-05-06"
    date_end_new = date_end.replace("-", "")

    download_cloudnet_products(date_ini, date_end, path_to_data, product=product_1, site=site)
    download_cloudnet_products(date_ini, date_end, path_to_data, product=product_2, site=site)

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

        if n_pixels_inside_cloud == 0:  # Only rain
            ds_classification[cloud_number] = "Noise"
            idx_noise = np.unique(indx[:, 0])
            time_idx_noise[idx_noise] = 1
            cloud_type["Noise"][idx_noise] = 1
        else:
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

            if n_pixels_inside_cloud > cloud_min_pixels:
                valid_clouds.append(cloud_number)
                time_idx_valid_clouds.append(np.unique(indx_inside_cloud[cloud_number][:, 0]))
            else:
                ds_classification[cloud_number] = "Noise"
                idx_noise = np.unique(indx_inside_cloud[cloud_number][:, 0])
                time_idx_noise[idx_noise]      = 1
                cloud_type["Noise"][idx_noise] = 1

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
    
    show_category = False
    if not process_all or make_plot:

        # fig, ax = plt.subplots(figsize=(15, 9))
        # for cloud_number, indx in cloud_indx.items():
        # # for cloud_number in cloud_composition.keys:
        # #     indx= cloud_indx[cloud_number]
        #     x = data.time.values[indx[:, 0]]
        #     y = data.height.values[indx[:, 1]] / 1e3
        #     sc = ax.scatter(x, y, color=np.random.rand(3,), s=1)
        # ax.set_ylabel('Height (km) a.m.s.l')
        # ax.set_xlabel('Time (UTC) HH:MM')
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # ax.set_title(f"Cloud Classification {date_str} - {site}")
        # ax.grid()
        # plt.show()


        if show_category:
            fig = plt.figure(figsize=(16, 11))
            gs = fig.add_gridspec(3, 2, width_ratios=[1, .02], height_ratios=[1, 1, .3], wspace=0.05, hspace=0.08)
        else:
            fig = plt.figure(figsize=(15, 9))
            gs = fig.add_gridspec(3, 2, width_ratios=[1, .02], height_ratios=[1, 1, .5], wspace=0.05, hspace=0.05)

        ax1 = fig.add_subplot(gs[1, 0])
        pc0 = ax1.pcolormesh(ds_cloud['time'], ds_cloud['height']/1e3, ds_cloud['cloud_classification'].T, cmap=cloud_cmap, vmin=0, vmax=len(cloud_category))

        sc1 = ax1.scatter(cloud_props['time'], cloud_props['cloud_base']/1e3, s=10, color='r', label='Cloud base')
        sc2 = ax1.scatter(cloud_props['time'], cloud_props['cloud_top']/1e3, s=10, color='k', label='Cloud top')
        ax1.xaxis.set_tick_params(labelbottom=False)
        ax1.set_ylabel('Height (km) a.m.s.l')
        ax1.grid()
        # ax1.legend()

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
        ax2.set_ylabel('Height (km) a.m.s.l')
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

        if case_to_only_save:
            fig.savefig(PATH_FIG_TEST + f"{date_str}_cloud_classification.png", dpi=300, bbox_inches='tight')
            case_to_only_save = False
        if show_figure:
            plt.show()

        # Plotting ds_cloud
        fig = plt.figure(figsize=(13, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, .02], wspace=0.05)

        ax = fig.add_subplot(gs[0, 0])
        pc0 = ax.pcolormesh(ds_cloud['time'], ds_cloud['height']/1e3, ds_cloud['cloud_classification'].T, cmap=cloud_cmap, vmin=0, vmax=len(cloud_category))

        sc1 = ax.scatter(cloud_props['time'], cloud_props['cloud_base']/1e3, s=10, color='r', label='Cloud base')
        sc2 = ax.scatter(cloud_props['time'], cloud_props['cloud_top']/1e3, s=10, color='k', label='Cloud top')

        ax.set_ylabel('Height (km) a.m.s.l')
        ax.set_xlabel('Time (UTC)')
        ax.grid()
        # ax.legend()

        ax.set_xlim(data.time.values[0], data.time.values[-1])
        ax.set_ylim([0, data.height[-1]/1e3])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        cax_scat = fig.add_subplot(gs[0, 1])
        cbar_scat = plt.colorbar(pc0, cax=cax_scat, ticks=[], orientation='vertical')

        for idx, (color, name) in enumerate(zip(cloud_cmap.colors, cloud_category)):
            rect = plt.Rectangle((0, idx), 1, 1, color=color)
            cbar_scat.ax.add_patch(rect)
            cbar_scat.ax.text(1.5, idx + 0.5, name, color='black', va='center', fontsize=14)
        fig.savefig(PATH_FIG + f"{date_str}_cloud_classification_new_method.png", dpi=300, bbox_inches='tight')
        plt.show()
    
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
