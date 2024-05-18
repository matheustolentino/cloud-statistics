from phd_clouds.constants import CLEAR_SKY, CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS, ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS, AERO_NO_CLOUD, INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD, CLASSIFICATION_TICK_LABELS
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

fontsize = 14
# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize

plt.ion()
plt.close('all')

PATH_FIG          = '../figures/'
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
case_study  = True
if case_study:
    path_output = "../../tests/data" # Path to save cloudnet downloaded files
    site= 'granada'
    product_1 = 'classification'
    product_2 = 'categorize'
    date_ini = "2022-11-17"
    date_end = "2022-11-17"
    date_end_new = date_end.replace("-", "")

    download_cloudnet_products(date_ini, date_end, path_output, product=product_1, site=site)
    download_cloudnet_products(date_ini, date_end, path_output, product=product_2, site=site)

    # Load the NetCDF file
    filenames = [f"{date_end_new}_{site}_{product_1}.nc"]
    categorize = xr.open_dataset(os.path.join(path_output, f"{date_end_new}_{site}_{product_2}.nc"))
else:
    path_output = "/media/matheustolen/Seagate Basic/cloudnet/classification" # Path with files already downloaded
    # Get a linst of filenames of products
    filenames = os.listdir(path_output)

for file in filenames:
    # Load the downloaded with xarray pandas:
    data = xr.open_dataset(os.path.join(path_output, file))

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
    distance = 1
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

        hydromet_freq = {}
        for hydromet_name, hydromet_val in HYDROMET_VALUES.items():
            count = np.count_nonzero(hydro_inside_cloud == hydromet_val)
            if n_pixels_inside_cloud == 0:  # Only rain
                hydromet_freq[hydromet_name] = 0
            else:
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
            time_idx_noise[idx_noise] = 1
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

    previous_single_layer = 0
    # Iterating inside each valid cloud group:
    for key in valid_clouds:
        indx     =  indx_inside_cloud[key]
        time_idx = np.unique(indx[:, 0])

        for i in time_idx:
            mask_time             = indx[:, 0] == i
            height_idx            = indx[mask_time, :][:, 1]
            layers                = groupSequence(height_idx, 1)
            n_layers              = len(layers)

            if n_layers == 1: #single layer
                cloud_occurrence["single_layer"][i] = 1
                cloud_props["cloud_base"][i] = data.height[layers[0][0]]
                cloud_props["cloud_top"][i]  = data.height[layers[0][-1]]
                cloud_props["cloud_thickness"][i] = cloud_props["cloud_top"][i] - cloud_props["cloud_base"][i]

                # cloud_type[i] = dic_acronyms[ds_classification[key]]
            # elif n_layers > 1: #multi layer
            #     real_n_layers = n_layers
            #     real_layers   = layers
            #     for layer in layers:
            #         hydromet_sequence = np.unique(cloud_classification.values[i, layer])
            #         unique_hydro_layer_type  = hydromet_sequence[0]
            #         if hydromet_sequence.shape[0] == 1 and unique_hydro_layer_type == DRIZZLE_OR_RAIN_LIQUID_DROPLETS:
            #             real_n_layers = n_layers - 1
            #             real_layers.remove(layer)
            #     if real_n_layers == 1:
            #         cloud_occurrence["single_layer"][i] = 1
            #         cloud_props["cloud_base"][i] = data.height[real_layers[0][0]]
            #         cloud_props["cloud_top"][i]  = data.height[real_layers[0][-1]]
            #         cloud_props["cloud_thickness"][i] = cloud_props["cloud_top"][i] - cloud_props["cloud_base"][i]
            #     else:
            #         cloud_occurrence["multi_layer"][i] = 1
            else:
                cloud_occurrence["multi_layer"][i] = 1
                # cloud_type[i] = "ML"

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

    # cloud_occurrence.coords['cloud_type'] = ('time', cloud_type)
    # cloud_props.coords['cloud_type']      = ('time', cloud_type)
    # Save cloud occurrence as netCDF
    date_str = cloud_occurrence.time[0].dt.strftime('%Y%m%d').values.item()
    cloud_occurrence.to_netcdf(path_save + f"/cloud_occurence/{date_str}_cloud_occurrence.nc")
    cloud_props.to_netcdf(path_save + f"/cloud_properties/{date_str}_cloud_props.nc")
    cloud_type.to_netcdf(path_save + f"/cloud_type/{date_str}_cloud_type.nc")

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
    if case_study:

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

        fig.savefig(PATH_FIG + f"{date_str}_cloud_classification.png", dpi=300, bbox_inches='tight')
        plt.show()


    # number_of_layers = pd.DataFrame(0, index=cloud_classification['time'],
    #                                 columns=["Liquid", "Ice", "Mixed_phase", "Pre_liquid", "Pre_mixed_phase"])

    # height_cloud_base         = pd.DataFrame(index=cloud_classification['time'],
    #                                     columns=["Liquid", "Ice", "Mixed_phase", "Pre_liquid", "Pre_mixed_phase"])
    # height_cloud_top          = pd.DataFrame(index=cloud_classification['time'],
    #                                     columns=["Liquid", "Ice", "Mixed_phase", "Pre_liquid", "Pre_mixed_phase"])


    # groups_height_idx = groupSequence([indx[:,1] for indx in cloud_indx.values()])
