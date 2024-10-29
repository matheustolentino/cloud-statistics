import xarray as xr
import numpy as np
import os
import dask
import matplotlib.pyplot as plt
from pdb import set_trace
import pandas as pd
from phd_clouds.constants import GRANADA_ALTITUDE, SEASONS # in meters
from phd_clouds.utils import get_complete_time, assign_season
import matplotlib.dates as mdates
from scipy import stats
import seaborn as sns
import dask.dataframe as dd
import matplotlib as mpl
from IPython import get_ipython
import matplotlib.animation as animation
import pickle

# ipython = get_ipython()
# if ipython is not None:
#     ipython.run_line_magic('matplotlib', 'inline')

PATH_FIG          = '../figures/'
PATH_FIG_PICKLE   = '../figures/pickle/'
PATH_SAVE_DATA    = '../data/'
fontsize = 14
# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize

# plt.ion()
plt.close('all')
# ipython.run_line_magic('matplotlib', 'notebook')
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

def plot_microphysics_evolution(sliced_microphys, sliced_integrated=None, cloud_type=None, var_short_name='lwc'):

    if sliced_integrated is not None:
        number_profiles = sliced_integrated.groupby("time.month").count(dim='time')
    else:
        number_profiles = sliced_microphys.max(dim="height", skipna=True).groupby("time.month").count()
    
    # breakpoint()
    mean_by_month    = sliced_microphys.groupby('time.month').median(dim='time', skipna=True).compute()
    # std_by_month     = sliced_microphys.groupby('time.month').std(dim='time', skipna=True).compute()
    mean_by_season   = sliced_microphys.groupby('time.season').median(dim='time', skipna=True).compute()
    # std_by_season    = sliced_microphys.groupby('time.season').std(dim='time', skipna=True).compute()
    
    cflevels = 10
    list_seasons =list(mean_by_season.season.values)
    if var_short_name == 'lwc':
        clevels = [1000, 1500]
        if cloud_type == 'Liquid' or cloud_type == 'Liquid-Precipitable':
            zlim    = [0, 7000]
            cflevels = np.arange(0, 500, 10)
        elif cloud_type == 'Mixed-Phase' or cloud_type == 'Mixed-Phase-Precipitable':
            zlim    = [0, 13000]
            cflevels = np.arange(0, 500, 10)
        elif cloud_type == 'Ice' or cloud_type == 'Ice-Precipitable':
            zlim    = [0, 13000]
            cflevels = np.arange(0, 500, 10)
    elif var_short_name == 'iwc':
        clevels = [50, 200]
        if cloud_type == 'Liquid' or cloud_type == 'Liquid-Precipitable':
            zlim    = [0, 8000]
            cflevels = np.arange(0, 500, 10)
        elif cloud_type == 'Mixed-Phase' or cloud_type == 'Mixed-Phase-Precipitable':
            zlim    = [0, 12000]
            cflevels = np.arange(0, 150, 10)
        elif cloud_type == 'Ice' or cloud_type == 'Ice-Precipitable':
            zlim    = [0, 13000]
            cflevels = np.arange(0, 150, 10)

    elif var_short_name == 'der':
        clevels = [20, 30]
        zlim = [0, 12000]
    elif var_short_name == 'ier':
        clevels = [30, 70]
        zlim = [0, 13000]
    
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1., 5, .15], height_ratios=[3, 1.2], hspace=0.1, wspace=0.05)
    ax = fig.add_subplot(gs[0, 1])
    # mesh = ax.pcolormesh(mean_by_month.month.values, mean_by_month.height.values,
    #                 mean_by_month.T, shading='nearest', cmap=sns.color_palette("icefire", as_cmap=True), 
    #                 vmin=cflevels[0], vmax=cflevels[-1])
    cf = ax.contourf(mean_by_month.month.values, mean_by_month.height.values,
                    mean_by_month.values.T, levels=cflevels, cmap=sns.color_palette("icefire", as_cmap=True))
    # Plot a heatmap
    # cf = ax.pcolormesh(mean_by_month.month.values, mean_by_month.height.values,
    #                 mean_by_month.values.T, shading='nearest', cmap="icefire",
    #                 vmin=cflevels[0], vmax=cflevels[-1])
    
    countour = ax.contour(mean_by_month.month.values, mean_by_month.height.values,
                    mean_by_month.values.T, levels=clevels, colors='white', linewidths=2)
    ax.clabel(countour, inline=True, fontsize=10)
    
    ax.set_xlabel("Time")
    ax.set_title(f"{sliced_microphys.attrs['long_name']} for {cloud_type} Clouds")
    ax.grid(True)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.xaxis.set_visible(False)
    # ax.set_xticks(np.arange(1, 13))
    ax.set_ylim(zlim)

    cax = fig.add_subplot(gs[0, 2])
    # cbar1 = plt.colorbar(mesh, cax=cax, orientation='vertical', label=f"{sliced_microphys.attrs['units']}")
    cbar1 = plt.colorbar(cf, cax=cax, orientation='vertical', label=f"{sliced_microphys.attrs['units']}")
    cbar1.ax.yaxis.set_label_position('right')

    ax2 = fig.add_subplot(gs[:1, 0], sharey=ax)
    max_value = np.nanmax(mean_by_season)
    for season in list_seasons:
        mean_profile = mean_by_season.sel(season=season)
        # std_profile = std_by_season.sel(season=season)
        ax2.plot(mean_profile, mean_profile.height, label=f"{season}")
        # ax2.fill_betweenx(mean_profile.height, mean_profile - std_profile, mean_profile + std_profile, alpha=0.3)
    ax2.set_xlabel(f"{sliced_microphys.attrs['long_name']} ({sliced_microphys.attrs['units']})")
    ax2.set_ylabel("Height (m)")
    # ax2.set_xlim([0, max_value + .1])
    # ax2.set_ylim([0, np.max(mean_by_month.height.values)])
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 1], sharex=ax)
    unique_months = np.unique(mean_by_month.month.values)
    for month in unique_months:

        if var_short_name == 'lwc' or var_short_name == 'iwc':
            monthly_integrated = sliced_integrated.sel(time=sliced_integrated['time.month'] == month)
            # removing NaNs from monthly data
            monthly_integrated = monthly_integrated.dropna(dim='time', how='all').values
            ax3.boxplot(monthly_integrated, positions=[month], showfliers=False, showmeans=True, patch_artist=True, widths=0.8,
                            meanprops=dict(marker='*', markerfacecolor='black', markeredgecolor='black'),
                            medianprops=dict(color='red', linewidth=1.5), boxprops=dict(facecolor='lightblue', color='black'))
        else:
            # monthly_microphys = sliced_microphys.sel(time=sliced_microphys['time.month'] == month)
            # # removing NaNs from monthly data
            # monthly_microphys = monthly_microphys.dropna(dim='time', how='all').values.ravel()
            # ax3.violinplot(monthly_microphys, positions=[month], showmeans=True, 
            #                showmedians=True, showextrema=False, widths=0.8)
            # breakpoint()
            monthly_microphys = sliced_microphys.sel(time=sliced_microphys['time.month'] == month)
            # removing NaNs from monthly data
            monthly_microphys = monthly_microphys.dropna(dim='time', how='all').values.ravel()
            monthly_microphys = monthly_microphys[~np.isnan(monthly_microphys)]
            ax3.violinplot(monthly_microphys, positions=[month], showmeans=False, 
                            showmedians=True, showextrema=False, widths=0.8)
            
            # MAke a violin plot, and show the means with a different marker style
            # ax3.violinplot(monthly_microphys, positions=[month], showmeans=True, showmedians=True, showextrema=False, widths=0.8,
            #                 meanline=True, meanprops=dict(marker='*', markerfacecolor='black', markeredgecolor='black'),
            #                 medianprops=dict(color='red', linewidth=1.5), boxprops=dict(facecolor='lightblue', color='black'))
            # ax3.set_ylabel(f"{chunked_microphys.attrs['long_name']} ({chunked_microphys.attrs['units']})")
            # ax3.set_xlabel("Months")
            # ax3.set_xticks(np.arange(1, 13))
            # ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            # ax3.grid(True, axis='y')
            # plt.show()

    if var_short_name == 'lwc' or var_short_name == 'iwc':
        ax3.set_ylabel(f"{sliced_integrated.attrs['long_name']} ({sliced_integrated.attrs['units']})")
    else:
        ax3.set_ylabel(f"{sliced_microphys.attrs['long_name']} ({sliced_microphys.attrs['units']})")
    ax3.set_xlabel("Months")
    ax3.set_xticks(np.arange(1, 13))
    ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    # ax3.set_ylim([-.02, 0.10])
    ax3.grid(True, axis='y')
    
    ax3_right = ax3.twinx()
    ax3_right.plot(number_profiles.month, number_profiles.values, '--', linewidth=1, color='blue')  # Set zorder to 0
    ax3_right.set_ylabel(r"N$_{Profiles}$", color="blue")
    ax3_right.tick_params(axis='y', colors='blue')
    # ax3_right.set_yticks(np.arange(np.min(number_profiles), np.max(number_profiles), 1000))

    ax3.spines['top'].set_visible(False)  # Remove the top spine
    ax3_right.spines['top'].set_visible(False)  # Remove the top spine
    ax3_right.spines['right'].set_color('blue')  # Set the color of the right spine to blue

    fig.savefig(f"{PATH_FIG}grouped_by_month_evolution_for_{get_var}_{cloud_type}_mesh.png", dpi=300, bbox_inches='tight')
    plt.show()

    with open(f"{PATH_FIG_PICKLE}grouped_by_month_evolution_for_{get_var}_{cloud_type}_mesh.pkl", "wb") as f:
        pickle.dump(fig, f)

    # # save all data as netCDF
    # if var_short_name == 'lwc' or var_short_name == 'iwc':
    #     mean_by_season.to_netcdf(f"{PATH_SAVE_DATA}mean_by_season_{get_var}_{cloud_type}.nc")
    #     mean_by_month.to_netcdf(f"{PATH_SAVE_DATA}mean_by_month_{get_var}_{cloud_type}.nc")
    return fig

# ------------------------------------------------------------------------------------------------------------------------------




# ------------------------------------------------------------------------------------------------------------------------------

# Function to group by height bins and compute mean within each bin
def groupby_bins_mean(ds, bin_edges):
    return ds.groupby_bins('height', bin_edges, labels=bin_edges[1:]).mean()

def open_dataset(file_path):
    return xr.open_dataset(file_path, chunks={'time': 1})

filepath_cloud_occurence = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification/cloud_occurence" # Path to save the cloud classification files
filepath_cloud_cloud_type = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification/cloud_type" # Path to save the cloud classification files
filepath_microphys        = "/home/matheustolen/Documentos/matheus_doctorado/output_retrievals" # Path to save the cloud classification files
get_var = "ier"

if get_var == "lwc":
    endswith = "lwc-scaled-adiabatic.nc"
elif get_var == "iwc":
    endswith = "iwc-Z-T-method.nc"
elif get_var == "der":
    endswith = "der.nc"
elif get_var == "ier":
    endswith = "ier.nc"
else:
    raise ValueError("Variable not recognized")

load_data = True # Set to True if you want to load the data from the netCDF files

if load_data:
    # Get the list of netCDF files in the specified directory
    file_paths_occurence = [os.path.join(filepath_cloud_occurence, file) for file in os.listdir(filepath_cloud_occurence) if file.endswith('.nc')]
    file_paths_cloud_type = [os.path.join(filepath_cloud_cloud_type, file) for file in os.listdir(filepath_cloud_cloud_type) if file.endswith('.nc')]
    filepaths_microphys = [os.path.join(filepath_microphys, file) for file in os.listdir(filepath_microphys) if file.endswith(endswith)] 
                           
    # Read the netCDF files into a list of xarray datasets
    datasets_occurence = [xr.open_dataset(file_path, engine='netcdf4', chunks={'time': -1}) for file_path in file_paths_occurence]
    datasets_cloud_type = [xr.open_dataset(file_path, engine='netcdf4', chunks={'time': -1}) for file_path in file_paths_cloud_type]
    
    print("Concatenating cloud occurene and cloud type datasets")
    # Concatenate the datasets along the time dimension
    chunked_occurence  = xr.concat(datasets_occurence, dim='time').sortby('time')
    chunked_cloud_type = xr.concat(datasets_cloud_type, dim='time').sortby('time')
    print("End of concatenation")

    dz = 20
    new_height = np.arange(0, 14000+dz, dz)
    
    print("Reading microphysics files")
    datasets_microphys = [xr.open_dataset(file_path, engine='netcdf4', chunks={'time': -1})[get_var].assign_coords(height=lambda ds: ds.height - GRANADA_ALTITUDE) for file_path in filepaths_microphys]
    
    print("Interpolating microphysics datasets")
    # Set the same height resolution for all datasets
    datasets_microphys = [ds.interp(height=new_height) for ds in datasets_microphys]
    #integrating the microphysics variable
    if get_var == "lwc" or get_var == "iwc":
        datasets_integrated = [ds.fillna(0.).integrate('height') for ds in datasets_microphys]
        min_dataset         = [ds.min(dim='height', skipna=True) for ds in datasets_microphys]
        chunked_integrated = xr.concat(datasets_integrated, dim='time').sortby('time')
        chunked_integrated.chunk({'time': 'auto'})
    
    print("Concatenating microphysics datasets")
    # Concatenate the interpolated datasets along the time dimension
    chunked_microphys = xr.concat(datasets_microphys, dim='time').sortby('time')
    
    chunked_occurence.chunk({'time': 'auto'})
    chunked_cloud_type.chunk({'time': 'auto'})
    chunked_microphys.chunk({'time': 'auto'})
    
    if get_var == "lwc":
        chunked_microphys = chunked_microphys*1e6
        chunked_microphys.attrs['units'] = 'mg m$^{-3}$'
        chunked_microphys.attrs['long_name'] = 'LWC'
        chunked_integrated = chunked_integrated*1e3
        chunked_integrated.attrs['units'] = 'g m$^{-2}$'
        chunked_integrated.attrs['long_name'] = 'LWP'
    elif get_var == "iwc":
        chunked_microphys = chunked_microphys*1e6
        chunked_microphys.attrs['units'] = 'mg m$^{-3}$'
        chunked_microphys.attrs['long_name'] = 'IWC'
        chunked_integrated = chunked_integrated*1e3
        chunked_integrated.attrs['units'] = 'g m$^{-2}$'
        chunked_integrated.attrs['long_name'] = 'IWP'
    elif get_var == "der":
        chunked_microphys = chunked_microphys*1e6
        chunked_microphys.attrs['units'] = '$\mu m$'
        chunked_microphys.attrs['long_name'] = 'Droplet Effective Radius'
    elif get_var == "ier":  
        chunked_microphys = chunked_microphys*1e6
        chunked_microphys.attrs['units'] = '$\mu m$'
        chunked_microphys.attrs['long_name'] = 'Ice Effective Radius'
    print("End of concatenation")

cloud_type_to_analise = 'Liquid'
ds_cloud_occurence = chunked_occurence.compute()
ds_cloud_type = chunked_cloud_type.compute()

condition_layer = (ds_cloud_occurence['single_layer'] == 1) & (ds_cloud_occurence['noise'] == 0)

if get_var == "lwc" or get_var == "iwc":
    chunked_integrated_single_layer = chunked_integrated.where(condition_layer)
    ds_integrated_single_layer = chunked_integrated_single_layer.compute()
    chunked_integrated_cloud_type = chunked_integrated_single_layer.where(ds_cloud_type[cloud_type_to_analise].astype(bool))
    ds_integrated = chunked_integrated_cloud_type.groupby('time.month').mean(dim='time', skipna=True).compute()

chunked_microphys = chunked_microphys.where(chunked_microphys > 0)
# ------------------------------------------------------------------------------------------------------------------------------
# Some verification plots
# ------------------------------------------------------------------------------------------------------------------------------
condition_test = (ds_cloud_occurence['single_layer'] == 1) & (ds_cloud_occurence['noise'] == 0) & (ds_cloud_type['Ice-Precipitable'].astype(bool))
test_microphys       = chunked_microphys.sel(time=condition_test)
if get_var == "lwc" or get_var == "iwc":
    test_integ_microphys = chunked_integrated.where(condition_layer)

# start_time = "2023-10-01"
# end_time   = "2023-10-30"

# sliced_microphys = test_microphys.sel(time=slice(start_time, end_time)).compute()
# indxs = np.unique(sliced_microphys.time.dt.date)
# i = 10
# # Plot the pcolormesh for the first day
# data = sliced_microphys.sel(time=indxs[i].strftime('%Y-%m-%d'))

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# # Plot the first subplot
# mesh1 = ax1.pcolormesh(data.time, data.height,
#                 data.T, shading='nearest', cmap='viridis')
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Height (m)")
# ax1.grid(True)
# # show x-axis in the format HH:MM
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# ax1.set_title(f"{indxs[i].strftime('%Y-%m-%d')}")

# data2 = data.where(data > 0)
# # Plot the second subplot
# mesh2 = ax2.pcolormesh(data2.time, data2.height,
#                 data2.T, shading='nearest', cmap='viridis')
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Height (m)")
# ax2.grid(True)
# # show x-axis in the format HH:MM
# ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# ax2.set_title(f"{indxs[i].strftime('%Y-%m-%d')}")

# # Add colorbars to both subplots
# cbar1 = plt.colorbar(mesh1, ax=ax1, label=f"{sliced_microphys.attrs['units']}")
# cbar2 = plt.colorbar(mesh2, ax=ax2, label=f"{sliced_microphys.attrs['units']}")

# plt.show()

# list_var = list(ds_cloud_type.data_vars)
# fig = plt.figure(figsize=(19, 12))
# gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], hspace=0.4, wspace=0.25)
# for i, cloud in enumerate(list_var[:-1]):
#     data = test_integ_microphys.where(ds_cloud_type[cloud].astype(bool)).compute()
#     ax = fig.add_subplot(gs[i])
#     for year in np.unique(data.time.dt.year):
#         data_year = data.sel(time=data.time.dt.year == year)
#         daily_data = data_year.groupby('time.dayofyear').median(dim='time', skipna=True)
#         ax.plot(daily_data.dayofyear, daily_data,'--o',  label=f"{year}")
#     ax.set_xlabel("Day of Year")
#     ax.set_ylabel(cloud)
#     ax.set_title(f"{cloud} Clouds")
#     if i == 0:
#         ax.legend()
# plt.show()


#------------------------------------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------------------------------------------------
# # removing zeros
# ds_microphys_nozero        = test_microphys.where(test_microphys > 0)

# ds_microphys_median_nozero = ds_microphys_nozero.groupby('time.month').median(dim='time', skipna=True).compute()
# ds_microphys_median_nozero.attrs['long_name'] = 'Median ' + ds_microphys_median_nozero.attrs['long_name'] + ' without zeros'
# ds_microphys_median_withzero = test_microphys.groupby('time.month').median(dim='time', skipna=True).compute()
# ds_microphys_median_withzero.attrs['long_name'] = 'Median ' + ds_microphys_median_withzero.attrs['long_name'] + ' with zeros'

# ds_microphys_mean_nozero = ds_microphys_nozero.groupby('time.month').mean(dim='time', skipna=True).compute()
# ds_microphys_mean_nozero.attrs['long_name'] = 'Mean ' + ds_microphys_mean_nozero.attrs['long_name'] + ' without zeros'
# ds_microphys_mean_withzero = test_microphys.groupby('time.month').mean(dim='time', skipna=True).compute()
# ds_microphys_mean_withzero.attrs['long_name'] = 'Mean ' + ds_microphys_mean_withzero.attrs['long_name'] + ' with zeros'

# # Make a subplots of poclormesh for median with zeros, median with zeros, mean with zeros and mean without zeros
# fig = plt.figure(figsize=(15, 8))
# gs = fig.add_gridspec(2, 4, width_ratios=[1, .0, 1, .05], hspace=0.4, wspace=0.1)

# ax = fig.add_subplot(gs[0, 0])
# mesh = ax.pcolormesh(ds_microphys_median_nozero.month, ds_microphys_median_nozero.height,
#                 ds_microphys_median_nozero.values.T, shading='nearest', cmap='icefire')
# ax.set_xlabel("Time")
# ax.set_ylabel("Height (m)")
# ax.set_title(f"{ds_microphys_median_nozero.attrs['long_name']}")
# ax.xaxis.set_tick_params(labelleft=False)
# ax.set_ylim([0, 6000])
# ax.grid(True)

# # cax = fig.add_subplot(gs[0, 1])
# # cbar1 = plt.colorbar(mesh, cax=cax, orientation='vertical', label=f"{ds_microphys_median_nozero.attrs['units']}")
# # cbar1.ax.yaxis.set_label_position('right')

# ax2 = fig.add_subplot(gs[1, 0], sharey=ax)
# mesh1 = ax2.pcolormesh(ds_microphys_mean_nozero.month, ds_microphys_mean_nozero.height,
#                 ds_microphys_mean_nozero.values.T, shading='nearest', cmap='icefire')
# ax2.set_xlabel("Time")
# ax.set_ylabel("Height (m)")
# ax2.set_title(f"{ds_microphys_mean_nozero.attrs['long_name']}")
# ax2.yaxis.set_tick_params(labelleft=False)
# ax2.grid(True)

# # cax2 = fig.add_subplot(gs[1, 1])
# # cbar2 = plt.colorbar(mesh1, cax=cax2, orientation='vertical', label=f"{ds_microphys_mean_nozero.attrs['units']}")
# # cbar2.ax.yaxis.set_label_position('right')

# ax3 = fig.add_subplot(gs[0, 2], sharey=ax)
# mesh2 = ax3.pcolormesh(ds_microphys_median_withzero.month, ds_microphys_median_withzero.height,
#                 ds_microphys_median_withzero.values.T, shading='nearest', cmap='icefire')
# ax3.set_xlabel("Time")
# ax3.set_title(f"{ds_microphys_median_withzero.attrs['long_name']}")
# ax3.yaxis.set_tick_params(labelleft=False)
# ax3.grid(True)

# cax3 = fig.add_subplot(gs[0, 3])
# cbar3 = plt.colorbar(mesh2, cax=cax3, orientation='vertical', label=f"{ds_microphys_median_withzero.attrs['units']}")
# cbar3.ax.yaxis.set_label_position('right')

# ax4 = fig.add_subplot(gs[1, 2], sharey=ax)
# mesh3 = ax4.pcolormesh(ds_microphys_mean_withzero.month, ds_microphys_mean_withzero.height,
#                 ds_microphys_mean_withzero.values.T, shading='nearest', cmap='icefire')
# ax4.set_xlabel("Time")
# ax4.set_title(f"{ds_microphys_mean_withzero.attrs['long_name']}")
# ax4.yaxis.set_tick_params(labelleft=False)
# ax4.grid(True)

# cax4 = fig.add_subplot(gs[1, 3])
# cbar4 = plt.colorbar(mesh3, cax=cax4, orientation='vertical', label=f"{ds_microphys_mean_withzero.attrs['units']}")
# cbar4.ax.yaxis.set_label_position('right')

# plt.suptitle(f"{cloud} Clouds")

# plt.show()
# ------------------------------------------------------------------------------------------------------------------------------

# # # ------------------------------------------------------------------------------------------------------------------------------
# test_microphys = test_microphys.coarsen(height = 5, boundary='trim').mean()
ds_microphys_season = test_microphys.groupby('time.season').median(dim='time', skipna=True).compute()
# percentile_25 = test_microphys.groupby('time.season').quantile(0.25, dim='time', skipna=True).compute()
# percentile_75 = test_microphys.groupby('time.season').quantile(0.75, dim='time', skipna=True).compute()
monthly_microphys = test_microphys.groupby('time.month').median(dim='time', skipna=True).compute()
# monthly_microphys = test_microphys.resample(time='1M').median(dim='time', skipna=True).compute()

fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 6], hspace=0.1, wspace=0.1)
ax = fig.add_subplot(gs[0, 1])
# Plot the mean profile of the microphysics variable
# mesh = ax.pcolormesh(monthly_microphys.time, monthly_microphys.height, 
#               monthly_microphys.T.values, shading='nearest', 
#               cmap=sns.color_palette("icefire", as_cmap=True), 
#               vmin=0, vmax=100)
# mesh = ax.pcolormesh(monthly_microphys.month, monthly_microphys.height,
#                 monthly_microphys.values.T, shading='nearest', 
#                 cmap=sns.color_palette("icefire", as_cmap=True),
#                 vmin=0, vmax=100)
mesh = ax.contourf(monthly_microphys.month, monthly_microphys.height,
                monthly_microphys.values.T, levels=[2, 4, 8, 16, 20, 30, 40, 50, 60, 80, 100],
                  cmap=sns.color_palette("icefire", as_cmap=True),
                  origin = 'upper',
                  extend='max')

countour = ax.contour(monthly_microphys.month, monthly_microphys.height,
                monthly_microphys.values.T, levels=[1000], colors='white')

# mesh = ax.contourf(monthly_microphys.time, monthly_microphys.height,
#                 monthly_microphys.values.T, levels=20, cmap=sns.color_palette("icefire", as_cmap=True))

# countour = ax.contour(monthly_microphys.time, monthly_microphys.height,
#                 monthly_microphys.values.T, levels=[1000], colors='black')
ax.set_xlabel('Time')
ax.set_title(f'Mean {monthly_microphys.attrs["long_name"]}')
ax.yaxis.set_tick_params(labelleft=False)
# ax.set_ylim([0, 4000])
ax.grid(True)

ax.clabel(countour, inline=True, fontsize=10)

cbar = fig.colorbar(mesh, ax=ax, label=r"mg m$^{-3}$", aspect=10)
# cbar = fig.colorbar(mesh, ax=ax, label=f"{monthly_microphys.attrs['units']}", aspect=10)
# cbar.set_ticks(np.linspace(0, 0.1, 11))

list_seasons =list(ds_microphys_season.season.values)

ax2 = fig.add_subplot(gs[0, 0], sharey=ax)
max_value = np.nanmax(ds_microphys_season)
for season in list_seasons:
    mean_profile = ds_microphys_season.sel(season=season)
    # std_profile = std_by_season.sel(season=season)
    ax2.plot(mean_profile, mean_profile.height, label=f"{season}")
    # ax2.fill_betweenx(mean_profile.height, mean_profile - std_profile, mean_profile + std_profile, alpha=0.3)
ax2.set_xlabel(f"{ds_microphys_season.attrs['long_name']} ({ds_microphys_season.attrs['units']})")
ax2.set_ylabel("Height (m)")
# ax2.set_xlim([0, max_value + .1])
# ax2.set_ylim([0, np.max(mean_by_month.height.values)])
ax2.grid(True)
ax2.legend()

plt.show()
# ------------------------------------------------------------------------------------------------------------------------------


# unique_months = np.unique(test_microphys['time.month'].values)
# fig, ax = plt.subplots(figsize=(12, 6))
# for month in unique_months:
#     monthly_microphys = test_microphys.sel(time=test_microphys['time.month'] == month)
#     # removing NaNs from monthly data
#     monthly_microphys = monthly_microphys.dropna(dim='time', how='all').values.ravel()
#     monthly_microphys = monthly_microphys[~np.isnan(monthly_microphys)]
#     ax.violinplot(monthly_microphys, positions=[month], showmeans=False, 
#                     showmedians=True, showextrema=False, widths=0.8)
# ax.set_ylabel(f"{chunked_microphys.attrs['long_name']} ({chunked_microphys.attrs['units']})")
# ax.set_xlabel("Months")
# ax.set_xticks(np.arange(1, 13))
# ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# ax.grid(True, axis='y')
# plt.show()

# sliced_microphys = test_microphys.sel(time=slice('2023-01-01', '2023-12-31'))
# sliced_integrated = chunked_integrated_single_layer.sel(time=slice('2023-01-01', '2023-12-31')).compute()

# fig_microphy = plot_microphysics_evolution(sliced_microphys, sliced_integrated, cloud_type=cloud_type_to_analise)
# # get axis
# ax = fig_microphy.get_axes()


list_figs = []
if get_var == "lwc" or get_var == "der":
    clouds_to_analyse = ['Liquid', 'Liquid-Precipitable', 'Mixed-Phase', 'Mixed-Phase-Precipitable', 'Ice-Precipitable']
elif get_var == "iwc" or get_var == "ier":
    clouds_to_analyse = ['Ice', 'Ice-Precipitable', 'Mixed-Phase', 'Mixed-Phase-Precipitable']

for cloud in clouds_to_analyse:
    condition       = (ds_cloud_occurence['single_layer'] == 1) & (ds_cloud_occurence['noise'] == 0) & (ds_cloud_type[cloud].astype(bool))
    cloud_microphys = chunked_microphys.sel(time=condition)
    
    if get_var == "lwc" or get_var == "iwc":
        integrated_microphys = chunked_integrated.sel(time=condition)
        
        # ----------------------------------------------------------------------------
        # Just for fast verification (Comment this part when running for all database)
        # ----------------------------------------------------------------------------
        # start = "2020-01-01"
        # end   = "2020-12-30"
        # cloud_microphys      = cloud_microphys.sel(time=slice(start, end))
        # integrated_microphys = integrated_microphys.sel(time=slice(start, end))
        # ----------------------------------------------------------------------------

        fig_microphy = plot_microphysics_evolution(cloud_microphys, 
                                               integrated_microphys, 
                                               cloud_type=cloud, 
                                               var_short_name=get_var)
    else:
        fig_microphy = plot_microphysics_evolution(cloud_microphys, 
                                               cloud_type=cloud, 
                                               var_short_name=get_var)
    
    list_figs.append(fig_microphy)
    fig_microphy.savefig(f"{PATH_FIG}grouped_by_month_evolution_for_{get_var}_{cloud}.png", dpi=300, bbox_inches='tight')

# save all
for i, fig in enumerate(list_figs):
    fig.savefig(f"{PATH_FIG}grouped_by_month_evolution_for_{get_var}_{clouds_to_analyse[i]}.png", dpi=300, bbox_inches='tight')

# axes = [fig.get_axes() for fig in list_figs]


