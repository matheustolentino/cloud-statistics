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
# ipython = get_ipython()
# if ipython is not None:
#     ipython.run_line_magic('matplotlib', 'inline')

PATH_FIG          = '../figures/'
fontsize = 14
# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize

# plt.ion()
plt.close('all')
# ipython.run_line_magic('matplotlib', 'notebook')
mpl.use('TkAgg')

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
    sliced_microphys = sliced_microphys.where(ds_cloud_type[cloud_type].astype(bool))
    if sliced_integrated is not None:
        sliced_integrated = sliced_integrated.where(ds_cloud_type[cloud_type].astype(bool)).compute()
        number_profiles = sliced_integrated.groupby("time.month").count(dim='time')
    else:
        number_profiles = sliced_microphys.max(dim="height").groupby("time.month").count()
    
    mean_by_month    = sliced_microphys.groupby('time.month').mean(dim='time', skipna=True).compute()
    # std_by_month     = sliced_microphys.groupby('time.month').std(dim='time', skipna=True).compute()
    mean_by_season   = sliced_microphys.groupby('time.season').mean(dim='time', skipna=True).compute()
    # std_by_season    = sliced_microphys.groupby('time.season').std(dim='time', skipna=True).compute()

    list_seasons =list(mean_by_season.season.values)
    if var_short_name == 'lwc' or var_short_name == 'iwc':
        clevels = [100, 500]
        zlim = [0, 12000]
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
    #                 mean_by_month.T, shading='nearest', cmap='turbo', vmin=0.001, vmax=.1)
    cf = ax.contourf(mean_by_month.month.values, mean_by_month.height.values,
                    mean_by_month.values.T, levels=10, cmap=sns.color_palette("coolwarm", as_cmap=True))
    countour = ax.contour(mean_by_month.month.values, mean_by_month.height.values,
                    mean_by_month.values.T, levels=clevels, colors='black')
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

    fig.savefig(f"{PATH_FIG}grouped_by_month_evolution_for_{get_var}_{cloud_type}.png", dpi=300, bbox_inches='tight')
    plt.show()
    return fig

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

    dz = 30
    new_height = np.arange(0, 14000+dz, dz)
    
    print("Reading microphysics files")
    datasets_microphys = [xr.open_dataset(file_path, engine='netcdf4', chunks={'time': -1})[get_var].assign_coords(height=lambda ds: ds.height - GRANADA_ALTITUDE) for file_path in filepaths_microphys]
    
    print("Interpolating microphysics datasets")
    # Set the same height resolution for all datasets
    datasets_microphys = [ds.interp(height=new_height) for ds in datasets_microphys]
    #integrating the microphysics variable
    if get_var == "lwc" or get_var == "iwc":
        datasets_integrated = [ds.fillna(0.).integrate('height') for ds in datasets_microphys]
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
        chunked_integrated = chunked_integrated*1e6
        chunked_integrated.attrs['units'] = 'mg m$^{-2}$'
        chunked_integrated.attrs['long_name'] = 'LWP'
    elif get_var == "iwc":
        chunked_microphys = chunked_microphys*1e6
        chunked_microphys.attrs['units'] = 'mg m$^{-3}$'
        chunked_microphys.attrs['long_name'] = 'IWC'
        chunked_integrated = chunked_integrated*1e6
        chunked_integrated.attrs['units'] = 'mg m$^{-2}$'
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


cloud_type_to_analise = 'Mixed-Phase'
ds_cloud_occurence = chunked_occurence.compute()
ds_cloud_type = chunked_cloud_type.compute()

condition_layer = (ds_cloud_occurence['single_layer'] == 1) & (ds_cloud_occurence['noise'] == 0)

if get_var == "lwc" or get_var == "iwc":
    chunked_integrated_single_layer = chunked_integrated.where(condition_layer)
    ds_integrated_single_layer = chunked_integrated_single_layer.compute()
    chunked_integrated_cloud_type = chunked_integrated_single_layer.where(ds_cloud_type[cloud_type_to_analise].astype(bool))
    ds_integrated = chunked_integrated_cloud_type.groupby('time.month').mean(dim='time', skipna=True).compute()


chunked_microphys_single_layer = chunked_microphys.where(condition_layer)
chunked_microphys_cloud_type = chunked_microphys_single_layer.where(ds_cloud_type[cloud_type_to_analise].astype(bool))

# fig_microphy = plot_microphysics_evolution(chunked_microphys_single_layer, 
#                                             chunked_integrated_single_layer, 
#                                             cloud_type='Mixed-Phase', 
#                                             var_short_name=get_var)


# ds_microphys = chunked_microphys_cloud_type.groupby('time.month').mean(dim='time', skipna=True).compute()
# ds_microphys_season = chunked_microphys_cloud_type.groupby('time.season').mean(dim='time', skipna=True).compute()
# # ds_microphys = chunked_microphys_cloud_type.resample(time='·M').mean(dim='time', skipna=True).compute()

# fig = plt.figure(figsize=(12, 6))
# gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 5], hspace=0.1, wspace=0.1)
# ax = fig.add_subplot(gs[0, 1])
# # Plot the mean profile of the microphysics variable
# # mesh = ax.pcolormesh(ds_microphys.time, ds_microphys.height, 
# #               ds_microphys.T.values, shading='nearest', 
# #               cmap='jet', norm = mpl.colors.LogNorm(vmin=0.001, vmax=0.1))
# # mesh = ax.pcolormesh(ds_microphys.month, ds_microphys.height,
# #                 ds_microphys.values.T, shading='nearest', 
# #                 cmap='coolwarm')
# mesh = ax.contourf(ds_microphys.month, ds_microphys.height,
#                 ds_microphys.values.T, levels=10, cmap=sns.color_palette("coolwarm", as_cmap=True))

# countour = ax.contour(ds_microphys.month, ds_microphys.height,
#                 ds_microphys.values.T, levels=[100], colors='black')
# ax.set_xlabel('Time')
# ax.set_title(f'Mean {ds_microphys.attrs["long_name"]} for Mixed-Phase Clouds')
# ax.yaxis.set_tick_params(labelleft=False)
# # ax.set_ylim([0, 4000])
# ax.grid(True)

# ax.clabel(countour, inline=True, fontsize=10)

# cbar = fig.colorbar(mesh, ax=ax, label=r"mg m$^{-3}$", aspect=10)
# # cbar = fig.colorbar(mesh, ax=ax, label=f"{ds_microphys.attrs['units']}", aspect=10)
# # cbar.set_ticks(np.linspace(0, 0.1, 11))

# list_seasons =list(ds_microphys_season.season.values)

# ax2 = fig.add_subplot(gs[0, 0], sharey=ax)
# max_value = np.nanmax(ds_microphys_season)
# for season in list_seasons:
#     mean_profile = ds_microphys_season.sel(season=season)
#     # std_profile = std_by_season.sel(season=season)
#     ax2.plot(mean_profile, mean_profile.height, label=f"{season}")
#     # ax2.fill_betweenx(mean_profile.height, mean_profile - std_profile, mean_profile + std_profile, alpha=0.3)
# ax2.set_xlabel(f"{ds_microphys_season.attrs['long_name']} ({ds_microphys_season.attrs['units']})")
# ax2.set_ylabel("Height (m)")
# # ax2.set_xlim([0, max_value + .1])
# # ax2.set_ylim([0, np.max(mean_by_month.height.values)])
# ax2.grid(True)
# ax2.legend()

# plt.show()

unique_months = np.unique(chunked_microphys_cloud_type['time.month'].values)
fig, ax = plt.subplots(figsize=(12, 6))
for month in unique_months:
    monthly_microphys = chunked_microphys_cloud_type.sel(time=chunked_microphys_cloud_type['time.month'] == month)
    # removing NaNs from monthly data
    monthly_microphys = monthly_microphys.dropna(dim='time', how='all').values.ravel()
    monthly_microphys = monthly_microphys[~np.isnan(monthly_microphys)]
    ax.violinplot(monthly_microphys, positions=[month], showmeans=False, 
                    showmedians=True, showextrema=False, widths=0.8)
ax.set_ylabel(f"{chunked_microphys.attrs['long_name']} ({chunked_microphys.attrs['units']})")
ax.set_xlabel("Months")
ax.set_xticks(np.arange(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.grid(True, axis='y')
plt.show()

# sliced_microphys = chunked_microphys_cloud_type.sel(time=slice('2023-01-01', '2023-12-31'))
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
    if get_var == "lwc" or get_var == "iwc":
        fig_microphy = plot_microphysics_evolution(chunked_microphys_single_layer, 
                                               chunked_integrated_single_layer, 
                                               cloud_type=cloud, 
                                               var_short_name=get_var)
    else:
        fig_microphy = plot_microphysics_evolution(chunked_microphys_single_layer, 
                                               cloud_type=cloud, 
                                               var_short_name=get_var)
    
#     list_figs.append(fig_microphy)
#     # fig_microphy.savefig(f"{PATH_FIG}grouped_by_month_evolution_for_{get_var}_{cloud}.png", dpi=300, bbox_inches='tight')

# # save all
# for i, fig in enumerate(list_figs):
#     fig.savefig(f"{PATH_FIG}grouped_by_month_evolution_for_{get_var}_{clouds_to_analyse[i]}.png", dpi=300, bbox_inches='tight')

# # axes = [fig.get_axes() for fig in list_figs]

