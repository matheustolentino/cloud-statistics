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

PATH_FIG          = '../../papers/cloud_statistics/figures/'
PATH_FIG_PICKLE   = '../../papers/cloud_statistics/figures/pickle/'
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

def plot_microphysics_evolution(sliced_microphys, 
                                sliced_integrated=None, 
                                cloud_type=None, 
                                var_short_name='lwc',
                                axes=None):
    
    if axes is None:
        fig = plt.figure(figsize=(18, 9))
        gs  = fig.add_gridspec(2, 3, width_ratios=[1., 5, .15], height_ratios=[3, 1.2], hspace=0.1, wspace=0.05)
        ax  = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])
        ax2 = fig.add_subplot(gs[:1, 0], sharey=ax)
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax)
        ax3_right = ax3.twinx()
    else:
        ax2, ax, ax3, cax, = axes
        ax3_right = ax3.twinx()

    if sliced_integrated is not None:
        number_profiles = sliced_integrated.groupby("time.month").count(dim='time')
    else:
        number_profiles = sliced_microphys.max(dim="height", skipna=True).groupby("time.month").count()
    
    # breakpoint()
    mean_by_month  = sliced_microphys.groupby('time.month').median(dim='time', skipna=True).compute()
    # std_by_month     = sliced_microphys.groupby('time.month').std(dim='time', skipna=True).compute()
    mean_by_season = sliced_microphys.groupby('time.season').median(dim='time', skipna=True).compute()
    # std_by_season    = sliced_microphys.groupby('time.season').std(dim='time', skipna=True).compute()
    
    cflevels = 10
    list_seasons =list(mean_by_season.season.values)
    if var_short_name == 'lwc':
        clevels = [1000, 1500]
        if cloud_type == 'Liquid' or cloud_type == 'Liquid-Precipitable':
            zlim    = [0, 7]
            cflevels = np.arange(0, 500, 10)
        elif cloud_type == 'Mixed-Phase' or cloud_type == 'Mixed-Phase-Precipitable':
            zlim    = [0, 13]
            cflevels = np.arange(0, 500, 10)
        elif cloud_type == 'Ice' or cloud_type == 'Ice-Precipitable':
            zlim    = [0, 13]
            cflevels = np.arange(0, 500, 10)
    elif var_short_name == 'iwc':
        clevels = [50, 200]
        if cloud_type == 'Liquid' or cloud_type == 'Liquid-Precipitable':
            zlim    = [0, 8]
            cflevels = np.arange(0, 500, 10)
        elif cloud_type == 'Mixed-Phase' or cloud_type == 'Mixed-Phase-Precipitable':
            zlim    = [0, 12]
            cflevels = np.arange(0, 150, 10)
        elif cloud_type == 'Ice' or cloud_type == 'Ice-Precipitable':
            zlim    = [0, 13]
            cflevels = np.arange(0, 150, 10)

    elif var_short_name == 'der':
        clevels = [20, 30]
        zlim = [0, 12]
    elif var_short_name == 'ier':
        clevels = [30, 70]
        zlim = [0, 13]
    
    # mesh = ax.pcolormesh(mean_by_month.month.values, mean_by_month.height.values,
    #                 mean_by_month.T, shading='nearest', cmap=sns.color_palette("icefire", as_cmap=True), 
    #                 vmin=cflevels[0], vmax=cflevels[-1])
    cf = ax.contourf(mean_by_month.month.values, mean_by_month.height.values/1e3,
                    mean_by_month.values.T, levels=cflevels, cmap=sns.color_palette("icefire", as_cmap=True))
    
    # cflevels = [2, 4, 8, 16, 20, 30, 40, 50, 60, 80, 100]
    # list_seasons =list(mean_by_season.season.values)
    # if var_short_name == 'lwc':
    #     clevels = [1000, 1500]
    #     if cloud_type == 'Liquid' or cloud_type == 'Liquid-Precipitable':
    #         zlim    = [0, 7]
    #         cflevels = [0, 5, 25, 50, 100, 200, 400, 800, 1000, 5000, 10000]
    #     elif cloud_type == 'Mixed-Phase' or cloud_type == 'Mixed-Phase-Precipitable':
    #         zlim    = [0, 13]
    #         cflevels = [0, 5, 25, 50, 100, 200, 400, 800, 1000, 5000, 10000]
    #     elif cloud_type == 'Ice' or cloud_type == 'Ice-Precipitable':
    #         zlim    = [0, 13]
    #         cflevels = [0, 5, 25, 50, 100, 200, 400, 800, 1000, 5000, 10000]
    # elif var_short_name == 'iwc':
    #     clevels = [50, 200]
    #     if cloud_type == 'Liquid' or cloud_type == 'Liquid-Precipitable':
    #         zlim    = [0, 8]
    #         cflevels = np.arange(0, 500, 10)
    #     elif cloud_type == 'Mixed-Phase' or cloud_type == 'Mixed-Phase-Precipitable':
    #         zlim    = [0, 12]
    #         cflevels = [0, 5, 25, 50, 100, 200, 400, 800, 1000, 5000, 10000]
    #     elif cloud_type == 'Ice' or cloud_type == 'Ice-Precipitable':
    #         zlim    = [0, 13]
    #         cflevels = [0, 5, 25, 50, 100, 200, 400, 800, 1000, 5000, 10000]

    # elif var_short_name == 'der':
    #     clevels = [20, 30]
    #     zlim = [0, 12]
    # elif var_short_name == 'ier':
    #     clevels = [30, 70]
    #     zlim = [0, 13]
    
    # norm = mpl.colors.Normalize(vmin=mean_by_month.values.min(), vmax=mean_by_month.values.max())
    # cmap = mpl.cm.get_cmap('jet')
    # # mesh = ax.pcolormesh(mean_by_month.month.values, mean_by_month.height.values,
    # #                 mean_by_month.T, shading='nearest', cmap=sns.color_palette("icefire", as_cmap=True), 
    # #                 vmin=cflevels[0], vmax=cflevels[-1])
    # cf = ax.contourf(mean_by_month.month.values, mean_by_month.height.values/1e3,
    #                 mean_by_month.values.T, levels=cflevels,
    #                 cmap=cmap.resampled(len(cflevels)),
    #                 extend='max')
    
    # Plot a heatmap
    # cf = ax.pcolormesh(mean_by_month.month.values, mean_by_month.height.values,
    #                 mean_by_month.values.T, shading='nearest', cmap='coolwarm')
    
    countour = ax.contour(mean_by_month.month.values, mean_by_month.height.values/1e3,
                    mean_by_month.values.T, levels=clevels, colors='white', linewidths=2)
    ax.clabel(countour, inline=True, fontsize=10)
    
    ax.set_xlabel("Time")
    ax.set_title(f"{sliced_microphys.attrs['long_name']} for {cloud_type} Clouds")
    ax.grid(True)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.xaxis.set_visible(False)
    # ax.set_xticks(np.arange(1, 13))
    ax.set_ylim(zlim)

    
    # cbar1 = plt.colorbar(mesh, cax=cax, orientation='vertical', label=f"{sliced_microphys.attrs['units']}")
    cbar1 = plt.colorbar(cf, cax=cax, orientation='vertical', label=f"{sliced_microphys.attrs['units']}")
    cbar1.ax.yaxis.set_label_position('right')

    
    max_value = np.nanmax(mean_by_season)
    for season in list_seasons:
        mean_profile = mean_by_season.sel(season=season)
        # std_profile = std_by_season.sel(season=season)
        ax2.plot(mean_profile, mean_profile.height/1e3, label=f"{season}")
        # ax2.fill_betweenx(mean_profile.height, mean_profile - std_profile, mean_profile + std_profile, alpha=0.3)
    ax2.set_xlabel(f"{sliced_microphys.attrs['long_name']} ({sliced_microphys.attrs['units']})")
    ax2.set_ylabel("Height (m)")
    # broke x-axis from 200 to 300 
    # ax2.set_xlim([0, max_value + .1])
    # ax2.set_xlim([0, max_value + .1])
    # ax2.set_ylim([0, np.max(mean_by_month.height.values)])
    ax2.grid(True)
    ax2.legend()

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
    ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
    
    # ax3.set_ylim([-.02, 0.10])
    ax3.grid(True, axis='y')
    
    
    ax3_right.plot(number_profiles.month, number_profiles.values, '--', linewidth=1, color='blue')  # Set zorder to 0
    ax3_right.set_ylabel(r"N$_{Profiles}$", color="blue")
    ax3_right.tick_params(axis='y', colors='blue')
    # ax3_right.set_yticks(np.arange(np.min(number_profiles), np.max(number_profiles), 1000))

    ax3.spines['top'].set_visible(False)  # Remove the top spine
    ax3_right.spines['top'].set_visible(False)  # Remove the top spine
    ax3_right.spines['right'].set_color('blue')  # Set the color of the right spine to blue

    # with open(f"{PATH_FIG_PICKLE}grouped_by_month_evolution_for_{get_var}_{cloud_type}_mesh.pkl", "wb") as f:
    #     pickle.dump(fig, f)

    # # save all data as netCDF
    # if var_short_name == 'lwc' or var_short_name == 'iwc':
    #     mean_by_season.to_netcdf(f"{PATH_SAVE_DATA}mean_by_season_{get_var}_{cloud_type}.nc")
    #     mean_by_month.to_netcdf(f"{PATH_SAVE_DATA}mean_by_month_{get_var}_{cloud_type}.nc")

def plot_microphysics_evolution2(sliced_microphys, 
                                sliced_integrated=None, 
                                cloud_type=None, 
                                var_short_name='lwc',
                                axes=None,
                                fig=None):
    
    if axes is None:
        fig = plt.figure(figsize=(18, 9))
        gs  = fig.add_gridspec(2, 3, width_ratios=[1., 5, .15], height_ratios=[3, 1.2], hspace=0.1, wspace=0.05)
        ax  = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])
        ax2 = fig.add_subplot(gs[:1, 0], sharey=ax)
        ax2b = fig.add_subplot(gs[:1, 1], frameon=False, sharey=ax2)
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax)
        ax3_right = ax3.twinx()
    else:
        ax2, ax, ax3, cax = axes
        ax3_right = ax3.twinx()

    if sliced_integrated is not None:
        number_profiles = sliced_integrated.groupby("time.month").count(dim='time')
    else:
        number_profiles = sliced_microphys.max(dim="height", skipna=True).groupby("time.month").count()
    
    mean_by_month  = sliced_microphys.groupby('time.month').median(dim='time', skipna=True).compute()
    mean_by_season = sliced_microphys.groupby('time.season').median(dim='time', skipna=True).compute()
    
    cflevels = 10
    list_seasons = list(mean_by_season.season.values)

    if var_short_name == 'lwc':
        clevels = [1000]
        if cloud_type == 'Liquid' or cloud_type == 'Liquid-Precipitable':
            zlim = [0, 7]
            cflevels = np.arange(0, 500, 10)
            # cflevels = np.array([50, 100, 400, 800, 1000, 5000, 10000, 30000])
        elif cloud_type == 'Mixed-Phase' or cloud_type == 'Mixed-Phase-Precipitable':
            zlim = [0, 13]
            cflevels = np.arange(0, 500, 10)
        elif cloud_type == 'Ice' or cloud_type == 'Ice-Precipitable':
            zlim = [0, 13]
            cflevels = np.arange(0, 500, 10)
    elif var_short_name == 'iwc':
        clevels = [50, 200]
        if cloud_type == 'Liquid' or cloud_type == 'Liquid-Precipitable':
            zlim = [0, 8]
            cflevels = np.arange(0, 500, 10)
        elif cloud_type == 'Mixed-Phase' or cloud_type == 'Mixed-Phase-Precipitable':
            zlim = [0, 12]
            cflevels = np.arange(0, 150, 10)
        elif cloud_type == 'Ice' or cloud_type == 'Ice-Precipitable':
            zlim = [0, 13]
            cflevels = np.arange(0, 150, 10)
    elif var_short_name == 'der':
        clevels = [20, 30]
        zlim = [0, 12]
    elif var_short_name == 'ier':
        clevels = [30, 70]
        zlim = [0, 13]
    
    # norm = mpl.colors.Normalize(vmin=cflevels[0], vmax=cflevels[-1], clip=False)
    cf = ax.contourf(mean_by_month.month.values, mean_by_month.height.values / 1e3,
                     mean_by_month.values.T, levels=cflevels, cmap=sns.color_palette("icefire", as_cmap=True),
                    )
    # make a pcolormesh 
    # cf = ax.pcolormesh(mean_by_month.month.values, mean_by_month.height.values / 1e3,
    #                      mean_by_month.values.T, shading='nearest', cmap=sns.color_palette("jet", as_cmap=True))

    countour = ax.contour(mean_by_month.month.values, mean_by_month.height.values / 1e3,
                          mean_by_month.values.T, levels=clevels, colors='white', linewidths=2)
    ax.clabel(countour, inline=True, fontsize=10)
    
    ax.set_xlabel("Time")
    ax.set_title(f"{sliced_microphys.attrs['long_name']} for {cloud_type} Clouds")
    ax.grid(True)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.xaxis.set_visible(False)
    ax.set_ylim(zlim)
    
    cbar1 = plt.colorbar(cf, cax=cax, orientation='vertical', label=f"{sliced_microphys.attrs['units']}")
    cbar1.ax.yaxis.set_label_position('right')

    max_value = np.nanmax(mean_by_season)
    broken_axis = False
    # define broken limites:
    if cloud_type == 'Liquid' and var_short_name == 'lwc':
        # xi_broken, xf_broken = 300, 600
        # dx_tick = 150
        xi_broken, xf_broken = 600, 1000
        dx_tick = 250
        # xtick_array = np.array(0, 300, 700)
        broken_axis = True
    # elif cloud_type == 'Mixed-Phase' and var_short_name == 'lwc':
    #     xi_broken, xf_broken = 600, 1400
    #     dx_tick = 200
    #     broken_axis = True
    # elif cloud_type == 'Mixed-Phase-Precipitable' and var_short_name == 'lwc':
    #     xi_broken, xf_broken = 500, 1300
    #     broken_axis = True
    #     dx_tick = 400
    elif cloud_type == 'Mixed-Phase-Precipitable' and var_short_name == 'iwc':
        xi_broken, xf_broken = 150, 200
        broken_axis = True
        dx_tick = 50
    elif cloud_type == 'Ice' and var_short_name == 'iwc':
        xi_broken, xf_broken = 60, 450
        dx_tick = 50
        broken_axis = True

    if broken_axis:
        # Define ax2b:
        gs2 = ax2.get_subplotspec().subgridspec(1, 2, width_ratios=[1.8, 1], wspace=0.2)
        ax2a = fig.add_subplot(gs2[0, 0], sharey=ax2)
        ax2b = fig.add_subplot(gs2[0, 1], sharey=ax2a)

        # Remove all values and borders of ax2
        ax2.set_xticks([])
        # ax2.tick_params(labelright=False)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        # Plot data on ax2a and ax2b
        for season in list_seasons:
            mean_profile = mean_by_season.sel(season=season)
            ax2a.plot(mean_profile, mean_profile.height / 1e3, label=f"{season}")
            ax2b.plot(mean_profile, mean_profile.height / 1e3)
        
        ax2a.set_xlim(0, xi_broken)
        ax2b.set_xlim(xf_broken, xf_broken+50)
        ax2a.spines['right'].set_visible(False)
        ax2b.spines['left'].set_visible(False)
        ax2a.tick_params(labelright='off')
        ax2b.yaxis.tick_right()
        # ax2b.set_yticks([])
        # ax2b.set_yticklabels([])
        ax2a.set_xticks(np.arange(0, xi_broken, dx_tick))
        ax2b.set_xticks(np.arange(xf_broken+dx_tick, xf_broken+dx_tick+1, 1))
        ax2b.tick_params(labelright=False)
        ax2a.tick_params(labelright=False)
        ax2a.set_ylabel("Height (km) a.g.l.")
        # ax2a.yaxis.set_tick_params(labelleft=True)  # Ensure labels on the left are visible

        # Add break lines
        d = .02  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax2a.transAxes, color='k', clip_on=False)
        ax2a.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax2a.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        
        kwargs.update(transform=ax2b.transAxes)
        ax2b.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2b.plot((-d, +d), (-d, +d), **kwargs)
        
        ax2a.legend(loc='upper center', bbox_to_anchor=(1.1, 1.25), ncol=2, fontsize='small')
        ax2a.set_xlabel(f"{sliced_microphys.attrs['long_name']} ({sliced_microphys.attrs['units']})")
    else: 
        for season in list_seasons:
            mean_profile = mean_by_season.sel(season=season)
            ax2.plot(mean_profile, mean_profile.height/1e3, label=f"{season}")
    
        ax2.set_xlabel(f"{sliced_microphys.attrs['long_name']} ({sliced_microphys.attrs['units']})")
        ax2.set_ylabel("Height (km) a.g.l.")

        ax2.grid(True)
        ax2.legend(loc='upper center', bbox_to_anchor=(.5, 1.25), ncol=2, fontsize='small')

    unique_months = np.unique(mean_by_month.month.values)
    for month in unique_months:
        if var_short_name in ['lwc', 'iwc']:
            monthly_integrated = sliced_integrated.sel(time=sliced_integrated['time.month'] == month)
            monthly_integrated = monthly_integrated.dropna(dim='time', how='all').values
            ax3.boxplot(monthly_integrated, positions=[month], showfliers=False, showmeans=True, patch_artist=True, widths=0.8,
                        meanprops=dict(marker='*', markerfacecolor='black', markeredgecolor='black'),
                        medianprops=dict(color='red', linewidth=1.5), boxprops=dict(facecolor='lightblue', color='black'))
        else:
            monthly_microphys = sliced_microphys.sel(time=sliced_microphys['time.month'] == month)
            monthly_microphys = monthly_microphys.dropna(dim='time', how='all').values.ravel()
            monthly_microphys = monthly_microphys[~np.isnan(monthly_microphys)]
            ax3.violinplot(monthly_microphys, positions=[month], showmeans=False, 
                           showmedians=True, showextrema=False, widths=0.8)
    
    if var_short_name == 'lwc' or var_short_name == 'iwc':
        ax3.set_ylabel(f"{sliced_integrated.attrs['long_name']} ({sliced_integrated.attrs['units']})")
    else:
        ax3.set_ylabel(f"{sliced_microphys.attrs['long_name']} ({sliced_microphys.attrs['units']})")
    ax3.set_xlabel("Months")
    ax3.set_xticks(np.arange(1, 13))
    ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
    
    # ax3.set_ylim([-.02, 0.10])
    ax3.grid(True, axis='y')
    
    ax3_right.plot(number_profiles.month, number_profiles.values, '--', linewidth=1, color='blue')  # Set zorder to 0
    ax3_right.set_ylabel(r"N$_{Profiles}$", color="blue")
    ax3_right.tick_params(axis='y', colors='blue')
    # ax3_right.set_yticks(np.arange(np.min(number_profiles), np.max(number_profiles), 1000))

    ax3.spines['top'].set_visible(False)  # Remove the top spine
    ax3_right.spines['top'].set_visible(False)  # Remove the top spine
    ax3_right.spines['right'].set_color('blue')  # Set the color of the right spine to blue

    # with open(f"{PATH_FIG_PICKLE}grouped_by_month_evolution_for_{get_var}_{cloud_type}_mesh.pkl", "wb") as f:
    #     pickle.dump(fig, f)

    # # save all data as netCDF
    # if var_short_name == 'lwc' or var_short_name == 'iwc':
    #     mean_by_season.to_netcdf(f"{PATH_SAVE_DATA}mean_by_season_{get_var}_{cloud_type}.nc")
    #     mean_by_month.to_netcdf(f"{PATH_SAVE_DATA}mean_by_month_{get_var}_{cloud_type}.nc")

def get_vars(cloud):
    if cloud == 'Liquid' or cloud == 'Liquid-Precipitable':
        vars = ['lwc', 'der']
    elif cloud == 'Mixed-Phase' or cloud == 'Mixed-Phase-Precipitable':
        vars = ['lwc', 'der', 'iwc', 'ier']
    elif cloud == 'Ice' or cloud == 'Ice-Precipitable':
        vars = ['iwc', 'ier']
    else:
        raise ValueError("Cloud type not recognized")
    return vars

def get_var_endswith(var_short_name):
    if var_short_name == "lwc":
        endswith = "lwc-scaled-adiabatic.nc"
    elif var_short_name == "iwc":
        endswith = "iwc-Z-T-method.nc"
    elif var_short_name == "der":
        endswith = "der.nc"
    elif var_short_name == "ier":
        endswith = "ier.nc"
    else:
        raise ValueError("Variable not recognized")
    return endswith

def get_fig_axes_variable(variable):
    if variable == 'lwc':
        fig_axes = lwc_axes
    elif variable == 'iwc':
        fig_axes = iwc_axes
    elif variable == 'der':
        fig_axes = der_axes
    elif variable == 'ier':
        fig_axes = ier_axes
    return fig_axes

def get_fig_axes_cloud(variable, cloud):
    if cloud == 'Liquid':
        if variable == 'lwc':
            fig_axes = lwc_or_iwc_non_prep_axes
        elif variable == 'der':
            fig_axes = der_or_ier_non_prep_axes
    elif cloud == 'Liquid-Precipitable':
        if variable == 'lwc':
            fig_axes = lwc_or_iwc_prep_axes
        elif variable == 'der':
            fig_axes = der_or_ier_prep_axes
    elif cloud == 'Ice':
        if variable == 'iwc':
            fig_axes = lwc_or_iwc_non_prep_axes
        elif variable == 'ier':
            fig_axes = der_or_ier_non_prep_axes
    elif cloud == 'Ice-Precipitable':
        if variable == 'iwc':
            fig_axes = lwc_or_iwc_prep_axes
        elif variable == 'ier':
            fig_axes = der_or_ier_prep_axes
    else:
        raise ValueError("Cloud type not recognized")
    return fig_axes

def read_cloud_files_2D(filepath_cloud_occurence, filepath_cloud_cloud_type):
    
    # Get the list of netCDF files in the specified directory
    file_paths_occurence = [os.path.join(filepath_cloud_occurence, file) for file in os.listdir(filepath_cloud_occurence) if file.endswith('.nc')]
    file_paths_cloud_type = [os.path.join(filepath_cloud_cloud_type, file) for file in os.listdir(filepath_cloud_cloud_type) if file.endswith('.nc')]

    # Read the netCDF files into a list of xarray datasets
    datasets_occurence = [xr.open_dataset(file_path, engine='netcdf4', chunks={'time': -1}) for file_path in file_paths_occurence]
    datasets_cloud_type= [xr.open_dataset(file_path, engine='netcdf4', chunks={'time': -1}) for file_path in file_paths_cloud_type]
    
    print("Concatenating cloud occurene and cloud type datasets")
    # Concatenate the datasets along the time dimension
    chunked_occurence  = xr.concat(datasets_occurence, dim='time').sortby('time')
    chunked_cloud_type = xr.concat(datasets_cloud_type, dim='time').sortby('time')

    chunked_occurence.chunk({'time': 'auto'})
    chunked_cloud_type.chunk({'time': 'auto'})

    return chunked_occurence.compute(), chunked_cloud_type.compute()

def read_microphysics(filepath_microphys, get_var, endswith, cloud_macrophysics=None):
    
    # Get the list of netCDF files in the specified directory
    filepaths_microphys = [os.path.join(filepath_microphys, file) for file in os.listdir(filepath_microphys) if file.endswith(endswith)] 

    # Set the height resolution for the interpolated datasets
    dz = 20
    new_height = np.arange(0, 14000+dz, dz)
    
    print("Reading microphysics files")
    # Read data in files between cloud_base and cloud_top heights inside cloud_macrophysics
    datasets_microphys = []
    for file_path in filepaths_microphys:
        ds_0 = xr.open_dataset(file_path, engine='netcdf4', chunks={'time': -1})[get_var].assign_coords(height=lambda ds: ds.height - GRANADA_ALTITUDE)
        if cloud_macrophysics is not None:
            cloud_base = cloud_macrophysics.sel(time=ds_0.time, method='nearest', tolerance=np.timedelta64(30, 's'))['cloud_base']
            cloud_top  = cloud_macrophysics.sel(time=ds_0.time, method='nearest', tolerance=np.timedelta64(30, 's'))['cloud_top'] 
            ds_1 = ds_0.where((ds_0.height >= cloud_base) & (ds_0.height <= cloud_top))
        
        datasets_microphys.append(ds_1)
        # # check if there is any differences in ds_1 and ds_0:
        # if cloud_macrophysics is not None:
        #     if not ds_0.equals(ds_1):
        #         print("There are differences between ds_0 and ds_1")

        # # make a pcolormash of ds and scatter plot of cloud_base and cloud_top
        # fig, ax = plt.subplots()
        # mesh = ax.pcolormesh(ds_1.time, ds_1.height, ds_1.T*1e6, shading='nearest', cmap='jet')
        # ax.scatter(ds_1.time, cloud_base, color='black', marker='x', label='cloud_base')
        # ax.scatter(ds_1.time, cloud_top, color='black', marker='x', label='cloud_top')
        # ax.set_xlabel("Time")
        # ax.set_ylabel("Height")
        # ax.set_title(f"{get_var} for {os.path.basename(file_path)}")
        # ax.legend()
        # plt.colorbar(mesh, ax=ax, label=f"{ds_1.attrs['units']}")
        # plt.show()

        # fig, ax = plt.subplots()
        # mesh = ax.pcolormesh(ds_0.time, ds_0.height, ds_0.T*1e6, shading='nearest', cmap='jet')
        # ax.scatter(ds_0.time, cloud_base, color='black', marker='x', label='cloud_base')
        # ax.scatter(ds_0.time, cloud_top, color='black', marker='x', label='cloud_top')
        # ax.set_xlabel("Time")
        # ax.set_ylabel("Height")
        # ax.set_title(f"{get_var} for {os.path.basename(file_path)}")
        # ax.legend()
        # plt.colorbar(mesh, ax=ax, label=f"{ds_0.attrs['units']}")
        # plt.show()

        # set_trace()

    # datasets_microphys = [xr.open_dataset(file_path, engine='netcdf4', chunks={'time': -1})[get_var].assign_coords(height=lambda ds: ds.height - GRANADA_ALTITUDE) for file_path in filepaths_microphys]

    print("Interpolating microphysics datasets")
    # Set the same height resolution for all datasets
    datasets_microphys = [ds.interp(height=new_height) for ds in datasets_microphys] # linear interpolation
    
    print("Concatenating microphysics datasets")
    # Concatenate the interpolated datasets along the time dimension
    chunked_microphys = xr.concat(datasets_microphys, dim='time').sortby('time')
    chunked_microphys = chunked_microphys.where(chunked_microphys > 0)
    chunked_microphys.chunk({'time': 'auto'})

    if get_var == "lwc":
        chunked_microphys = chunked_microphys*1e6
        chunked_microphys.attrs['units'] = 'mg m$^{-3}$'
        chunked_microphys.attrs['long_name'] = 'LWC'
    elif get_var == "iwc":
        chunked_microphys = chunked_microphys*1e6
        chunked_microphys.attrs['units'] = 'mg m$^{-3}$'
        chunked_microphys.attrs['long_name'] = 'IWC'
    elif get_var == "der":
        chunked_microphys = chunked_microphys*1e6
        chunked_microphys.attrs['units'] = '$\mu m$'
        chunked_microphys.attrs['long_name'] = '$r_{liq}$'
    elif get_var == "ier":  
        chunked_microphys = chunked_microphys*1e6
        chunked_microphys.attrs['units'] = '$\mu m$'
        chunked_microphys.attrs['long_name'] = '$r_{ice}$'
    print("End of concatenation")
    
    if get_var == "der" or get_var == "ier":
        return chunked_microphys, None
    else:
        print("Integrating microphysics in the column")
        datasets_integrated = [ds.fillna(0.).integrate('height') for ds in datasets_microphys]
        # min_dataset       = [ds.min(dim='height', skipna=True) for ds in datasets_microphys]
        chunked_integrated  = xr.concat(datasets_integrated, dim='time').sortby('time')
        chunked_integrated.chunk({'time': 'auto'})
        chunked_integrated  = chunked_integrated*1e3 # Convert to g m^-2
        
        if get_var == "lwc":
            chunked_integrated.attrs['units'] = 'g m$^{-2}$'
            chunked_integrated.attrs['long_name'] = 'LWP'
        else: 
            chunked_integrated.attrs['units'] = 'g m$^{-2}$'
            chunked_integrated.attrs['long_name'] = 'IWP'
        return chunked_microphys, chunked_integrated

# --------------------------------------------------------------
# Main code
# --------------------------------------------------------------
filepath_cloud_occurence  = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification/cloud_occurence" # Path to save the cloud classification files
filepath_cloud_cloud_type = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification/cloud_type" # Path to save the cloud classification files
filepath_microphys        = "/home/matheustolen/Documentos/matheus_doctorado/output_retrievals" # Path to save the cloud classification files
filepath_cloud_prop       = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification/cloud_properties" # Path to save the cloud classification files

ds_cloud_occurence, ds_cloud_type = read_cloud_files_2D(filepath_cloud_occurence, 
                                                        filepath_cloud_cloud_type)
file_paths_cloud_prop     = [os.path.join(filepath_cloud_prop, file) for file in os.listdir(filepath_cloud_prop) if file.endswith('.nc')]
mixed_phase_cloud         = False
clouds = ['Mixed-Phase-Precipitable']

clouds = ['Ice', 'Ice-Precipitable']
letters = iter('abcdefghijklmnopqrstuvwxyz')
print("read cloud macrophysics")
# read cloud prop but convert altitud to height using granada altitude
datasets_cloud_prop = [xr.open_dataset(file_path)-GRANADA_ALTITUDE for file_path in file_paths_cloud_prop]
ds_cloud_prop       = xr.concat(datasets_cloud_prop, dim='time').sortby('time')

if mixed_phase_cloud:
    for cloud in clouds:
        print(f"Ploting {cloud} clouds")
        variables = get_vars(cloud)
        condition = (ds_cloud_occurence['single_layer'] == 1) &\
            (ds_cloud_occurence['noise'] == 0) &\
                (ds_cloud_type[cloud].astype(bool))
        
        if cloud == 'Mixed-Phase' or cloud == 'Mixed-Phase-Precipitable':
            fig = plt.figure(figsize=(24, 15))
            gs  = fig.add_gridspec(5, 7, 
                                hspace=0.2,
                                wspace=0.08, 
                                width_ratios=[.4, 1.2, .05, .3, .4, 1.2, .05],
                                height_ratios=[1, .3, .2, 1, 0.3])
            
            # Use these axes for lwc
            ax1  = fig.add_subplot(gs[0,0]) # vertical profiles
            ax2  = fig.add_subplot(gs[0,1], sharey=ax1) # contour plot
            ax3  = fig.add_subplot(gs[1,1], sharex=ax2) # monthly boxplots
            cax2 = fig.add_subplot(gs[0,2]) # colorbar
            lwc_axes = [ax1, ax2, ax3, cax2]
            
            # Use these axes for iwc
            ax4  = fig.add_subplot(gs[0,4], sharey=ax1) # vertical profiles
            ax5  = fig.add_subplot(gs[0,5], sharey=ax1) # contour plot
            ax6  = fig.add_subplot(gs[1,5], sharex=ax5) # monthly boxplots
            cax5 = fig.add_subplot(gs[0,6]) # colorbar
            iwc_axes = [ax4, ax5, ax6, cax5]

            # Use these axes for der
            ax7  = fig.add_subplot(gs[3,0]) # vertical profiles
            ax8  = fig.add_subplot(gs[3,1], sharey=ax7) # contour plot
            ax9  = fig.add_subplot(gs[4,1], sharex=ax8) # monthly boxplots
            cax8 = fig.add_subplot(gs[3,2]) # colorbar
            der_axes = [ax7, ax8, ax9, cax8]    

            # Use these axes for ier
            ax10  = fig.add_subplot(gs[3,4], sharey=ax7) # vertical profiles
            ax11  = fig.add_subplot(gs[3,5], sharey=ax7) # contour plot
            ax12  = fig.add_subplot(gs[4,5], sharex=ax11) # monthly boxplots
            cax11 = fig.add_subplot(gs[3,6]) # colorbar
            ier_axes = [ax10, ax11, ax12, cax11]

        for i, variable in enumerate(variables):
            var_endswith = get_var_endswith(variable)
            print(f"Reading {variable} variable for {cloud} clouds...")
            chunked_microphys, chunked_integrated = read_microphysics(filepath_microphys, 
                                                                    variable, 
                                                                    var_endswith,
                                                                    ds_cloud_prop)
            print("Finished reading microphysics files...")
            sliced_microphys = chunked_microphys.sel(time=condition)
            
            print(f"Plotting {variable} for {cloud} clouds")
            fig_axes = get_fig_axes_variable(variable)
            fig_axes[0].text(-0.1, 1.1, f"{next(letters)})", transform=fig_axes[0].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
            if chunked_integrated is not None:
                sliced_integrated_microphys = chunked_integrated.sel(time=condition)
                plot_microphysics_evolution2(sliced_microphys, 
                                                            sliced_integrated_microphys, 
                                                            cloud_type=cloud, 
                                                            var_short_name=variable,
                                                            axes=fig_axes,
                                                            fig=fig)
                    
            else:
                plot_microphysics_evolution2(sliced_microphys, 
                                                            cloud_type=cloud, 
                                                            var_short_name=variable,
                                                            axes=fig_axes,
                                                            fig=fig)
                
        fig.savefig(f"{PATH_FIG}monthly_evolution_for_{cloud}_clouds_test.png", dpi=400, bbox_inches='tight')
        plt.show()
else:
    for cloud in clouds:
        print(f"Ploting {cloud} clouds")
        variables = get_vars(cloud)
        condition = (ds_cloud_occurence['single_layer'] == 1) &\
            (ds_cloud_occurence['noise'] == 0) &\
                (ds_cloud_type[cloud].astype(bool))
        
        if cloud == 'Liquid' or cloud == 'Ice':
            fig = plt.figure(figsize=(24, 15))
            gs  = fig.add_gridspec(5, 7, 
                                hspace=0.2,
                                wspace=0.08, 
                                width_ratios=[.4, 1.2, .05, .3, .4, 1.2, .05],
                                height_ratios=[1, .3, .2, 1, 0.3])
            
            # If non-precipitating cloud
            # Use these axes for lwc or iwc
            ax1  = fig.add_subplot(gs[0,0]) # vertical profiles
            ax2  = fig.add_subplot(gs[0,1], sharey=ax1) # contour plot
            ax3  = fig.add_subplot(gs[1,1], sharex=ax2) # monthly boxplots
            cax2 = fig.add_subplot(gs[0,2]) # colorbar
            lwc_or_iwc_non_prep_axes = [ax1, ax2, ax3, cax2]
            
            # If precipitating cloud
            # Use these axes for lwc or iwc
            ax4  = fig.add_subplot(gs[0,4], sharey=ax1) # vertical profiles
            ax5  = fig.add_subplot(gs[0,5], sharey=ax1) # contour plot
            ax6  = fig.add_subplot(gs[1,5], sharex=ax5) # monthly boxplots
            cax5 = fig.add_subplot(gs[0,6]) # colorbar
            lwc_or_iwc_prep_axes = [ax4, ax5, ax6, cax5]
            
            # If non-precipitating cloud
            # Use these axes for der or ier
            ax7  = fig.add_subplot(gs[3,0]) # vertical profiles
            ax8  = fig.add_subplot(gs[3,1], sharey=ax7) # contour plot
            ax9  = fig.add_subplot(gs[4,1], sharex=ax8) # monthly boxplots
            cax8 = fig.add_subplot(gs[3,2]) # colorbar
            der_or_ier_non_prep_axes = [ax7, ax8, ax9, cax8]    
            
            # If precipitating cloud
            # Use these axes for der or ier
            ax10  = fig.add_subplot(gs[3,4], sharey=ax7) # vertical profiles
            ax11  = fig.add_subplot(gs[3,5], sharey=ax7) # contour plot
            ax12  = fig.add_subplot(gs[4,5], sharex=ax11) # monthly boxplots
            cax11 = fig.add_subplot(gs[3,6]) # colorbar
            der_or_ier_prep_axes = [ax10, ax11, ax12, cax11]

        for i, variable in enumerate(variables):
            var_endswith = get_var_endswith(variable)
            print(f"Reading {variable} variable for {cloud} clouds...")
            chunked_microphys, chunked_integrated = read_microphysics(filepath_microphys, 
                                                                    variable, 
                                                                    var_endswith,
                                                                    ds_cloud_prop)
            print("Finished reading microphysics files...")
            sliced_microphys = chunked_microphys.sel(time=condition)
            
            print(f"Plotting {variable} for {cloud} clouds")
            fig_axes = get_fig_axes_cloud(variable, cloud)
            fig_axes[0].text(-0.1, 1.1, f"{next(letters)})", transform=fig_axes[0].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
            # set_trace()
            if chunked_integrated is not None:
                sliced_integrated_microphys = chunked_integrated.sel(time=condition)
                plot_microphysics_evolution2(sliced_microphys, 
                                                            sliced_integrated_microphys, 
                                                            cloud_type=cloud, 
                                                            var_short_name=variable,
                                                            axes=fig_axes,
                                                            fig=fig)
                    
            else:
                plot_microphysics_evolution2(sliced_microphys, 
                                                            cloud_type=cloud, 
                                                            var_short_name=variable,
                                                            axes=fig_axes,
                                                            fig=fig)
        if cloud == 'Liquid-Precipitable' or cloud == 'Ice-Precipitable':    
            fig.savefig(f"{PATH_FIG}monthly_evolution_for_{cloud}_clouds_test.png", dpi=400, bbox_inches='tight')
            plt.show()