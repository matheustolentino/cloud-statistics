import glob
import struct
import numpy as np
from pdb import set_trace
import datetime
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import time as time_module
import seaborn as sns
import matplotlib.dates as mdates
import dask
import re
import matplotlib.font_manager as fm
import os
from phd_clouds.mwr_class import Mwr
from phd_clouds.utils import assign_season
from phd_clouds.constants import SEASONS
import matplotlib as mpl
from IPython import get_ipython


# Define the path to the figures directory
PATH_FIG = '../figures/'

# Define the fontsize
fontsize = 14

# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize

plt.ion()
plt.close('all')

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('matplotlib', 'inline')

# Test the Mwr class for one day of data
path_daily_mwr = '/home/matheustolen/shared/RAW/UGR/mwr/Y2023/M10/D01'
mwr_files  = glob.glob(path_daily_mwr + '/*.LWP', recursive=True)

mwr_object = Mwr()

mwr      = mwr_object.concatenate_mwr_files_lwp(mwr_files)
resampled_mwd = mwr.resample(time='30S').mean()

# Plot LWP in function of time and its 30s average
fig, ax = plt.subplots(figsize=(10, 5))
resampled_mwd['lwp'].plot(ax=ax, label='LWP')
ax.set_xlabel('Time')
ax.set_ylabel('LWP (g/m$^{2}$)')
ax.legend()
plt.show()

mwr_files  = glob.glob(path_daily_mwr + '/*.TPC', recursive=True)

mwr = mwr_object.concatenate_mwr_files_tpc(mwr_files)

fig, ax = plt.subplots(figsize=(10, 5))
mwr.temperature.T.plot(ax=ax, cmap='jet')
ax.set_xlabel('Time (UTC)')
ax.set_ylabel('Altitude (m)')
plt.show()

mwr_files  = glob.glob(path_daily_mwr + '/*.HPC', recursive=True)

rh, q = mwr_object.concatenate_mwr_files_hpc(mwr_files)

fig, ax = plt.subplots(figsize=(10, 5))
rh.relative_humidity.T.plot(ax=ax, cmap='jet')
ax.set_xlabel('Time (UTC)')
ax.set_ylabel('Altitude (m)')
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
q.humidity.T.plot(ax=ax, cmap='jet')
ax.set_xlabel('Time (UTC)')
ax.set_ylabel('Altitude (m)')
plt.show()

set_trace()
# -----------------------------------------------------------------------------------------------
# Input directory containing the MWR data
#------------------------------------------------------------------------------------------
input_directory = '/home/matheustolen/shared/NAS_raw_data/UGR/mwr/'
# -----------------------------------------------------------------------------------------------
quicklook_temperature = True
quicklook_humidity = False
save_data = False
# -----------------------------------------------------------------------------------------------
# initialize the Mwr class
# -----------------------------------------------------------------------------------------------
mwr_data = Mwr()
# -----------------------------------------------------------------------------------------------
# Use glob to find files matching the pattern '*.CMP.TPC' in all subdirectories of input_directory
mwr_files_tpc = glob.glob(input_directory + '**/*.CMP.TPC', recursive=True)

# Use glob to find files matching the pattern '*.CMP.TPC' in all subdirectories of input_directory
mwr_files_hpc = glob.glob(input_directory + '**/*.HPC', recursive=True)

# # Filter mwr_files_hpc to get only files with the specified structure
# pattern = r'ZENITH_\d{2}\d{2}\d{2}.HPC'
# filtered_files_hpc = [file for file in mwr_files_hpc if re.match(pattern, os.path.basename(file))]
# print(mwr_files_hpc)
# print('Number of files found: ', len(mwr_files_hpc))
# print('Number of filtered files found: ', len(filtered_files_hpc))
# set_trace()
# -----------------------------------------------------------------------------------------------
if quicklook_temperature:
    # data, altitude_range, header, tpcRetrieval = mwr_data.loadTPC(mwr_files_tpc[0])
    # time = data[:, 0]
    # Reading the first file in the input directory and concatenating it into a single dataset
    # mwr_temp = mwr_data.concatenate_mwr_files_tpc([mwr_files_tpc[0]])

    start_time = time_module.time()
    print("Reading all temperature files in the input directory and concatenating it into a single dataset")
    mwr_temp = mwr_data.concatenate_mwr_files_tpc_2(mwr_files_tpc)
    print("Done!")
    end_time = time_module.time()
    print("Time taken in minutes: ", (end_time - start_time)/60)

    # # Resampling the dataset to monthly frequency
    # resampled_mwr = mwr_temp_concatenated.resample(time='1M').mean()

    # # Reindexed dataset for montly frequency with nan in months with no data
    # start_time = resampled_mwr['time'].min().values
    # end_time = resampled_mwr['time'].max().values
    # new_time_index = pd.date_range(start=start_time, end=end_time, freq='M',normalize=True)

    # reindexed_mwr = resampled_mwr.reindex(time=new_time_index)

    # # Plotting the Temperature variable using pcolormesh
    # fig, ax = plt.subplots()
    # im = ax.pcolormesh(mwr_temp['time'], mwr_temp['altitude'], mwr_temp['temperature'].T, shading='auto', cmap='jet')
    # fig.colorbar(im, ax=ax, label='Temperature (K)')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Altitude')
    # ax.set_title('Temperature Variation')
    # plt.show()

#--------------------------------------------------------------------------------------------------------------
if quicklook_humidity:
    # Reading the first file in the input directory and concatenating it into a single dataset
    # mwr_humidity = mwr_data.concatenate_mwr_files_hpc([mwr_files_hpc[0]])

    # Reading all humidity the files in the input directory and concatenating them into a single dataset
    print("Reading all humidity files in the input directory and concatenating them into a single dataset")
    start_time = time_module.time()
    # mwr_humidity_concatenated = mwr_data.concatenate_mwr_files_hpc_2(mwr_files_hpc)
    mwr_rh, mwr_h = mwr_data.concatenate_mwr_files_hpc_2(mwr_files_hpc)
    print("Done!")
    end_time = time_module.time()
    print("Time taken in minutes: ", (end_time - start_time)/60)

    # # Plotting the Humidity and Relative Humidity variables using pcolormesh
    # fig, ax = plt.subplots()
    # im = ax.pcolormesh(mwr_humidity['time'], mwr_humidity['altitude'], mwr_humidity['humidity'].T, shading='auto', cmap='jet')
    # fig.colorbar(im, ax=ax, label='Humidity (g/m^3)')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Altitude')
    # ax.set_title('Humidity Variation')
    # plt.show()

    # fig, ax = plt.subplots()
    # im = ax.pcolormesh(mwr_humidity['time'], mwr_humidity['altitude'], mwr_humidity['relative_humidity'].T, shading='auto', cmap='jet')
    # fig.colorbar(im, ax=ax, label='Relative Humidity (%)')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Altitude')
    # ax.set_title('Relative Humidity Variation')
    # plt.show()

    # # Plotting the Humidity and Relative Humidity variables using pcolormesh
    # fig, ax = plt.subplots()
    # im = ax.pcolormesh(mwr_humidity_concatenated['time'], mwr_humidity_concatenated['altitude'], mwr_humidity_concatenated['humidity'].T, shading='auto', cmap='jet')
    # fig.colorbar(im, ax=ax, label='Humidity (g/m^3)')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Altitude')
    # ax.set_title('Humidity Variation')
    # plt.show()

    # fig, ax = plt.subplots()
    # im = ax.pcolormesh(mwr_humidity_concatenated['time'], mwr_humidity_concatenated['altitude'], mwr_humidity_concatenated['relative_humidity'].T, shading='auto', cmap='jet')
    # fig.colorbar(im, ax=ax, label='Relative Humidity (%)')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Altitude (m)')
    # ax.set_title('Relative Humidity Variation')
    # plt.show()

# Plotting the Humidity and Relative Humidity variables using pcolormesh
# df = mwr_rh.to_dataframe()
# # Check for duplicated time values
# duplicated_times = df.index[df.index.duplicated()]


mwr_h = mwr_h.drop_duplicates('time', keep='first')
mwr_rh = mwr_rh.drop_duplicates('time', keep='first')
mwr_temp = mwr_temp.drop_duplicates('time', keep='first')
# # Pressure profile from standar atmosphere

# hb = 0
# dh = 7.5
# tb = mwr_temp['temperature'][0, 0] # Temperature at ground [K]
# pb = 940 # Pressure at ground [hPa]

# h_std_at, t_std_at, p_std_at = mwr_data.standardAtmosphere(hb, dh, tb, pb)
# temp_potential = mwr_temp['temperature'] * (1000/ p_std_at) ** (287.058 / 1004.5)

mwr_profiles = mwr_data.merge_mwr_ds([mwr_rh, mwr_h, mwr_temp], dim='time')
mwr_profiles = assign_season(mwr_profiles, SEASONS)
# calculating temperature surface for stations
mwr_profiles['temperature_surface'] = mwr_profiles['temperature'][:, 0]
mwr_profiles['relative_humidity_surface'] = mwr_profiles['relative_humidity'][:, 0]
mean_temp_surface = mwr_profiles['temperature_surface'].groupby('season').mean('time') -273.15
std_temp_surface = mwr_profiles['temperature_surface'].groupby('season').std('time')
mean_rh_surface = mwr_profiles['relative_humidity_surface'].groupby('season').mean('time')
std_rh_surface = mwr_profiles['relative_humidity_surface'].groupby('season').std('time')

for season in SEASONS.keys():
    print(f"Mean temperature at surface for {season} is: {mean_temp_surface.sel(season=season).values:.2f} +- {std_temp_surface.sel(season=season).values:.2f} C")
    print(f"Mean relative humidity at surface for {season} is: {mean_rh_surface.sel(season=season).values:.2f} +- {std_rh_surface.sel(season=season).values:.2f} %")

# Save the dataset as a NetCDF file
if save_data:
    folder_to_save_mwr_ds = f"../../../processed_data/mwr_profiles/"
    if not os.path.exists(folder_to_save_mwr_ds):
        os.makedirs(folder_to_save_mwr_ds)
        # Save mwr_profiles as NetCDF file
        mwr_profiles.to_netcdf(folder_to_save_mwr_ds + 'mwr_profiles.nc')

# Resampling the dataset to monthly frequency
# resampled_merged_mwr = mwr_profiles.resample(time='1M').mean()

# Reindexed dataset for montly frequency with nan in months with no data
# start_time     = resampled_merged_mwr['time'].min().values
# end_time       = resampled_merged_mwr['time'].max().values
# new_time_index = pd.date_range(start=start_time, end=end_time, freq='M', normalize=True)
# reindexed_mwr  = resampled_merged_mwr.reindex(time=new_time_index)

# fig = plt.figure(figsize=(12, 8))
# gs = fig.add_gridspec(2, 1, hspace=0.08)

# # Plotting the Temperature variable using seaborn
# ax1 = fig.add_subplot(gs[0, 0])
# im1 = ax1.pcolormesh(reindexed_mwr.time.values, reindexed_mwr.altitude.values/1e3, reindexed_mwr.temperature.values.T-273.15, shading='auto', cmap='rainbow')
# fig.colorbar(im1, ax=ax1, label=r'Temperature ${\circ}$(C)', aspect=10, ticks=np.arange(-60, 31,10))
# ax1.set_ylabel('Height a.g.l (km)')  # Increase font size of y-axis label
# # ax1.set_title('Temperature Variation')  # Increase font size of title
# ax1.grid(True)  # Add grid lines

# # Customize x-axis tick labels
# ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
# ax1.xaxis.set_tick_params(labelbottom=False)
# # ax1.set_xlim([reindexed_mwr['time'].min().values, datetime.datetime(2024, 2, 1)])

# # Plotting the Relative Humidity variable using seaborn
# ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
# im2 = ax2.pcolormesh(reindexed_mwr.time.values, reindexed_mwr.altitude.values/1e3, reindexed_mwr.relative_humidity.values.T, shading='auto',
#                     cmap='rainbow',
#                     vmin=0,
#                     vmax=70)
# fig.colorbar(im2, ax=ax2, label='Relative Humidity (%)', aspect=10)
# ax2.set_xlabel('Month/Year')  # Increase font size of x-axis label
# ax2.set_ylabel('Height a.g.l (km)')  # Increase font size of y-axis label
# # ax2.set_title('Temperature Variation')  # Increase font size of title
# ax2.grid(True)  # Add grid lines

# # Customize x-axis tick labels
# ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
# ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
# ax2.xaxis.set_tick_params(rotation=30)
# ax2.set_xlim([reindexed_mwr['time'].min().values, datetime.datetime(2024, 1, 1)])

# fig.savefig(PATH_FIG + 'mwr_time_series_TRH.png', dpi=300, bbox_inches='tight')
# plt.show()

monthly_mwr = mwr_profiles.groupby('time.month').mean('time')

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 1, hspace=0.08)

# Plotting the Temperature variable using seaborn
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.pcolormesh(monthly_mwr.month, monthly_mwr.altitude.values/1e3, monthly_mwr.temperature.values.T-273.15, shading='auto', cmap='rainbow')
fig.colorbar(im1, ax=ax1, label=r'Temperature ${\circ}$(C)', aspect=10, ticks=np.arange(-60, 31,10))
ax1.set_ylabel('Height a.g.l (km)')  # Increase font size of y-axis label
# ax1.set_title('Temperature Variation')  # Increase font size of title
ax1.grid(True)  # Add grid lines

# Customize x-axis tick labels
ax1.xaxis.set_tick_params(labelbottom=False)
# ax1.set_xlim([monthly_mwr
#['time'].min().values, datetime.datetime(2024, 2, 1)])

# Plotting the Relative Humidity variable using seaborn
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
im2 = ax2.pcolormesh(monthly_mwr.month, monthly_mwr.altitude.values/1e3, monthly_mwr.relative_humidity.values.T, shading='auto',
                    cmap='rainbow',
                    vmin=0,
                    vmax=60)
fig.colorbar(im2, ax=ax2, label='Relative Humidity (%)', aspect=10)
ax2.set_xlabel('Month/Year')  # Increase font size of x-axis label
ax2.set_ylabel('Height a.g.l (km)')  # Increase font size of y-axis label
# ax2.set_title('Temperature Variation')  # Increase font size of title
ax2.grid(True)  # Add grid lines

# Customize x-axis tick labels
ax2.set_xticks(np.arange(1, 13))
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax2.xaxis.set_tick_params(rotation=30)
# ax2.set_xlim([monthly_mwr['month'].min().values, datetime.datetime(2024, 1, 1)])

fig.savefig(PATH_FIG + 'mwr_time_series_TRH.png', dpi=300, bbox_inches='tight')
plt.show()

set_trace() 

# Plotting the Temperature variable using seaborn
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.pcolormesh(reindexed_mwr.time.values, reindexed_mwr.altitude.values/1e3, reindexed_mwr.humidity.values.T, shading='auto',
                    cmap='rainbow',
                    vmin=0,
                    vmax=7)
fig.colorbar(im, ax=ax, label=r'Humidity (g/m$^{3}$)')
ax.set_xlabel('Month/Year')  # Increase font size of x-axis label
ax.set_ylabel('Altitude (km)')  # Increase font size of y-axis label
# ax.set_title('Temperature Variation')  # Increase font size of title
ax.grid(True)  # Add grid lines

# Customize x-axis tick labels
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
ax.xaxis.set_tick_params(rotation=30)

# ax.set_xlim([reindexed_mwr['time'].min().values, datetime.datetime(2024, 2, 1)])
fig.savefig(PATH_FIG + 'mwr_time_series_H.png', dpi=300, bbox_inches='tight')
plt.show()
