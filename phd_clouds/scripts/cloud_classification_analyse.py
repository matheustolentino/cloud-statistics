import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from pdb import set_trace
import pandas as pd
from phd_clouds.constants import GRANADA_ALTITUDE, SEASONS # in meters
from phd_clouds.utils import get_complete_time, assign_season
import matplotlib.dates as mdates
from scipy import stats
import seaborn as sns
import matplotlib

# matplotlib.use('TkAgg')
PATH_FIG          = '../figures/'
fontsize = 14
# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize

plt.ion()
plt.close('all')

# Define a function to calculate skewness
def calculate_skewness(group):
    return xr.apply_ufunc(
        stats.skew, 
        group,
        input_core_dims=[['time']],
        kwargs={'axis': -1},
        vectorize=True
    )

filepath_cloud_occurence = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification/cloud_occurence" # Path to save the cloud classification files
filepath_cloud_cloud_type = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification/cloud_type" # Path to save the cloud classification files
filepath_cloud_prop = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification/cloud_properties" # Path to save the cloud classification files

load_data = True # Set to True if you want to load the data from the netCDF files

if load_data:
    # Get the list of netCDF files in the specified directory
    file_paths_occurence = [os.path.join(filepath_cloud_occurence, file) for file in os.listdir(filepath_cloud_occurence) if file.endswith('.nc')]
    file_paths_cloud_type = [os.path.join(filepath_cloud_cloud_type, file) for file in os.listdir(filepath_cloud_cloud_type) if file.endswith('.nc')]
    file_paths_cloud_prop = [os.path.join(filepath_cloud_prop, file) for file in os.listdir(filepath_cloud_prop) if file.endswith('.nc')]

    # Read the netCDF files into a list of xarray datasets
    datasets_occurence = [xr.open_dataset(file_path) for file_path in file_paths_occurence]
    datasets_cloud_type = [xr.open_dataset(file_path) for file_path in file_paths_cloud_type]
    datasets_cloud_prop = [xr.open_dataset(file_path) for file_path in file_paths_cloud_prop]

    # Concatenate the datasets_occurence along the time coordinate
    ds_cloud_occurence = xr.concat(datasets_occurence, dim='time').sortby('time')
    ds_cloud_type = xr.concat(datasets_cloud_type, dim='time').sortby('time')
    ds_cloud_prop = xr.concat(datasets_cloud_prop, dim='time').sortby('time')

list_cloud_colors = ["#FFFFFF", "#007CFF", "blue", "cyan", "grey", "yellow", "orange", "magenta"]

# check if the variables are mutually exclusive
time_size           = ds_cloud_occurence['time'].size
sum_cloud_occurence = ds_cloud_occurence.sum(dim='time').to_array().values

# Check if the sum of the cloud occurence is equal to the size of the time dimension
if time_size == np.sum(sum_cloud_occurence[:-1]):
    print("The cloud occurence is mutually exclusive.")

monthly_cloud_occurence = ds_cloud_occurence.groupby('time.month').mean(dim='time')

ds_cloud_prop = assign_season(ds_cloud_prop, SEASONS)

# fig = plt.figure(figsize=(12, 6))
# gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.0005], hspace=0.2)
# ax = fig.add_subplot(gs[0, 0])
# for var in monthly_cloud_occurence.data_vars:
#     if var != 'time':
#         ax.plot(monthly_cloud_occurence['month'], monthly_cloud_occurence[var]*100, '-s', label=var.replace('_', '-').capitalize())  
# ax.set_ylabel('Frequency of occurence (%)')
# ax.set_xlabel('Month')
# ax.grid()
# ax.set_xticks(monthly_cloud_occurence['month'])
# ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
# ax.legend()
# fig.savefig(PATH_FIG + 'new_method_monthly_cloud_occurence.png', dpi=300, bbox_inches='tight')
# plt.show()

#count number of data per month for ds_cloud_occurence:
data_available_per_month = ds_cloud_occurence['single_layer'].groupby('time.month').count(dim='time')

time_complete            = get_complete_time(ds_cloud_occurence['single_layer'], start_month=True)

df                       = pd.DataFrame(index=time_complete)
df['month'] = df.index.month
total_data_per_month     = df.groupby('month').size()

freq_data_available  = data_available_per_month/total_data_per_month


fig1 = plt.figure(figsize=(20, 6))  # Increase the size of the plot for better visibility
gs1  = gs = fig1.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.1)
ax1  = fig1.add_subplot(gs1[0, 0])

color_data_ava = ["#ababab","#ffffff"]
label_data_ava = ["Available", "Missing"]
bar_width = .95 
# bar_width = pd.Timedelta(days=30)
for i in range(2):
    if i == 0:
        freq = freq_data_available
        bot  = np.zeros(freq.month.shape[0])
        hatch="/"
    else:
        freq = 1-freq_data_available
        hatch=None

    p = ax1.bar(freq.month, 100*freq.values,
        label=label_data_ava[i],
        bottom=bot,
        color=color_data_ava[i],
        edgecolor="black",
        width=bar_width,
        hatch = hatch)
    bot += 100*freq.values
# Adding name of the months values
ax1.set_xticks(freq_data_available.month)
ax1.set_ylabel("Data occurence (%)")
ax1.set_xlabel("Months")
# hide x-axis label
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
ax1.set_ylim([0, 101])
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize='large')  # Increase the size of the legend
# add name to months

ax2 = fig1.add_subplot(gs1[0, 1])
for var in monthly_cloud_occurence.data_vars:
    if var != 'time':
        ax2.plot(monthly_cloud_occurence['month'], monthly_cloud_occurence[var]*100, '-s', label=var.replace('_', '-').capitalize())  
ax2.set_ylabel('Frequency of occurence (%)')
ax2.set_xlabel('Month')
ax2.grid()
ax2.set_xticks(monthly_cloud_occurence['month'])
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
ax2.legend()
fig1.savefig(PATH_FIG + 'new_method_monthly_cloud_occurence.png', dpi=300, bbox_inches='tight')
plt.show()

# single_layer_mask = ds_cloud_occurence['single_layer']
single_layer_cloud_type = ds_cloud_type.where(ds_cloud_occurence['single_layer'], other=0)
monthly_single_layer_frequency = single_layer_cloud_type.groupby('time.month').mean(dim='time')

list_var_names = list(monthly_single_layer_frequency.data_vars)
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.0005], hspace=0.2)
ax = fig.add_subplot(gs[0, 0])
for i, var in enumerate(list_var_names[:-1]):
    ax.plot(monthly_single_layer_frequency['month'], monthly_single_layer_frequency[var]*100, '--o', label=var.replace('_', '-').capitalize(),
            color=list_cloud_colors[i+1])  
ax.set_ylabel('Cloud Occurence (%)')
ax.set_xlabel('Month')
# ax.set_title('Single Layer Cloud Frequency')
ax.grid()
ax.set_xticks(monthly_cloud_occurence['month'])
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
ax.legend()
fig.savefig(PATH_FIG + 'new_method_monthly_single_layer_frequency.png', dpi=300, bbox_inches='tight')
plt.show()


# single_layer_mask = ds_cloud_occurence['single_layer']
single_layer_cloud_type = ds_cloud_type.where(ds_cloud_occurence['single_layer'], other=0)
hourly_single_layer_frequency = single_layer_cloud_type.groupby('time.hour').mean(dim='time')

list_var_names = list(hourly_single_layer_frequency.data_vars)
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.0005], hspace=0.2)
ax = fig.add_subplot(gs[0, 0])
for i, var in enumerate(list_var_names[:-1]):
    ax.plot(hourly_single_layer_frequency['hour'], hourly_single_layer_frequency[var]*100, '--o', label=var.replace('_', '-').capitalize(),
            color=list_cloud_colors[i+1])  
ax.set_ylabel('Cloud Occurence (%)')
ax.set_xlabel('Hour')
# ax.set_title('Single Layer Cloud Frequency')
ax.grid()
ax.set_xticks(hourly_single_layer_frequency['hour'])
# ax.set_xticklabels(['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'], rotation=30)
ax.legend()
fig.savefig(PATH_FIG + 'new_method_hourly_single_layer_frequency.png', dpi=300, bbox_inches='tight')
plt.show()

# # Calculate the cloud base height
# # fig, ax         = plt.subplots(1, 1, figsize=(15, 6))
# for i, var in enumerate(list_var_names[:-1]):
#     fig, ax         = plt.subplots(1, 1, figsize=(15, 6))
#     mask_cloud_type = single_layer_cloud_type[var].astype(bool)
#     freq_str = '10D'
#     # mask_effective = (mask_cloud_type.resample(time=freq_str).sum() > 50).astype(float)
#     base_cloud_type_mean = (ds_cloud_prop['cloud_base'].sel(time=mask_cloud_type).resample(time=freq_str).mean() - GRANADA_ALTITUDE)/1000.
#     base_cloud_type_50th = (ds_cloud_prop['cloud_base'].sel(time=mask_cloud_type).resample(time=freq_str).median() - GRANADA_ALTITUDE)/1000.
#     base_cloud_type_10th = (ds_cloud_prop['cloud_base'].sel(time=mask_cloud_type).resample(time=freq_str).quantile(0.1) - GRANADA_ALTITUDE)/1000.
#     base_cloud_type_90th = (ds_cloud_prop['cloud_base'].sel(time=mask_cloud_type).resample(time=freq_str).quantile(0.9) - GRANADA_ALTITUDE)/1000.

#     # ax.plot(base_cloud_type_mean.time, base_cloud_type_mean, '-*r', markersize=4)
#     ax.plot(base_cloud_type_50th.time, base_cloud_type_50th, '-ok', markersize=4)
#     ax.fill_between(base_cloud_type_50th.time, base_cloud_type_10th, base_cloud_type_90th, color=list_cloud_colors[i+1], alpha=0.3, label='10-90 percentile')
    
#     # ax2 = ax.twinx()
#     # skewness_cloud_type = ds_cloud_prop['cloud_base'].sel(time=mask_cloud_type).resample(time=freq_str).map(calculate_skewness)
#     # ax2.plot(skewness_cloud_type.time, skewness_cloud_type, '--sb', markersize=2)
#     # ax2.set_ylabel('Skewness')
#     # ax2.grid()

#     ax.set_ylabel('Cloud Base Height (m)')
#     ax.set_xlabel('Time')
#     ax.set_title(var.replace('_', '-').capitalize())
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
#     ax.xaxis.set_minor_locator(mdates.MonthLocator())
#     ax.grid()
#     plt.show()

# # ax.set_ylabel('Cloud Base Height (m)')
# # ax.set_xlabel('Time')
# # ax.set_title(var.replace('_', '-').capitalize())
# # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
# # ax.xaxis.set_minor_locator(mdates.MonthLocator())
# # ax.grid()
# # plt.show()

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4)

for i, var in enumerate(list_var_names[:-1]):
    ax = fig.add_subplot(gs[i//2, i%2])
    mask_cloud_type = single_layer_cloud_type[var].astype(bool)
    freq_str = '10D'
    base_cloud_type_mean = (ds_cloud_prop['cloud_base'].sel(time=mask_cloud_type).resample(time=freq_str).mean() - GRANADA_ALTITUDE)/1000.
    base_cloud_type_50th = (ds_cloud_prop['cloud_base'].sel(time=mask_cloud_type).resample(time=freq_str).median() - GRANADA_ALTITUDE)/1000.
    base_cloud_type_10th = (ds_cloud_prop['cloud_base'].sel(time=mask_cloud_type).resample(time=freq_str).quantile(0.1) - GRANADA_ALTITUDE)/1000.
    base_cloud_type_90th = (ds_cloud_prop['cloud_base'].sel(time=mask_cloud_type).resample(time=freq_str).quantile(0.9) - GRANADA_ALTITUDE)/1000.

    ax.plot(base_cloud_type_50th.time, base_cloud_type_50th, '-ok', markersize=4)
    ax.fill_between(base_cloud_type_50th.time, base_cloud_type_10th, base_cloud_type_90th, color=list_cloud_colors[i+1], alpha=0.3, label='10-90 percentile')
    
    ax.set_ylabel('Cloud Base Height (m)')
    ax.set_xlabel('Time')
    ax.set_title(var.replace('_', '-').capitalize())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid()
fig.savefig(PATH_FIG + 'new_method_cloud_base_height.png', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4)

for i, var in enumerate(list_var_names[:-1]):
    ax = fig.add_subplot(gs[i//2, i%2])
    mask_cloud_type = single_layer_cloud_type[var].astype(bool)
    freq_str = '10D'
    cloud_thickness_mean = (ds_cloud_prop['cloud_thickness'].sel(time=mask_cloud_type).resample(time=freq_str).mean())/1000.
    cloud_thickness_50th = (ds_cloud_prop['cloud_thickness'].sel(time=mask_cloud_type).resample(time=freq_str).median())/1000.
    cloud_thickness_10th = (ds_cloud_prop['cloud_thickness'].sel(time=mask_cloud_type).resample(time=freq_str).quantile(0.1))/1000.
    cloud_thickness_90th = (ds_cloud_prop['cloud_thickness'].sel(time=mask_cloud_type).resample(time=freq_str).quantile(0.9))/1000.

    ax.plot(cloud_thickness_50th.time, cloud_thickness_50th, '-ok', markersize=4)
    ax.fill_between(cloud_thickness_50th.time, cloud_thickness_10th, cloud_thickness_90th, color=list_cloud_colors[i+1], alpha=0.3, label='10-90 percentile')
    
    ax.set_ylabel('Cloud Thickness (km)')
    ax.set_xlabel('Time')
    ax.set_title(var.replace('_', '-').capitalize())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid()
fig.savefig(PATH_FIG + 'new_method_cloud_thickness.png', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4)

for i, var in enumerate(list_var_names[:-1]):
    ax = fig.add_subplot(gs[i//2, i%2])
    mask_cloud_type = single_layer_cloud_type[var].astype(bool)
    freq_str = '10D'
    cloud_top_mean = (ds_cloud_prop['cloud_top'].sel(time=mask_cloud_type).resample(time=freq_str).mean())/1000.
    cloud_top_50th = (ds_cloud_prop['cloud_top'].sel(time=mask_cloud_type).resample(time=freq_str).median())/1000.
    cloud_top_10th = (ds_cloud_prop['cloud_top'].sel(time=mask_cloud_type).resample(time=freq_str).quantile(0.1))/1000.
    cloud_top_90th = (ds_cloud_prop['cloud_top'].sel(time=mask_cloud_type).resample(time=freq_str).quantile(0.9))/1000.

    ax.plot(cloud_top_50th.time, cloud_top_50th, '-ok', markersize=4)
    ax.fill_between(cloud_top_50th.time, cloud_top_10th, cloud_top_90th, color=list_cloud_colors[i+1], alpha=0.3, label='10-90 percentile')
    
    ax.set_ylabel('Cloud Top Height (km)')
    ax.set_xlabel('Time')
    ax.set_title(var.replace('_', '-').capitalize())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid()
fig.savefig(PATH_FIG + 'new_method_cloud_top.png', dpi=300, bbox_inches='tight')
plt.show()

# single_layer_cloud_type.to_dataframe().reset_index().drop(columns=['time'])

# single_layer_cloud_prop = ds_cloud_prop.sel(time=ds_cloud_occurence['single_layer'].astype(bool))
# single_layer_cloud_prop = single_layer_cloud_prop.assign_coords(year=single_layer_cloud_prop['time'].dt.year, month=single_layer_cloud_prop['time'].dt.month)
# single_layer_cloud_prop = assign_season(single_layer_cloud_prop, SEASONS)

# df_cloud_prop = (single_layer_cloud_prop/1000).to_dataframe().reset_index().drop(columns=['time', 'month', 'year'])

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4)

for i, var in enumerate(list_var_names[:-1]):
    ax = fig.add_subplot(gs[i//2, i%2])
    mask_cloud_type = single_layer_cloud_type[var].astype(bool)

    base_cloud_type = ds_cloud_prop['cloud_base'].sel(time=mask_cloud_type)/1000.
    
    sns.violinplot(x=base_cloud_type.time.dt.month.values, y=base_cloud_type.values, ax=ax, color=list_cloud_colors[i+1], density_norm='count', inner="quart")
    # sns.violinplot(x=base_cloud_type.time.dt.season.values, y=base_cloud_type.values, ax=ax, color=list_cloud_colors[i+1])
    
    ax.set_ylabel('Cloud Base Height (m)')
    ax.set_xlabel('Month')
    ax.set_title(var.replace('_', '-').capitalize())
    # ax.set_xticks(range(1, 13))
    # ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.grid()
fig.savefig(PATH_FIG + 'new_method_cloud_base_height_violin_season.png', dpi=300, bbox_inches='tight')
plt.show()

ds_cloud_type_single_layer = ds_cloud_type.sel(time=ds_cloud_occurence['single_layer'].astype(bool))
df_cloud_type_single_layer = ds_cloud_type_single_layer.to_dataframe().reset_index().melt(id_vars=['time'], var_name='cloud_type', value_name='cloud_occurence').drop(columns=['time'])


