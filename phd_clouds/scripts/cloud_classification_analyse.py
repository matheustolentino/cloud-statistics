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
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('matplotlib', 'inline')
    
PATH_FIG          = '../figures/'
fontsize = 14
# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize

# plt.ion()
plt.close('all')
# mpl.use('TkAgg')

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

if use_old_method:
    filepath_cloud_occurence = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification_without_filtering_ice_ABL/cloud_occurence" # Path to save the cloud classification files
    filepath_cloud_cloud_type = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification_without_filtering_ice_ABL/cloud_type" # Path to save the cloud classification files
    filepath_cloud_prop = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification_without_filtering_ice_ABL/cloud_properties" # Path to save the cloud classification files
    filepath_categorize = "/media/matheustolen/Seagate Basic/cloudnet/categorize" # Path to save the cloud classification files
else:
    filepath_cloud_occurence = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification/cloud_occurence" # Path to save the cloud classification files
    filepath_cloud_cloud_type = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification/cloud_type" # Path to save the cloud classification files
    filepath_cloud_prop = "/media/matheustolen/Seagate Basic/cloudnet/cloud_classification/cloud_properties" # Path to save the cloud classification files
    filepath_categorize = "/media/matheustolen/Seagate Basic/cloudnet/categorize" # Path to save the cloud classification files

if load_data:
    # Get the list of netCDF files in the specified directory
    file_paths_occurence = [os.path.join(filepath_cloud_occurence, file) for file in os.listdir(filepath_cloud_occurence) if file.endswith('.nc')]
    file_paths_cloud_type = [os.path.join(filepath_cloud_cloud_type, file) for file in os.listdir(filepath_cloud_cloud_type) if file.endswith('.nc')]
    file_paths_cloud_prop = [os.path.join(filepath_cloud_prop, file) for file in os.listdir(filepath_cloud_prop) if file.endswith('.nc')]
    file_path_categorize = [os.path.join(filepath_categorize, file) for file in os.listdir(filepath_categorize) if file.endswith('.nc')]
    
    # Read the netCDF files into a list of xarray datasets
    print("Reading the netCDF files into a list of xarray datasets of new method...")
    datasets_occurence = [xr.open_dataset(file_path) for file_path in file_paths_occurence]
    datasets_cloud_type = [xr.open_dataset(file_path) for file_path in file_paths_cloud_type]
    datasets_cloud_prop = [xr.open_dataset(file_path) for file_path in file_paths_cloud_prop]
    datasets_categorize = [xr.open_dataset(file_path)['lwp'] for file_path in file_path_categorize]
    
    print("Concatenating the datasets along the time coordinate...")
    # Concatenate the datasets_occurence along the time coordinate
    ds_cloud_occurence = xr.concat(datasets_occurence, dim='time').sortby('time')
    ds_cloud_type = xr.concat(datasets_cloud_type, dim='time').sortby('time')
    ds_cloud_prop = xr.concat(datasets_cloud_prop, dim='time').sortby('time')
    ds_categorize = xr.concat(datasets_categorize, dim='time').sortby('time')
    
    # -----------------------------------------------------------------------
    # Old Method of cloud classification 
    # -----------------------------------------------------------------------
    root_folder = '../../../processed_data/'
    target_parent_folder = "number_of_layers"
    # chirp_lwp = reading_dataset_chunking(root_folder, target_parent_folder)
    print("Reading the netCDF files into a list of xarray datasets of old method...")
    chirp_layers = reading_dataset_chunking(root_folder, target_parent_folder)

    layer_list   = [ds for ds in chirp_layers.values()]
    cloud_layers = xr.concat(layer_list, dim='time').sortby('time')

    df_layers     = cloud_layers.to_dataframe()
    mask_single   = (df_layers.sum(axis=1) == 1.0).to_numpy() # Mask with single layer for any kind of cloud
    mask_multi    = (df_layers.sum(axis=1) > 1.0) # Mask with multi layer clouds
    mask_w_clouds = (df_layers.sum(axis=1) == 0.0) # Mask with no clouds
    del df_layers

    #-------------------------------------------------------------------------------------------------
    # NOTE: New variable is a mask, keep this sequence to avoid integer greater thabn 1
    #-------------------------------------------------------------------------------------------------
    old_method_cloud_ava = cloud_layers.where(mask_single, other=0.0)
    old_method_cloud_ava['no_clouds']  = xr.DataArray(mask_w_clouds.astype(np.float64), dims='time')
    old_method_cloud_ava['multilayer'] = xr.DataArray(mask_multi.astype(np.float64), dims='time')
    old_method_cloud_ava = old_method_cloud_ava.assign_coords(years=old_method_cloud_ava['time'].dt.year,
                                                               month=old_method_cloud_ava['time'].dt.month
                                                               ).compute()
    print("End of reading the netCDF files into a list of xarray datasets...")


list_cloud_colors = ["#FFFFFF", "#007CFF", "blue", "cyan", "grey", "yellow", "orange", "magenta"]

# check if the variables are mutually exclusive
time_size           = ds_cloud_occurence['time'].size
sum_cloud_occurence = ds_cloud_occurence.sum(dim='time').to_array().values

# Check if the sum of the cloud occurence is equal to the size of the time dimension
if time_size == np.sum(sum_cloud_occurence[:-1]):
    print("The cloud occurence is mutually exclusive.")

monthly_cloud_occurence = ds_cloud_occurence.groupby('time.month').mean(dim='time')

ds_cloud_prop = assign_season(ds_cloud_prop, SEASONS)

if use_old_method:
    # single_layer_mask = ds_cloud_occurence['single_layer']
    ds_cloud_type = assign_season(ds_cloud_type, SEASONS)
    single_layer_cloud_type = ds_cloud_type.where(ds_cloud_occurence['single_layer'], other=0)
    monthly_single_layer_frequency = single_layer_cloud_type.groupby('time.month').mean(dim='time')
    
    list_var_names = list(monthly_single_layer_frequency.data_vars)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(len(SEASONS)//2, 2, height_ratios=[1]*(len(SEASONS)//2), hspace=0.15)
    season_order_to_follow= ['winter', 'spring', 'summer','fall']
    for i, season in enumerate(season_order_to_follow):
        season_mask = ds_cloud_type['season'] == season
        ax = fig.add_subplot(gs[i//2, i%2])
        hourly_season_frequency = single_layer_cloud_type.where(season_mask).groupby('time.hour').mean(dim='time')
        for var in list_var_names[:-1]:
            ax.plot(hourly_season_frequency['hour'], hourly_season_frequency[var]*100, '-', 
                label=var.replace('_', '-').capitalize(),
                color=list_cloud_colors[list_var_names.index(var)+1], markersize=6,
                marker='o', markeredgecolor='black')
        ax.set_ylabel('Cloud Occurence (%)')
        if i ==2 or i == 3:
            ax.set_xlabel('Hourly Daytime UTC')
        else:
            ax.xaxis.set_tick_params(labelbottom=False)
        ax.set_title(season.capitalize())
        ax.grid()
        # ax.set_xticks(hourly_season_frequency['hour'])
        ax.set_xticks(hourly_season_frequency['hour'][::3])
        ax.set_xticklabels(['00', '03', '06', '09', '12', '15', '18', '21'], rotation=40)
        # ax.set_xticklabels(['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'], rotation=40)
        if i == 0:
            ax.legend(loc='upper right', fontsize='small', ncol=2)
        ax.set_ylim([0, 20])
    fig.savefig(PATH_FIG + 'hourly_seasonly_single_layer_frequency_old_method.png', dpi=300, bbox_inches='tight')
    plt.show()

    resample_single_layer_frequency = single_layer_cloud_type.resample(time='1D').mean()
    resample_old_method_cloud_ava = old_method_cloud_ava.resample(time='1D').mean()
    pearson_corr = resample_single_layer_frequency.to_dataframe().drop(columns=['Noise']).corr(method='pearson')
    pearson_corr_old_method = resample_old_method_cloud_ava.to_dataframe().drop(columns=['no_clouds', 'multilayer']).corr(method='pearson')
    # plot upper triangle of the matrix of correlation coefficients

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 2, wspace=0.1, hspace=1,  width_ratios=[1, 1], height_ratios=[1,.0005])

    # Plot for pearson_corr
    ax1 = fig.add_subplot(gs[0, 0])
    heat1 = ax1.imshow(pearson_corr,
                    cmap='coolwarm',
                    vmin=-1, vmax=1)
    ax1.set_xticks(np.arange(len(pearson_corr.columns)))
    ax1.set_yticks(np.arange(len(pearson_corr.columns)))
    ax1.set_xticklabels(pearson_corr.columns, rotation=60)
    ax1.set_yticklabels(pearson_corr.columns)
    for i in range(len(pearson_corr.columns)):
        for j in range(len(pearson_corr.columns)):
            text = ax1.text(j, i, round(pearson_corr.iloc[i, j], 2),
                            ha="center", va="center", color="black")
    ax1.set_title('New Method')

    # cax1 = fig.add_subplot(gs[0, 1])
    # cbar1 = plt.colorbar(heat1, label='Correlation Coefficient', cax=cax1)

    # Plot for pearson_corr_old_method
    ax2 = fig.add_subplot(gs[0, 1])
    heat2 = ax2.imshow(pearson_corr_old_method,
                    cmap='coolwarm',
                    vmin=-1, vmax=1)
    ax2.set_xticks(np.arange(len(pearson_corr_old_method.columns)))
    ax2.set_yticks(np.arange(len(pearson_corr_old_method.columns)))
    ax2.set_xticklabels(pearson_corr_old_method.columns, rotation=60)
    ax2.set_yticklabels(pearson_corr_old_method.columns)
    for i in range(len(pearson_corr_old_method.columns)):
        for j in range(len(pearson_corr_old_method.columns)):
            text = ax2.text(j, i, round(pearson_corr_old_method.iloc[i, j], 2),
                            ha="center", va="center", color="black")
    ax2.set_title('Old Method')
    # cax2 = fig.add_subplot(gs[1, :])
    # cbar2 = plt.colorbar(heat2, label='Pearson Correlation Coefficient', cax=cax2, orientation='horizontal')
    # fig.savefig(PATH_FIG + 'cloud_freq_methods_comparison_correlation_coefficient.png', dpi=300, bbox_inches='tight')
    plt.show()

    clouds_to_analyse_old_method =  ["Liquid", "Pre_liquid", "Ice", "Mixed_phase", "Pre_mixed_phase"]
    new_list_color               = ["#007CFF", "blue", "cyan", "yellow", "orange"]
    monthly_old_method_cloud_occurence = old_method_cloud_ava[clouds_to_analyse_old_method].groupby('time.month').mean(dim='time')

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    for i, var in enumerate(list_var_names[:-1]):
        # ax.plot(monthly_single_layer_frequency['month'], monthly_single_layer_frequency[var]*100, '--o', label=var.replace('_', '-').capitalize(),
        #     color=list_cloud_colors[i+1])
        ax.scatter(monthly_single_layer_frequency['month'], monthly_single_layer_frequency[var]*100, label=var.replace('_', '-').capitalize(),
            color=list_cloud_colors[i+1], s=150, alpha=0.6, edgecolors='black', linewidth=1.5)
        ax.plot(monthly_single_layer_frequency['month'], monthly_single_layer_frequency[var]*100, '-',
                color=list_cloud_colors[i+1], linewidth=1.5, alpha=0.6)
    ax.set_ylabel('Cloud Occurence (%)')
    ax.set_xlabel('Month')
    # ax.set_title('Single Layer Cloud Frequency')
    ax.grid()
    # ax.xaxis.set_tick_params(labelbottom=False)
    ax.set_xticks(monthly_cloud_occurence['month'])
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
    ax.legend()
    
    ax2 = fig.add_subplot(gs[1, 0])
    for i, var in enumerate(clouds_to_analyse_old_method):
        ax2.scatter(monthly_old_method_cloud_occurence['month'], monthly_old_method_cloud_occurence[var]*100, label=var.replace('_', '-').capitalize(),
            color=new_list_color[clouds_to_analyse_old_method.index(var)], s=150, alpha=0.6, edgecolors='black', linewidth=1.5)
        ax2.plot(monthly_old_method_cloud_occurence['month'], monthly_old_method_cloud_occurence[var]*100, '-',
                color=new_list_color[clouds_to_analyse_old_method.index(var)], linewidth=1.5, alpha=0.6)
    ax2.set_ylabel('Cloud Occurence (%)')
    ax2.set_xlabel('Month')
    ax2.legend()
    ax2.set_xticks(monthly_old_method_cloud_occurence['month'])
    ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
    ax2.grid()

    fig.savefig(PATH_FIG + 'comparison_inter_anual_coud_occurence_old_new_method.png', dpi=300, bbox_inches='tight')
    plt.show()



else:
    # #count number of data per month for ds_cloud_occurence:
    # data_available_per_month = ds_cloud_occurence['single_layer'].groupby('time.month').count(dim='time')

    # time_complete            = get_complete_time(ds_cloud_occurence['single_layer'], start_month=True)

    # df                       = pd.DataFrame(index=time_complete)
    # df['month'] = df.index.month
    # total_data_per_month     = df.groupby('month').size()

    # freq_data_available  = data_available_per_month/total_data_per_month


    # fig1 = plt.figure(figsize=(20, 6))  # Increase the size of the plot for better visibility
    # gs1  = gs = fig1.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.1)
    # ax1  = fig1.add_subplot(gs1[0, 0])

    # color_data_ava = ["#ababab","#ffffff"]
    # label_data_ava = ["Available", "Missing"]
    # bar_width = .95 
    # # bar_width = pd.Timedelta(days=30)
    # for i in range(2):
    #     if i == 0:
    #         freq = freq_data_available
    #         bot  = np.zeros(freq.month.shape[0])
    #         hatch="/"
    #     else:
    #         freq = 1-freq_data_available
    #         hatch=None

    #     p = ax1.bar(freq.month, 100*freq.values,
    #         label=label_data_ava[i],
    #         bottom=bot,
    #         color=color_data_ava[i],
    #         edgecolor="black",
    #         width=bar_width,
    #         hatch = hatch)
    #     bot += 100*freq.values
    # # Adding name of the months values
    # ax1.set_xticks(freq_data_available.month)
    # ax1.set_ylabel("Data occurence (%)")
    # ax1.set_xlabel("Months")
    # # hide x-axis label
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
    # ax1.set_ylim([0, 101])
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize='large')  # Increase the size of the legend
    # # add name to months

    # ax2 = fig1.add_subplot(gs1[0, 1])
    # for var in monthly_cloud_occurence.data_vars:
    #     if var != 'time':
    #         ax2.plot(monthly_cloud_occurence['month'], monthly_cloud_occurence[var]*100, '-s', label=var.replace('_', '-').capitalize())  
    # ax2.set_ylabel('Frequency of occurence (%)')
    # ax2.set_xlabel('Month')
    # ax2.grid()
    # ax2.set_xticks(monthly_cloud_occurence['month'])
    # ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
    # ax2.legend()
    # fig1.savefig(PATH_FIG + 'new_method_monthly_cloud_occurence.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # -----------------------------------------------------------------------
    # Data Availability plot
    # -----------------------------------------------------------------------

    ds_cloud_occurence   = ds_cloud_occurence.assign_coords(year=ds_cloud_occurence['time'].dt.year, month=ds_cloud_occurence['time'].dt.month)
    count_available_data = ds_cloud_occurence['single_layer'].groupby('time.month').count(dim='time')
    time_complete            = get_complete_time(ds_cloud_occurence, start_month=True)

    df          = pd.DataFrame(index=time_complete)
    df['month'] = df.index.month

    total_data_per_month = df.groupby('month').size()
    unique_years         = np.unique(ds_cloud_occurence['year'].values)
    str_year             = [str(year) for year in unique_years]

    # color_data_ava = ["#ababab","#ffffff"]
    color_data_ava = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff", "#ff00ff", "#ffa500", "#ababab"]
    label_data_ava = str_year + ['Missing Data']

    fig1 = plt.figure(figsize=(20, 6))  # Increase the size of the plot for better visibility
    gs1  = gs = fig1.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.1)
    ax1  = fig1.add_subplot(gs1[0, 0])

    bottom = np.zeros(len(total_data_per_month))
    for year in unique_years:
        sliced_year = ds_cloud_occurence['single_layer'].where(ds_cloud_occurence['year'] == year)
        data_available_per_month = sliced_year.groupby('time.month').count(dim='time')

        freq_data_available  = 100*data_available_per_month/total_data_per_month
        
        bar_width = .95 
        ax1.bar(freq_data_available.month, freq_data_available.values, bar_width,
                label=str(year), color=color_data_ava[unique_years.tolist().index(year)],
                bottom=bottom)
        bottom += freq_data_available.values

    # Adding missing data BAR 
    missing_data = (1 - count_available_data/total_data_per_month)*100
    ax1.bar(missing_data.month, missing_data.values, bar_width, label='Missing Data', color=color_data_ava[-1],
            bottom=bottom)

    # Adding name of the months values
    ax1.set_xticks(freq_data_available.month)
    ax1.set_ylabel("Data occurence (%)")
    ax1.set_xlabel("Months")
    # hide x-axis label
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
    ax1.set_ylim([0, 101])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(unique_years)//2+1, frameon=False, fontsize='small')  # Decrease the size of the legend
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


    fig2 = plt.figure(figsize=(12, 6))
    ax2 = fig2.add_subplot(111)
    for var in monthly_cloud_occurence.data_vars:
        if var != 'time':
            ax2.plot(monthly_cloud_occurence['month'], monthly_cloud_occurence[var]*100, '-s', label=var.replace('_', '-').capitalize())  
    ax2.set_ylabel('Frequency of occurence (%)')
    ax2.set_xlabel('Month')
    ax2.grid()
    ax2.set_xticks(monthly_cloud_occurence['month'])
    ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
    ax2.legend()
    fig2.savefig(PATH_FIG + 'new_method_monthly_cloud_occurence.png', dpi=300, bbox_inches='tight')
    plt.show()

    # color_data_ava = sns.color_palette("deep", len(unique_years)+1).as_hex() + ["#FFFFFF"]
    # # Rest of the code...
    # fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    # bottom = np.zeros(len(total_data_per_month))
    # for year in unique_years:
    #     sliced_year = ds_cloud_occurence['single_layer'].where(ds_cloud_occurence['year'] == year)
    #     data_available_per_month = sliced_year.groupby('time.month').count(dim='time')

    #     freq_data_available  = 100*data_available_per_month/total_data_per_month
    #     ax.bar(freq_data_available.month, freq_data_available.values, 0.95,
    #              label=str(year), color=color_data_ava[unique_years.tolist().index(year)],
    #              bottom=bottom)
    #     bottom += freq_data_available.values

    # # Adding missing data BAR
    # missing_data = (1 - count_available_data/total_data_per_month)*100
    # ax.bar(missing_data.month, missing_data.values, 0.95, label='Missing Data', color=color_data_ava[-1],
    #         bottom=bottom)

    # # Adding name of the months values
    # ax.set_xticks(freq_data_available.month)
    # ax.set_ylabel("Data occurence (%)")
    # ax.set_xlabel("Months")
    # # hide x-axis label
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
    # ax.set_ylim([0, 101])
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(unique_years)//2+1, frameon=False, fontsize='medium', edgecolor='black')  # Decrease the size of the legend
    # fig.savefig(PATH_FIG + 'new_method_separated_monthly_data_occurence.png', dpi=300, bbox_inches='tight')
    # plt.show()


    # single_layer_mask = ds_cloud_occurence['single_layer']
    single_layer_cloud_type = ds_cloud_type.where(ds_cloud_occurence['single_layer'], other=0)
    monthly_single_layer_frequency = single_layer_cloud_type.groupby('time.month').mean(dim='time')

    list_var_names = list(monthly_single_layer_frequency.data_vars)
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.0005], hspace=0.2)
    ax = fig.add_subplot(gs[0, 0])
    for i, var in enumerate(list_var_names[:-1]):
        # ax.plot(monthly_single_layer_frequency['month'], monthly_single_layer_frequency[var]*100, '--o', label=var.replace('_', '-').capitalize(),
        #     color=list_cloud_colors[i+1])
        ax.scatter(monthly_single_layer_frequency['month'], monthly_single_layer_frequency[var]*100, label=var.replace('_', '-').capitalize(),
            color=list_cloud_colors[i+1], s=150, alpha=0.6, edgecolors='black', linewidth=1.5)
        ax.plot(monthly_single_layer_frequency['month'], monthly_single_layer_frequency[var]*100, '-',
                color=list_cloud_colors[i+1], linewidth=1.5, alpha=0.6)
    ax.set_ylabel('Cloud Occurence (%)')
    ax.set_xlabel('Month')
    # ax.set_title('Single Layer Cloud Frequency')
    ax.grid()
    ax.set_xticks(monthly_cloud_occurence['month'])
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
    ax.legend()
    fig.savefig(PATH_FIG + 'new_method_monthly_single_layer_frequency.png', dpi=300, bbox_inches='tight')
    plt.show()
    

    # list_var = list(ds_cloud_type.data_vars)
    # fig = plt.figure(figsize=(19, 12))
    # gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], hspace=0.4, wspace=0.25)
    # mask = (ds_cloud_occurence['single_layer'].astype(bool)) & (ds_cloud_occurence['noise'] == 0)
    # single_layer_categorize = ds_categorize.where(mask)
    # for i, cloud in enumerate(list_var[:-1]):
    #     # data = single_layer_cloud_type.where(ds_cloud_type[cloud].astype(bool))
    #     data = single_layer_categorize.where(ds_cloud_type[cloud].astype(bool))
    #     ax = fig.add_subplot(gs[i])
    #     for year in np.unique(data.time.dt.year):
    #         # data_year = data[cloud].sel(time=data.time.dt.year == year)
    #         data_year = data.sel(time=data.time.dt.year == year)
    #         daily_data = data_year.groupby('time.dayofyear').mean(dim='time')
    #         ax.plot(daily_data.dayofyear, daily_data,'--o',  label=f"{year}")
    #     ax.set_xlabel("Day of Year")
    #     ax.set_ylabel(f"{cloud} ({single_layer_categorize.units})")
    #     ax.set_title(f"{cloud} Clouds")
    #     if i == 0:
    #         ax.legend()
    # plt.show()


    # heat = sns.heatmap(pearson_corr,
    #                    mask=mask,
    #                    annot=True,
    #                    fmt=".2f",  # Display values with 2 decimal places
    #                    vmin=-1, vmax=1, center=0, 
    #                    cmap='coolwarm', cbar_kws={'label': 'Correlation Coefficient'},
    #                    ax=ax)

    # ax.set_xticklabels(ax.get_xticklabels(), rotation=60)  # Rotate x-axis labels by 60 degrees

    # plt.show()

    # # single_layer_mask = ds_cloud_occurence['single_layer']
    # single_layer_cloud_type = ds_cloud_type.where(ds_cloud_occurence['single_layer'], other=0)
    # hourly_single_layer_frequency = single_layer_cloud_type.groupby('time.hour').mean(dim='time')

    # list_var_names = list(hourly_single_layer_frequency.data_vars)
    # fig = plt.figure(figsize=(12, 6))
    # gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.0005], hspace=0.2)
    # ax = fig.add_subplot(gs[0, 0])
    # for i, var in enumerate(list_var_names[:-1]):
    #     ax.plot(hourly_single_layer_frequency['hour'], hourly_single_layer_frequency[var]*100, '--o', label=var.replace('_', '-').capitalize(),
    #             color=list_cloud_colors[i+1])  
    # ax.set_ylabel('Cloud Occurence (%)')
    # ax.set_xlabel('Hour')
    # # ax.set_title('Single Layer Cloud Frequency')
    # ax.grid()
    # ax.set_xticks(hourly_single_layer_frequency['hour'])
    # # ax.set_xticklabels(['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'], rotation=30)
    # ax.legend()
    # fig.savefig(PATH_FIG + 'new_method_hourly_single_layer_frequency.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # -----------------------------------------------------------------------
    # ----------------- Hourly Single Layer Cloud Frequency -----------------
    # -----------------------------------------------------------------------
    # single_layer_mask = ds_cloud_occurence['single_layer']
    ds_cloud_type = assign_season(ds_cloud_type, SEASONS)
    single_layer_cloud_type = ds_cloud_type.where(ds_cloud_occurence['single_layer'], other=0)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(len(SEASONS)//2, 2, height_ratios=[1]*(len(SEASONS)//2), hspace=0.15)
    season_order_to_follow= ['winter', 'spring', 'summer','fall']
    for i, season in enumerate(season_order_to_follow):
        season_mask = ds_cloud_type['season'] == season
        ax = fig.add_subplot(gs[i//2, i%2])
        hourly_season_frequency = single_layer_cloud_type.where(season_mask).groupby('time.hour').mean(dim='time')
        for var in list_var_names[:-1]:
            ax.plot(hourly_season_frequency['hour'], hourly_season_frequency[var]*100, '-', 
                label=var.replace('_', '-').capitalize(),
                color=list_cloud_colors[list_var_names.index(var)+1], markersize=6,
                marker='o', markeredgecolor='black')
        ax.set_ylabel('Cloud Occurence (%)')
        if i ==2 or i == 3:
            ax.set_xlabel('Hourly Daytime UTC')
        else:
            ax.xaxis.set_tick_params(labelbottom=False)
        ax.set_title(season.capitalize())
        ax.grid()
        # ax.set_xticks(hourly_season_frequency['hour'])
        ax.set_xticks(hourly_season_frequency['hour'][::3])
        ax.set_xticklabels(['00', '03', '06', '09', '12', '15', '18', '21'], rotation=40)
        # ax.set_xticklabels(['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'], rotation=40)
        if i == 0:
            ax.legend(loc='upper right', fontsize='small', ncol=2)
        ax.set_ylim([0, 10])
    fig.savefig(PATH_FIG + 'hourly_seasonly_single_layer_frequency.png', dpi=300, bbox_inches='tight')
    plt.show()

    single_layer_cloud_base = ds_cloud_prop['cloud_base'].where(ds_cloud_occurence['single_layer'], other=np.nan)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(len(SEASONS)//2, 2, height_ratios=[1]*(len(SEASONS)//2), hspace=0.15)
    season_order_to_follow= ['winter', 'spring', 'summer','fall']
    for i, season in enumerate(season_order_to_follow):
        season_mask = ds_cloud_type['season'] == season
        ax = fig.add_subplot(gs[i//2, i%2])
        for var in list_var_names[:-1]:
            cloud_type_mask = single_layer_cloud_type[var].astype(bool)
            total_mask      = cloud_type_mask & season_mask
            cloud_base_type = single_layer_cloud_base.where(total_mask).groupby('time.hour').mean(dim='time')
            std_cloud_base_type = single_layer_cloud_base.where(total_mask).groupby('time.hour').std(dim='time')
            # print(f"season: {season}, var: {var}, mean: {cloud_base_type.mean().values}, std: {std_cloud_base_type.mean().values}")
            ax.plot(cloud_base_type['hour'], cloud_base_type, '-', 
                label=var.replace('_', '-').capitalize(),
                color=list_cloud_colors[list_var_names.index(var)+1], markersize=6,
                marker='o', markeredgecolor='black')
            ax.fill_between(cloud_base_type['hour'], cloud_base_type-std_cloud_base_type, cloud_base_type+std_cloud_base_type, color=list_cloud_colors[list_var_names.index(var)+1], alpha=0.3)
        ax.set_ylabel(f"{'cloud_base'}, m")
        if i ==2 or i == 3:
            ax.set_xlabel('Hourly Daytime UTC')
        else:
            ax.xaxis.set_tick_params(labelbottom=False)
        ax.set_title(season.capitalize())
        ax.grid()
        # ax.set_xticks(hourly_season_frequency['hour'])
        ax.set_xticks(hourly_season_frequency['hour'][::3])
        ax.set_xticklabels(['00', '03', '06', '09', '12', '15', '18', '21'], rotation=40)
        # ax.set_xticklabels(['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'], rotation=40)
        if i == 0:
            ax.legend(loc='upper right', fontsize='small', ncol=2)
        # ax.set_ylim([0, 23])
    # fig.savefig(PATH_FIG + 'hourly_seasonly_single_layer_cloud_base.png', dpi=300, bbox_inches='tight')
    plt.show()


    # single_layer_lwp = ds_categorize.where(ds_cloud_occurence['single_layer'], other=np.nan)
    # fig = plt.figure(figsize=(14, 8))
    # gs = fig.add_gridspec(len(SEASONS)//2, 2, height_ratios=[1]*(len(SEASONS)//2), hspace=0.15)
    # season_order_to_follow= ['winter', 'spring', 'summer','fall']
    # for i, season in enumerate(season_order_to_follow):
    #     season_mask = ds_cloud_type['season'] == season
    #     ax = fig.add_subplot(gs[i//2, i%2])
    #     for var in list_var_names[:-1]:
    #         cloud_type_mask = single_layer_cloud_type[var].astype(bool)
    #         total_mask      = cloud_type_mask & season_mask
    #         lwp_cloud_type = single_layer_lwp.where(total_mask).groupby('time.hour').mean(dim='time')
    #         std_lwp_cloud_type = single_layer_lwp.where(total_mask).groupby('time.hour').std(dim='time')
    #         ax.plot(lwp_cloud_type['hour'], lwp_cloud_type, '-', 
    #             label=var.replace('_', '-').capitalize(),
    #             color=list_cloud_colors[list_var_names.index(var)+1], markersize=6,
    #             marker='o', markeredgecolor='black')
    #         # ax.fill_between(lwp_cloud_type['hour'], lwp_cloud_type-std_lwp_cloud_type, lwp_cloud_type+std_lwp_cloud_type, color=list_cloud_colors[list_var_names.index(var)+1], alpha=0.3)
    #     ax.set_ylabel(f"LWP, {lwp_cloud_type.units}")
    #     if i ==2 or i == 3:
    #         ax.set_xlabel('Hourly Daytime UTC')
    #     else:
    #         ax.xaxis.set_tick_params(labelbottom=False)
    #     ax.set_title(season.capitalize())
    #     ax.grid()
    #     # ax.set_xticks(hourly_season_frequency['hour'])
    #     ax.set_xticks(hourly_season_frequency['hour'][::3])
    #     ax.set_xticklabels(['00', '03', '06', '09', '12', '15', '18', '21'], rotation=40)
    #     # ax.set_xticklabels(['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'], rotation=40)
    #     if i == 0:
    #         ax.legend(loc='upper right', fontsize='small', ncol=2)
    #     # ax.set_ylim([0, 23])
    # # fig.savefig(PATH_FIG + 'hourly_seasonly_single_layer_lwp.png', dpi=300, bbox_inches='tight')
    # plt.show()


    # mask_large_cloud_thickeness = ds_cloud_prop['cloud_thickness'] < 1000
    # fig = plt.figure(figsize=(14, 8))
    # gs = fig.add_gridspec(len(SEASONS)//2, 2, height_ratios=[1]*(len(SEASONS)//2), hspace=0.15)
    # season_order_to_follow= ['winter', 'spring', 'summer','fall']
    # for i, season in enumerate(season_order_to_follow):
    #     season_mask = ds_cloud_type['season'] == season
    #     ax = fig.add_subplot(gs[i//2, i%2])
    #     for var in list_var_names[:-1]:
    #         cloud_type_mask = single_layer_cloud_type[var].astype(bool)
    #         total_mask      = cloud_type_mask & season_mask
    #         target_cloud = mask_large_cloud_thickeness.where(total_mask).groupby('time.hour').mean(dim='time')
    #         ax.plot(target_cloud['hour'], target_cloud, '-', 
    #             label=var.replace('_', '-').capitalize(),
    #             color=list_cloud_colors[list_var_names.index(var)+1], markersize=6,
    #             marker='o', markeredgecolor='black')
    #     ax.set_ylabel(f"{'cloud_thickness'}, m")
    #     if i ==2 or i == 3:
    #         ax.set_xlabel('Hourly Daytime UTC')
    #     else:
    #         ax.xaxis.set_tick_params(labelbottom=False)
    #     ax.set_title(season.capitalize())
    #     ax.grid()
    #     # ax.set_xticks(hourly_season_frequency['hour'])
    #     ax.set_xticks(hourly_season_frequency['hour'][::3])
    #     ax.set_xticklabels(['00', '03', '06', '09', '12', '15', '18', '21'], rotation=40)
    #     # ax.set_xticklabels(['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'], rotation=40)
    #     if i == 0:
    #         ax.legend(loc='upper right', fontsize='small', ncol=2)
    #     # ax.set_ylim([0, 23])
    # # fig.savefig(PATH_FIG + 'hourly_seasonly_single_layer_cloud_base.png', dpi=300, bbox_inches='tight')
    # plt.show()


    # -----------------------------------------------------------------------

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
        freq_str = 'M'
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

    mask_cloud_type = single_layer_cloud_type['Ice-Precipitable'].astype(bool)
    test = (ds_cloud_prop['cloud_base'].sel(time=mask_cloud_type).resample(time=freq_str).mean() - GRANADA_ALTITUDE)/1000
    df_test = test.to_dataframe().dropna()
    s=sm.tsa.seasonal_decompose(df_test.cloud_base, model="additive", period=3)
    s.plot()
    # # -----------------------------------------------------------------------
    # # Cloud Thickness
    # # -----------------------------------------------------------------------

    # fig = plt.figure(figsize=(15, 10))
    # gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4)

    # for i, var in enumerate(list_var_names[:-1]):
    #     ax = fig.add_subplot(gs[i//2, i%2])
    #     mask_cloud_type = single_layer_cloud_type[var].astype(bool)
    #     freq_str = '10D'
    #     cloud_thickness_mean = (ds_cloud_prop['cloud_thickness'].sel(time=mask_cloud_type).resample(time=freq_str).mean())/1000.
    #     cloud_thickness_50th = (ds_cloud_prop['cloud_thickness'].sel(time=mask_cloud_type).resample(time=freq_str).median())/1000.
    #     cloud_thickness_10th = (ds_cloud_prop['cloud_thickness'].sel(time=mask_cloud_type).resample(time=freq_str).quantile(0.1))/1000.
    #     cloud_thickness_90th = (ds_cloud_prop['cloud_thickness'].sel(time=mask_cloud_type).resample(time=freq_str).quantile(0.9))/1000.

    #     ax.plot(cloud_thickness_50th.time, cloud_thickness_50th, '-ok', markersize=4)
    #     ax.fill_between(cloud_thickness_50th.time, cloud_thickness_10th, cloud_thickness_90th, color=list_cloud_colors[i+1], alpha=0.3, label='10-90 percentile')
        
    #     ax.set_ylabel('Cloud Thickness (km)')
    #     ax.set_xlabel('Time')
    #     ax.set_title(var.replace('_', '-').capitalize())
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    #     ax.xaxis.set_minor_locator(mdates.MonthLocator())
    #     ax.grid()
    # fig.savefig(PATH_FIG + 'new_method_cloud_thickness.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # # -----------------------------------------------------------------------
    # # Cloud Top Height
    # # -----------------------------------------------------------------------

    # fig = plt.figure(figsize=(15, 10))
    # gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4)

    # for i, var in enumerate(list_var_names[:-1]):
    #     ax = fig.add_subplot(gs[i//2, i%2])
    #     mask_cloud_type = single_layer_cloud_type[var].astype(bool)
    #     freq_str = '10D'
    #     cloud_top_mean = (ds_cloud_prop['cloud_top'].sel(time=mask_cloud_type).resample(time=freq_str).mean())/1000.
    #     cloud_top_50th = (ds_cloud_prop['cloud_top'].sel(time=mask_cloud_type).resample(time=freq_str).median())/1000.
    #     cloud_top_10th = (ds_cloud_prop['cloud_top'].sel(time=mask_cloud_type).resample(time=freq_str).quantile(0.1))/1000.
    #     cloud_top_90th = (ds_cloud_prop['cloud_top'].sel(time=mask_cloud_type).resample(time=freq_str).quantile(0.9))/1000.

    #     ax.plot(cloud_top_50th.time, cloud_top_50th, '-ok', markersize=4)
    #     ax.fill_between(cloud_top_50th.time, cloud_top_10th, cloud_top_90th, color=list_cloud_colors[i+1], alpha=0.3, label='10-90 percentile')
        
    #     ax.set_ylabel('Cloud Top Height (km)')
    #     ax.set_xlabel('Time')
    #     ax.set_title(var.replace('_', '-').capitalize())
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    #     ax.xaxis.set_minor_locator(mdates.MonthLocator())
    #     ax.grid()
    # fig.savefig(PATH_FIG + 'new_method_cloud_top.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # # -----------------------------------------------------------------------
    # # End
    # # -----------------------------------------------------------------------

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

        base_cloud_type = (ds_cloud_prop['cloud_base'].sel(time=mask_cloud_type) - GRANADA_ALTITUDE)/1000
        # base_cloud_type = ds_cloud_prop['cloud_base'].where(mask_cloud_type)/1000.
        df_cloud_type   = base_cloud_type.to_dataframe().reset_index().drop(columns=['time'])

        # print(f"For the cloud type: {var}, count by season are as follows: {df_cloud_type.groupby('season').size()}")
        
        # sns.violinplot(data = df_cloud_type, 
        #                 x='season',
        #                 y='cloud_base', 
        #                 ax=ax, 
        #                 color=list_cloud_colors[i+1], 
        #                 density_norm='count', 
        #                 inner="quart")
        sns.violinplot(x=base_cloud_type.season.values, 
                    y=base_cloud_type.values, ax=ax, 
                    color=list_cloud_colors[i+1], 
                    scale='count', inner="quart",
                    orient='v',
                    alpha=0.4,
                    linewidth=1.5)   
        # sns.violinplot(x=base_cloud_type.time.dt.season.values, y=base_cloud_type.values, ax=ax, color=list_cloud_colors[i+1])
        
        ax.set_ylabel('Cloud Base Height (m)')
        ax.set_xlabel('Month')
        ax.set_title(var.replace('_', '-').capitalize())
        # ax.set_xticks(range(1, 13))
        # ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.grid()
    fig.savefig(PATH_FIG + 'new_method_cloud_base_height_violin_season.png', dpi=300, bbox_inches='tight')
    plt.show()



    list_var_names = list(ds_cloud_type.data_vars)
    split_pairs_var_names = [list_var_names[i:i+2] for i in range(0, len(list_var_names)-2, 2)]

    dic = {}
    dic["Ice"] = ["Ice", "Ice-Precipitable"]
    dic["Liquid"] = ["Liquid", "Liquid-Precipitable"]
    dic["Mixed"] = ["Mixed-Phase", "Mixed-Phase-Precipitable"]

    dataframes = []
    for pair_var_names in split_pairs_var_names:
        pair_single_layer   = ds_cloud_type[pair_var_names].where(ds_cloud_occurence['single_layer'], other=0)
        mask_pair_var       = pair_single_layer[pair_var_names[0]].astype(bool) | pair_single_layer[pair_var_names[1]].astype(bool)
        sliced_cloud_type  = pair_single_layer.sel(time=mask_pair_var.values)
        
        sliced_cloud_type['cloud_type'] = xr.DataArray(np.where(sliced_cloud_type[pair_var_names[0]] == 1,
                                                    pair_var_names[0],
                                                    pair_var_names[1]), dims= "time")

        single_layer_cloud_prop = ds_cloud_prop.where(ds_cloud_occurence['single_layer'], other=np.nan)
        sliced_cloud_prop = single_layer_cloud_prop.sel(time=mask_pair_var.values)/1000
        df = sliced_cloud_type.merge(sliced_cloud_prop).to_dataframe().reset_index().drop(columns=['time', 'month', 'year', pair_var_names[0], pair_var_names[1]])
        df['cloud_base'] = df['cloud_base'] - GRANADA_ALTITUDE/1000
        df['cloud_top']  = df['cloud_top'] - GRANADA_ALTITUDE/1000
        dataframes.append(df)
    
    ys = ['cloud_base', 'cloud_top', 'cloud_thickness']
    colors_violin = ['Blues', 'Greys', 'YlOrBr']
    fig = plt.figure(figsize=(22, 18))
    gs  = fig.add_gridspec(len(dataframes), len(ys), height_ratios=[1]*len(dataframes), hspace=0.15, wspace=0.2)
    for i, df in enumerate(dataframes):
        for j, y in enumerate(ys):
            # Make a slipted violin plot for each dataframe
            ax = fig.add_subplot(gs[j, i])
            sns.violinplot(data = df,
                            x='season',
                            y=y,
                            hue='cloud_type',   
                            ax=ax,
                            split=True,
                            palette=colors_violin[i],
                            inner="quart",
                            order=['spring', 'summer', 'fall', 'winter'],
                            scale='count',
                            gap=.4,
                            )  # Specify the order of x-axis categories
            ax.set_ylabel(f'{y}')
            ax.set_xlabel('Seasons')
            ax.grid()
            if y == 'cloud_thickness':
                ax.set_ylim([0, 10])
            else:
                ax.set_ylim([0, 13])
            quartiles = df.groupby(['season', 'cloud_type'])[y].quantile([0.25, 0.5, 0.75]).unstack()
            # print(f"Season: {df.season.unique()}, {y}: {quartiles}")
    fig.savefig(PATH_FIG + 'new_method_cloud_prop_violin_season.png', dpi=300, bbox_inches='tight')
    plt.show()

    # # Create a dataframe for liquid and liquid preicipitable
    # df_cloud_type_single_layer = ds_cloud_type_single_layer.to_dataframe().reset_index().melt(id_vars=['time'],
    #                                                                                             var_name='cloud_type',
    #                                                                                             value_name='cloud_occurence'
    #                                                                                             ).drop(columns=['month', 'season', 'year'])
    # df_cloud_type_single_layer = ds_cloud_type_single_layer.to_dataframe().reset_index().melt(id_vars=['time'], 
    #                                                                                           var_name='cloud_type', 
    #                                                                                           value_name='cloud_occurence'
    #                                                                                           ).drop(columns=['month', 'season', 'year'])
    # add a column for the season
