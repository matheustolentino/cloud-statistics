
#-------------------------------------------------------------------------------------------------------
# import classes
#-------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.ticker as ticker
import pandas as pd
import os
import xarray as xr
from scipy.stats import gaussian_kde
import scipy.stats as stats
import locale
import seaborn as sns
from pdb import set_trace
from typing import Dict, Union, Any, List
import dask
import dask.config
import time as time_module
from dateutil.relativedelta import relativedelta
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, griddata
from itertools import chain
import concurrent.futures
# import seaborn as sns

# Set the option to split large chunks
dask.config.set(**{'array.slicing.split_large_chunks': True})
#dask.config.set(num_workers=4)

# Define the fontsize
fontsize = 14

# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize

plt.ion()
plt.close('all')
locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
# Set up Seaborn for better visualization
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.5})
# Customize tick parameters to have black markers only at the axis
# ---------------------------------------------------------------------------------------------
# Paths to the data files and save figures
# ---------------------------------------------------------------------------------------------
PATH_FIG    = '../figures/'
root_folder = '../../../processed_data/'
# ---------------------------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------------------------
SEASONS = {
        'summer': (6, 8),   # from 1st June to 31st August
        'fall': (9, 11),    # from 1st September to 30th November
        'winter': (12,2),   # from 1st December to 28th February
        'spring': (3, 5)    # from 1st March to 31st May
    }
# ---------------------------------------------------------------------------------------------
def create_data_availability_plot(reindexed_variable: xr.DataArray, freq_str: str, figname: str):
    """
    Create a data availability plot using Dask arrays.

    Parameters:
        reindexed_variable (xr.DataArray): xarray DataArray to plot.
        freq_str (str): Frequency string for resampling (e.g., 'D' for daily, 'H' for hourly).

    Returns:
        None
    """
    data_availability = [(reindexed_variable > 0).resample(time=freq_str).mean(),
                         (reindexed_variable == 0).resample(time=freq_str).mean(),
                         (np.isnan(reindexed_variable)).resample(time=freq_str).mean()]
    labels = ["Hydrometeor", "Clear Sky", "Missing"]
    colors = ["#28fc21", "#07a8e3", "#ffffff"]
    bot = np.zeros(data_availability[0].shape)
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the size of the plot for better visibility
    for i, data in enumerate(data_availability):
        freq = data.compute() # Compute the Dask array
        # Calculate the width of the bars based on the frequency of the data
        if i==0:
            w = (freq.time.max().values - freq.time.min().values) / len(freq.time)
        p = ax.bar(freq.time, 100*freq.values,
                label=f'{labels[i]}',
                bottom=bot,
                color=colors[i],
                edgecolor="black",
                width=w)  # Set the width of the bars
        bot += 100*freq.values

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.set_xlabel("Month/Year")
    ax.set_ylabel("Frequency [%]")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(data_availability), frameon=False)
    ax.set_xticks(freq.time[::3])
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(figname, dpi=300)
    plt.show()

def create_hydromet_bar_plot(freq_list: xr.DataArray, freq_str: str,
                             labels: list, colors: list, figname: str):
    """
    Create a data availability plot using Dask arrays.

    Parameters:
        reindexed_variable (xr.DataArray): xarray DataArray to plot.
        freq_str (str): Frequency string for resampling (e.g., 'D' for daily, 'H' for hourly).

    Returns:
        None
    """
    bot = np.zeros(freq_list[0].shape)
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the size of the plot for better visibility
    for i, data in enumerate(freq_list):
        freq = data.compute() # Compute the Dask array
        # Calculate the width of the bars based on the frequency of the data
        if i==0:
            w = (freq.time.max().values - freq.time.min().values) / len(freq.time)
        p = ax.bar(freq.time, 100*freq.values,
                label=f'{labels[i]}',
                bottom=bot,
                color=colors[i],
                edgecolor="black",
                width=w)  # Set the width of the bars
        bot += 100*freq.values

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency [%]")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(freq_list), frameon=False)
    ax.set_xticks(freq.time[::3])
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(figname, dpi=300)
    plt.show()

def create_hydromet_box_plot(reindexed_variable: xr.DataArray, figname: str = None):
    """
    Create a box plot for each month using Dask arrays.

    Parameters:
        reindexed_variable (xr.DataArray): xarray DataArray to plot.
        figname (str): File name for saving the figure.

    Returns:
        None
    """

    df = reindexed_variable.to_dataframe()
    df['Total'] = df['Total']*100 # Convert DataArray to DataFrame
    df['data_available'] = df['Total'].notnull()
    df['period'] = df.index.to_period('M')
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the size of the plot for better visibility
    sns.boxplot(x='period',y='Total',data=df,ax=ax,
                fliersize=2,
                medianprops={"color": "red"},
                whiskerprops={'color': 'black'},
                capprops={'color': 'black'},
                boxprops={'edgecolor': 'black'})  # Change the color of the whiskers to blue
    # Add line plot for df['data_available']
    # data_av = df['data_available'].resample("M").mean()*100
    # data_av['period'] = df['period'] = df.index.to_period('M')
    # sns.lineplot(x = df['period'].unique(), y=df['data_available'].resample("M").mean()*100, color='blue', linewidth=2.5, label='Available Data')
    # set_trace()
    # sns.lineplot(x='period',y='data_available', data=data_av,
    #              ax=ax, color='blue', linewidth=2.5, label='Available Data')
    ax.set_ylabel("Frequency [%]")
    ax.set_xlabel("Year-Month")
    ax.set_title("Profile Percentage - Box Plot")

    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.set_xticks(ax.get_xticks()[::3])
    ax.grid(True)
    ax.set_ylim([0, 110])
    plt.xticks(rotation=45)
    # ax.set_yscale('log')
    plt.tight_layout()
    fig.savefig(figname, dpi=300)
    plt.show()

def plot_cloud_frequency(reindexed_variable: xr.DataArray, freq_str: str, figname: str):
    """
    Create a data availability plot using Dask arrays.

    Parameters:
        reindexed_variable (xr.DataArray): xarray DataArray to plot.
        freq_str (str): Frequency string for resampling (e.g., 'D' for daily, 'H' for hourly).

    Returns:
        None
    """
    # includinf new variable to reindexed_variable:
    # including new variable to reindexed_variable:
    colors = ["#f953d2", "#53d2f9", "#c9b337", "#2521b6", "#fc564f", "#49eb34","#ababab","#ffffff"]

    cloud_frequency = (reindexed_variable == 1).resample(time=freq_str).mean()
    # df = reindexed_variable.to_dataframe()
    # mask_no_single_layer = df.isna().sum(axis=1) == df.columns.size # mask for no single layer and missing data
    # mask_no_single_layer = reindexed_variable.isnull().sum(axis=1) == df.columns.size
    #cloud_frequency['missing'] = reindexed_variable.multilayer.isnull().resample(time=freq_str).mean()
    bar_width = (cloud_frequency.time.max().values - cloud_frequency.time.min().values) / cloud_frequency.time.shape[0]
    # bar_width = pd.Timedelta(days=30)
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the size of the plot for better visibility
    for i, var_name in enumerate(cloud_frequency.data_vars):
        freq = cloud_frequency[var_name]
        if i==0:
            bot = np.zeros(freq.time.shape[0])
        p = ax.bar(freq.time, 100*freq.values,
                label=var_name.replace('_', '-').title(),
                bottom=bot,
                color=colors[i],
                edgecolor="black",
                width=bar_width)  # Set the width of the bars
        bot += 100*freq.values

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.set_xlabel("Month/Year")
    ax.set_ylabel("Frequency [%]")
    ax.set_ylim([0, 100])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(cloud_frequency.data_vars)/2, frameon=False)
    ax.set_xticks(freq.time[::3])
    plt.xticks(rotation=45)

    plt.tight_layout()
    fig.savefig(figname, dpi=300)
    plt.show()

def plot_cloud_frequency2(reindexed_variable: xr.DataArray, freq_str: str, figname: str):
    """
    Create a data availability plot using Dask arrays.

    Parameters:
        reindexed_variable (xr.DataArray): xarray DataArray to plot.
        freq_str (str): Frequency string for resampling (e.g., 'D' for daily, 'H' for hourly).
        figname (str): Filepath to save the figure.

    Returns:
        None
    """
    # includinf new variable to reindexed_variable:
    # including new variable to reindexed_variable:
    colors = ["#f953d2", "#53d2f9", "#c9b337", "#2521b6", "#fc564f", "#49eb34","#ababab","#ffffff"]

    cloud_frequency = (reindexed_variable == 1).resample(time=freq_str).mean()

    years = np.unique(cloud_frequency.time.dt.year)  # Get unique years

    fig, axs = plt.subplots(len(years), 1, figsize=(12, 8*len(years)))  # Create subplots for each year

    for i, year in enumerate(years):
        ax = axs[i] if len(years) > 1 else axs  # Use the same axis if there's only one year
        year_data = cloud_frequency.sel(time=cloud_frequency.time.dt.year == year)  # Select data for the current year

        bar_width = (year_data.time.max().values - year_data.time.min().values) / year_data.time.shape[0]
        bot = np.zeros(year_data.time.shape[0])
        for j, var_name in enumerate(year_data.data_vars):
            freq = year_data[var_name]
            p = ax.bar(freq.time, 100*freq.values,
                    label=var_name.replace('_', '-').title(),
                    bottom=bot,
                    color=colors[j],
                    edgecolor="black",
                    width=bar_width)
            bot += 100*freq.values

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.set_xlabel("Month/Year")
        ax.set_ylabel("Frequency [%]")
        ax.set_ylim([0, 100])
        ax.set_xticks(year_data.time[::1])

        if i == 0:  # Add legend only for the first subplot
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(year_data.data_vars)/2, frameon=False)

        # Remove spines, save the figure, and show the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        plt.xticks(rotation=45)

    plt.tight_layout()
    fig.savefig(figname, dpi=300)
    plt.show()

def plot_cloud_frequency3(reindexed_variable: xr.DataArray, freq_str: str, figname: str):
    """
    Create a data availability plot using Dask arrays.

    Parameters:
        reindexed_variable (xr.DataArray): xarray DataArray to plot.
        freq_str (str): Frequency string for resampling (e.g., 'D' for daily, 'H' for hourly).
        figname (str): Filepath to save the figure.

    Returns:
        None
    """
    # includinf new variable to reindexed_variable:
    # including new variable to reindexed_variable:
    colors = ["#f953d2", "#53d2f9", "#c9b337", "#2521b6", "#fc564f", "#49eb34","#ababab","#ffffff"]

    cloud_frequency = (reindexed_variable == 1).resample(time=freq_str).mean()

    years = np.unique(cloud_frequency.time.dt.year)  # Get unique years

    for year in years:
        year_data = cloud_frequency.sel(time=cloud_frequency.time.dt.year == year)  # Select data for the current year

        fig, ax = plt.subplots(figsize=(14, 9))  # Create a new figure and axis for each year

        bar_width = (year_data.time.max().values - year_data.time.min().values) / year_data.time.shape[0]
        bot = np.zeros(year_data.time.shape[0])
        for j, var_name in enumerate(year_data.data_vars):
            freq = year_data[var_name]
            p = ax.bar(freq.time, 100*freq.values,
                    label=var_name.replace('_', '-').title(),
                    bottom=bot,
                    color=colors[j],
                    edgecolor="black",
                    width=bar_width)
            bot += 100*freq.values
            # print(f"Frequency of {var_name} in {year}: {freq.values*100:.2f}%")

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.set_xlabel("Month/Year")
        ax.set_ylabel("Frequency [%]")
        ax.set_xticks(year_data.time[::1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(year_data.data_vars)/2, frameon=False)

        # Remove spines, save the figure, and show the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        plt.xticks(rotation=30)
        fig.savefig(f"{figname}_{year}.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

def plot_cloud_frequency4(reindexed_variable: xr.DataArray, freq_str: str, figname: str):
    """
    Create a data availability plot using Dask arrays.

    Parameters:
        reindexed_variable (xr.DataArray): xarray DataArray to plot.
        freq_str (str): Frequency string for resampling (e.g., 'D' for daily, 'H' for hourly).

    Returns:
        None
    """
    # includinf new variable to reindexed_variable:
    # including new variable to reindexed_variable:
    colors = ["#f953d2", "#53d2f9", "#c9b337", "#2521b6", "#fc564f", "#49eb34","#ababab","#ffffff"]

    cloud_frequency = (reindexed_variable == 1).groupby('month').mean()
    # df = reindexed_variable.to_dataframe()
    # mask_no_single_layer = df.isna().sum(axis=1) == df.columns.size # mask for no single layer and missing data
    # mask_no_single_layer = reindexed_variable.isnull().sum(axis=1) == df.columns.size
    #cloud_frequency['missing'] = reindexed_variable.multilayer.isnull().resample(time=freq_str).mean()
    # bar_width = (cloud_frequency.time.max().values - cloud_frequency.time.min().values) / cloud_frequency.time.shape[0]
    bar_width = .95
    # bar_width = pd.Timedelta(days=30)
    fig, ax = plt.subplots(figsize=(10, 8))  # Increase the size of the plot for better visibility
    for i, var_name in enumerate(cloud_frequency.data_vars):
        freq = cloud_frequency[var_name].compute()
        if i==0:
            bot = np.zeros(freq.month.shape[0])
        p = ax.bar(freq.month, 100*freq.values,
                label=var_name.replace('_', '-').title(),
                bottom=bot,
                color=colors[i],
                edgecolor="black",
                width=bar_width)  # Set the width of the bars
        bot += 100*freq.values
    # Adding name of the months values
    ax.set_xticks(cloud_frequency.month)
    ax.set_xlabel("Month")
    ax.set_ylabel("Frequency [%]")
    ax.set_ylim([0, 100])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(cloud_frequency.data_vars)/2, frameon=False)

    plt.tight_layout()
    fig.savefig(figname, dpi=300)
    plt.show()

def plot_2d_and_vertical_frequency(dataset1, month_mask, freq_str: str, path: str):
    # Get variable names excluding the first and last two from dataset1
    variable_names = list(dataset1.data_vars)
    # Iterate through the variables
    for var_name in variable_names:

        # Average variable monthly and remove months with
        variable = 100*dataset1[var_name].resample(time=freq_str).mean().compute()
        variable = variable.sel(time=month_mask)

        # reindexing the variable to fill the missing months with NaNs
        new_time_index = pd.date_range(start=variable.time.values.min(), end=variable.time.values.max(), freq="M")
        variable = variable.reindex(time=new_time_index)

        # Create the figure and GridSpec layout
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(1, 3, width_ratios=[3.5, 0.09, 1])  # Four columns: heatmap, spacer, line plot, colorbar

        # Add heatmap on the left
        ax_heatmap = plt.subplot(gs[0])
        p1 = ax_heatmap.pcolormesh(variable.time, variable.range_bins/1e3, variable.values, shading='nearest', cmap='jet')
        ax_heatmap.set_title(f"{var_name}")
        ax_heatmap.set_xlabel("Day/Month/Year")
        ax_heatmap.set_ylabel("Range [km]")
        ax_heatmap.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
        ax_heatmap.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better visibility
        ax_heatmap.set_ylim([0, 10])

        # Add colorbar after the heatmap
        ax_colorbar = plt.subplot(gs[1])
        cbar = fig.colorbar(p1, cax=ax_colorbar, label='Frequency [%]')
        cbar.ax.yaxis.set_ticks_position('left')  # Move the colorbar tick labels to the left side
        cbar.ax.yaxis.set_label_position('left')  # Move the colorbar label to the left side

        # Plot from dataset2 in the third column
        ax_lineplot = plt.subplot(gs[2], sharey=ax_heatmap)
        # variable2 = mask_hydromet.mean(dim='time')
        variable2 = 100*dataset1[var_name].mean(dim='time')
        ax_lineplot.plot(variable2.values, variable2.range_bins/1e3, label=var_name, color=np.random.rand(3), marker="o")
        # ax_lineplot.set_title(f"Line Plot for {var_name}")
        ax_lineplot.set_xlabel("Frequency [%]")
        ax_lineplot.grid(True)
        ax_lineplot.set_ylim([0, 10])

        # Adjust vertical spacing between subplots
        # plt.subplots_adjust(wspace=horizontal_space)

        # Adjust layout for the subplots
        plt.tight_layout()  # Adjust the left subplot to occupy most of the figure space
        fig.savefig(f"{path}{var_name}_2d_histogram_hydrometeors.png", dpi=300)
        plt.show()

def plot_2d_and_vertical_frequency_2(dataset, freq_str: str, path: str):
    # Get variable names from the first key in the dataset dictionary
    variable_names = list(dataset[list(dataset.keys())[0]].data_vars)

    # Iterate through the variable names
    for var_name in variable_names:
        # Create the figure and GridSpec layout
        fig = plt.figure(figsize=(15, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[3.5, 0.2, 1])  # Four columns: heatmap, spacer, line plot, colorbar

        # Iterate through each key in the dataset dictionary
        for key in dataset.keys():
            # Get the variable for the current key and variable name
            variable = dataset[key][var_name] == 1

            variable1 = variable.resample(time=freq_str).mean(dim='time')
            # Add heatmap on the left
            ax_heatmap = plt.subplot(gs[0])
            p1 = ax_heatmap.pcolormesh(variable1.time, variable1.range/1e3, variable1.values.T, shading='auto', cmap='cividis')
            ax_heatmap.set_title(f"2D Frequency Plot for {var_name}")
            ax_heatmap.set_xlabel("Time")
            ax_heatmap.set_ylabel("Range [km]")
            ax_heatmap.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
            ax_heatmap.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better visibility

            # Add colorbar after the heatmap
            ax_colorbar = plt.subplot(gs[1])
            cbar = fig.colorbar(p1, cax=ax_colorbar, label='Frequency')
            cbar.ax.yaxis.set_ticks_position('left')  # Move the colorbar tick labels to the left side
            cbar.ax.yaxis.set_label_position('left')  # Move the colorbar label to the left side

            # Plot from dataset2 in the third column
            ax_lineplot = plt.subplot(gs[2], sharey=ax_heatmap)
            variable2 = variable.mean(dim='time')
            ax_lineplot.plot(variable2.values, variable2.range/1e3, label=key, color=np.random.rand(3), marker="o")
            ax_lineplot.set_title(f"Line Plot for {var_name}")
            ax_lineplot.set_xlabel("Frequency [%]")
            ax_lineplot.grid(True)
            plt.legend()

        # Adjust vertical spacing between subplots
        # plt.subplots_adjust(wspace=horizontal_space)

        # Adjust layout for the subplots
        plt.tight_layout()  # Adjust the left subplot to occupy most of the figure space
        fig.savefig(f"{path}{var_name}_2d_histogram_hydrometeors.png", dpi=300)
        plt.show()

def plot_2d_and_vertical_frequency2(dataset1, dataset2):
    # Get variable names excluding the first and last two from dataset1
    variable_names = list(dataset1.data_vars)
    # Iterate through the variables
    for var_name in variable_names:
        variable = dataset1[var_name]

        # Create the figure and GridSpec layout
        fig = plt.figure(figsize=(15, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[3.5, 0.09, 1])  # Four columns: heatmap, spacer, line plot, colorbar

        # Add heatmap on the left
        ax_heatmap = plt.subplot(gs[0])
        p1 = ax_heatmap.pcolormesh(variable.time, variable.range/1e3, variable.values.T, shading='auto', cmap='cividis')
        ax_heatmap.set_title(f"2D Frequency Plot for {var_name}")
        ax_heatmap.set_xlabel("Time")
        ax_heatmap.set_ylabel("Range [km]")
        ax_heatmap.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
        ax_heatmap.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better visibility

        # Add colorbar after the heatmap
        ax_colorbar = plt.subplot(gs[1])
        cbar = fig.colorbar(p1, cax=ax_colorbar, label='Frequency')
        cbar.ax.yaxis.set_ticks_position('left')  # Move the colorbar tick labels to the left side
        cbar.ax.yaxis.set_label_position('left')  # Move the colorbar label to the left side

        # Plot from dataset2 in the third column
        ax_lineplot = plt.subplot(gs[2], sharey=ax_heatmap)
        variable2 = dataset2[var_name].mean(dim='time')
        ax_lineplot.plot(variable2.values, variable2.range/1e3, label=var_name, color=np.random.rand(3), marker="o")
        ax_lineplot.set_title(f"Line Plot for {var_name}")
        ax_lineplot.set_xlabel("Frequency [%]")
        ax_lineplot.grid(True)

        # Adjust vertical spacing between subplots
        # plt.subplots_adjust(wspace=horizontal_space)

        # Adjust layout for the subplots
        plt.tight_layout()  # Adjust the left subplot to occupy most of the figure space
        plt.show()

def plot_time_evolution_frequency(dataset, figname: str):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed

    # Iterate through variable names and plot each variable using the axis.plot method
    markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'H', 'x']
    for i, var_name in enumerate(dataset.data_vars):
        ax.plot(dataset["time"], dataset[var_name], label=var_name, marker=markers[i % len(markers)])

    # Add labels, title, and legend
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency [%]")
    ax.set_title("Variable Plots")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set x-axis tick locator to show ticks by month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))  # Set x-axis tick formatter to show month and year
    ax.set_xticks(dataset["time"][::3])  # Set x-axis ticks at every 3 months
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    # Show the plot
    fig.savefig(figname, dpi=300)
    plt.show()

def plot_cfads2(dataset, bin_edges, nbins=50, figname='cfads.png'):
    """
    Plot 2D histograms with frequency-based contour plots for variables in the dataset.

    Parameters:
    - dataset (dask.array.Array): The chunked dataset containing the variables.
    """

    # Extract variable names from the dataset
    var_names = dataset.data_vars.keys()
    range_values = dataset.range/1e3  # Assuming range is constant across chunks

    # Define a Seaborn color palette for the colormap with inverted colors
    cmap = sns.color_palette("turbo", as_cmap=True)
    # Create subplots for each variable using gridspec
    fig = plt.figure(figsize=(12, 6))
    num_vars = len(var_names)
    gs = gridspec.GridSpec(1, 2 * num_vars, width_ratios=[1, 0.02] * num_vars)  # Adjust the number of columns as needed
    for i, var_name in enumerate(var_names):
        ax = plt.subplot(gs[2*i])
        var_values = dataset[var_name].data

        # Create a 2D histogram for each chunk
        hist = np.zeros((nbins-1,nbins-1))
        #set_trace()
        for chunk in var_values.blocks:
            chunked_values = chunk.compute().ravel()
            nan_mask       = np.isnan(chunked_values)
            flattened_var  = chunked_values[~nan_mask]
            flattened_range_var = np.tile(range_values, chunk.shape[0])[~nan_mask]
            chunk_hist, _, _= np.histogram2d(flattened_var, flattened_range_var, bins=bin_edges[var_name])
            # Update the value using the current step
            hist += chunk_hist

        bin_area = np.outer(np.diff(bin_edges[var_name][0]), np.diff(bin_edges[var_name][1]))
        hist = hist / (np.sum(hist) * bin_area)

        # Calculate bin centers for the contour plot
        x_centers = (bin_edges[var_name][0][:-1] + bin_edges[var_name][0][1:]) / 2
        y_centers = (bin_edges[var_name][1][:-1] + bin_edges[var_name][1][1:]) / 2

        # Define levels for contour plot
        colorbar_max = 0.5 * hist.max()
        levels = np.linspace(0, colorbar_max, 17)

        # Contour plot for the current variable
        contour = ax.contourf(x_centers, y_centers, hist.T, levels=levels, cmap=cmap, extend='both')
        ax.set_xlabel(var_name)
        ax.set_ylabel('Height [km]')
        ax.set_title(f'2D Histogram of {var_name} vs Height')

        # Increase resolution of x-axis
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

        # Add grid
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        # Create a colorbar for the current variable
        cbar_ax = plt.subplot(gs[2 * i + 1])
        cbar = plt.colorbar(contour, cax=cbar_ax, label='Density', extend='both')
        cbar.formatter.set_useMathText(True)
        cbar_ticks = np.linspace(0, colorbar_max, 10)  # Adjust the number of ticks as needed
        cbar.set_ticks(cbar_ticks)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
    plt.tight_layout()
    fig.savefig(figname, dpi=300)
    plt.show()

def plot_cfads(dataset):
    """
    Plot 2D histograms with frequency-based contour plots for variables in the dataset.

    Parameters:
    - dataset (dask.array.Array): The chunked dataset containing the variables.
    """

    # Extract variable names from the dataset
    var_names = dataset.data_vars.keys()
    range_values = dataset.range  # Assuming range is constant across chunks

    # Define a Seaborn color palette for the colormap with inverted colors
    cmap = sns.color_palette("turbo", as_cmap=True)
    # Create subplots for each variable using gridspec
    fig = plt.figure(figsize=(12, 6))
    num_vars = len(var_names)
    gs = gridspec.GridSpec(1, 2 * num_vars, width_ratios=[1, 0.02] * num_vars)  # Adjust the number of columns as needed
    for i, var_name in enumerate(var_names):
        ax = plt.subplot(gs[2*i])
        var_values = dataset[var_name]

        # Get the chunk size of the variable
        chunk_size = var_values.chunks
        # Create a 2D histogram for each chunk
        histograms = []
        # Initialize a starting value
        start_value = 0
        for step in chunk_size[0]:
            chunked_values = var_values[start_value:start_value + step, :]
            set_trace()
            print(chunked_values.dims['time' ])
            flattened_var = chunked_values.values[~np.isnan(chunked_values.values)]
            flattened_range_var = np.tile(range_values, step)[~np.isnan(chunked_values.values.ravel())]
            hist, x_edges, y_edges = np.histogram2d(flattened_var, flattened_range_var, bins=(50, 50))
            histograms.append(hist)
            # Update the value using the current step
            start_value += step
        hist = np.sum(histograms, axis=0)

        # Calculate bin centers for the contour plot
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2 / 1e3  # Convert to km

        # Define levels for contour plot
        colorbar_max = 0.2 * hist.max()
        levels = np.linspace(0, colorbar_max, 17)

        # Contour plot for the current variable
        contour = ax.contourf(x_centers, y_centers, hist.T, levels=levels, cmap=cmap, extend='both')
        ax.set_xlabel(var_name)
        ax.set_ylabel('Height [km]')
        ax.set_title(f'2D Histogram of {var_name} vs Height')

        # Increase resolution of x-axis
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

        # Add grid
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        # Calculate colorbar limits
        colorbar_max = 0.5 * hist.max()
        levels = np.linspace(0, colorbar_max, 17)
        # Create a colorbar for the current variable
        cbar_ax = plt.subplot(gs[2 * i + 1])
        cbar = plt.colorbar(contour, cax=cbar_ax, label='Density', extend='both')
        cbar.formatter.set_useMathText(True)
        cbar_ticks = np.linspace(0, colorbar_max, 10)  # Adjust the number of ticks as needed
        cbar.set_ticks(cbar_ticks)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
    plt.tight_layout()
    plt.show()

def plot_histograms_with_profiles(datasets: list, bin_width: float,
                                  labels: list, x_label: str, y_label: str,
                                  xticks_resolution: float = 1.0,
                                  xlim: list = None, ylim: list = None,  figname = None) -> None:
    """
    Plot histograms for multiple xarray DataArrays, with labeled bars indicating the number of profiles.

    Parameters:
        datasets (list of xr.DataArray): List of xarray DataArray objects.
        bin_width (float): Width of histogram bins.
        labels (list of str): List of labels for each dataset in the legend.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.

    Returns:
        None: Displays the histogram plot.
    """
    # Calculate histograms and shared bin edges
    min_value = min(np.nanmin(data) for data in datasets)
    max_value = max(np.nanmax(data) for data in datasets)
    bin_edges = np.arange(min_value, max_value + bin_width, bin_width)
    # Calculate histograms for each dataset
    histograms = [np.histogram(data, bins=bin_edges)[0] for data in datasets]

    # Get the number of profiles for each dataset
    num_profiles = [data.sizes["time"] for data in datasets]

    # Plot histograms for each dataset
    fig, axes = plt.subplots(figsize=(10, 6))

    bin_widths = bin_edges[1] - bin_edges[0]
    norm_histograms = [100 * (hist / num_profiles[i]) for i, hist in enumerate(histograms)]

    for i, norm_hist in enumerate(norm_histograms):
        label = f"{labels[i]} ({num_profiles[i]} profiles)"
        axes.hist(bin_edges[:-1], bins=bin_edges,
                   weights=norm_hist,
                   histtype='step',
                   label=label,
                   linewidth=2.0)

    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_yscale('log')
    axes.legend()
    axes.grid()
    # Set x-axis ticks to bin_edges with specified resolution
    # x_ticks = np.arange(min(bin_edges), max(bin_edges) + xticks_resolution, xticks_resolution)
    if xlim:
        x_ticks = np.arange(min(bin_edges), max(xlim) + xticks_resolution, xticks_resolution)
        axes.set_xlim(xlim)
    else:
        x_ticks = np.arange(min(bin_edges), max(bin_edges) + xticks_resolution, xticks_resolution)
    axes.set_ylim(ylim)
    axes.set_xticks(x_ticks)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(figname, dpi=300)
    plt.show()

# def create_frequency_cloud_layers_plot(frequency_cloud_layers):
#     # List of markers for the plot
#     markers = ["o", "*", "s", "<", "X"]

#     # Create a subplot
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Loop over each variable and create a plot
#     for i, variable in enumerate(frequency_cloud_layers):
#         p = ax.plot(frequency_cloud_layers['time'],
#                     frequency_cloud_layers[variable],
#                     label=variable,
#                     color=np.random.rand(3),  # Generate random color
#                     marker=markers[i])

#     ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set x-axis tick locator to show ticks by month
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))  # Set x-axis tick formatter to show month and year
#     ax.set_xticks(dataset["time"][::3])  # Set x-axis ticks at every 3 months
#     plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
#     # Add a legend
#     ax.legend()
#     plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

#     # Display the plot
#     plt.show()

def interpolate_2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of gridded 2d data.

    Args:
        x: 1-D array.
        y: 1-D array.
        z: 2-D array at points (x, y).
        x_new: 1-D array.
        y_new: 1-D array.

    Returns:
        Interpolated data.

    Notes:
        Does not work with nans. Ignores mask of masked data. Does not extrapolate.

    """
    fun = RectBivariateSpline(x, y, z, kx=1, ky=1)
    return fun(x_new, y_new)


def interpolate_2d_nearest(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ma.MaskedArray:
    """2D nearest neighbor interpolation preserving mask.

    Args:
        x: 1D array, x-coordinates.
        y: 1D array, y-coordinates.
        z: 2D masked array, data values.
        x_new: 1D array, new x-coordinates.
        y_new: 1D array, new y-coordinates.

    Returns:
        Interpolated 2D masked array.

    Notes:
        Points outside the original range will be interpolated but masked.

    """
    data = np.ma.copy(z)
    fun = RegularGridInterpolator(
        (x, y),
        data,
        method="nearest",
        bounds_error=False,
        fill_value=np.ma.masked,
    )
    xx, yy = np.meshgrid(x_new, y_new)
    return fun((xx, yy)).T

def interpolate_2d_mask(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ma.MaskedArray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ma.MaskedArray:
    """2D linear interpolation preserving the mask.

    Args:
        x: 1D array, x-coordinates.
        y: 1D array, y-coordinates.
        z: 2D masked array, data values.
        x_new: 1D array, new x-coordinates.
        y_new: 1D array, new y-coordinates.

    Returns:
        Interpolated 2D masked array.

    Notes:
        Points outside the original range will be nans (and masked). Uses linear
        interpolation. Input data may contain nan-values.

    """
    z = np.ma.array(np.ma.masked_invalid(z, copy=True))  # ma.array() to avoid pylint nag
    # Interpolate ignoring masked values:
    valid_points = np.logical_not(z.mask)  # ~z.mask causes pylint nag
    xx, yy = np.meshgrid(y, x)
    x_valid = xx[valid_points]
    y_valid = yy[valid_points]
    z_valid = z[valid_points]
    xx_new, yy_new = np.meshgrid(y_new, x_new)
    data = griddata(
        (x_valid, y_valid), z_valid.ravel(), (xx_new, yy_new), method="linear"
    )
    # Preserve mask:
    mask_fun = RectBivariateSpline(x, y, z.mask[:], kx=1, ky=1)
    mask = mask_fun(x_new, y_new)
    mask[mask < 0.5] = 0
    masked_array = np.ma.array(data, mask=mask.astype(bool))
    masked_array = np.ma.masked_invalid(masked_array)
    return masked_array

# def incremental_cfads(root_folder: str, target_parent_folder: str, file_extension: str = '.nc', chunk_size: int = 1000):
#     """
#     Read NetCDF files from subdirectories of a root folder, incrementally update and plot 2D histograms with frequency-based contour plots for given variables.

#     Args:
#         root_folder (str): The root folder to start the search from.
#         target_parent_folder (str): The name of the target parent folder to process data from.
#         variables (list): List of variable names to plot.
#         file_extension (str, optional): The file extension to filter files. Defaults to '.nc'.
#         chunk_size (int, optional): Size of chunks for incremental processing. Defaults to 1000.
#     """
#     # Define a Seaborn color palette for the colormap with inverted colors
#     cmap = sns.color_palette("turbo", as_cmap=True)

#     last_filepath = None  # Initialize variable to track the last filepath
#     fig = None  # Initialize the figure variable
#     gs = None  # Initialize the gridspec
#     colorbars = {}  # Initialize a dictionary to store colorbars for each variable

#     for parent_folder, _, _ in os.walk(root_folder):
#         parent_folder_name = os.path.basename(parent_folder)

#         if parent_folder_name == target_parent_folder:
#             for _, _, filenames in os.walk(parent_folder):
#                 for filename in filenames:
#                     filepath = os.path.join(parent_folder, filename)

#                     if filename.endswith(file_extension):
#                         dataset = xr.open_dataset(filepath)
#                         try:
#                             # Check if the actual filepath before parent folder change
#                             if last_filepath is None or os.path.dirname(filepath) != os.path.dirname(last_filepath):
#                                 # Close the previous figure if it exists
#                                 if fig is not None:
#                                     plt.close(fig)

#                                 # Create a new figure with a single gridspec
#                                 num_vars = len(dataset.data_vars)
#                                 num_cols = num_vars * 2
#                                 fig = plt.figure(figsize=(12 + num_cols, 6))
#                                 gs = gridspec.GridSpec(1, num_cols, figure=fig, width_ratios=[1, 0.05] * num_vars, wspace=0.6)
#                                 cumulative_histograms = {var_name: None for var_name in dataset}
#                                 bin_centers = {var_name: None for var_name in dataset}

#                             last_filepath = filepath  # Update last filepath

#                             # Create subplots and colorbars using the single gridspec
#                             for i, var_name in enumerate(dataset):
#                                 ax = fig.add_subplot(gs[0, i * 2])
#                                 cax = fig.add_subplot(gs[0, i * 2 + 1])
#                                 # set_trace()
#                                 var_values = dataset[var_name]
#                                 range_values = dataset.range
#                                 flattened_var = var_values.values[~np.isnan(var_values.values)]
#                                 flattened_range_var = np.tile(range_values, var_values.shape[0])[~np.isnan(var_values.values.ravel())]

#                                 chunk_histogram, x_edges, y_edges = np.histogram2d(flattened_var, flattened_range_var, bins=(50, 50), density=True)

#                                 if np.all(np.isnan(chunk_histogram)):
#                                     print(f"Chunk histogram for {var_name} contains only NaN values.")
#                                 else:
#                                     if cumulative_histograms[var_name] is None:
#                                         cumulative_histograms[var_name] = chunk_histogram
#                                         bin_centers[var_name] = (x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2 / 1e3  # Convert to km
#                                     else:
#                                         cumulative_histograms[var_name] += chunk_histogram

#                                     ax.clear()  # Clear the previous plot
#                                     contour = ax.contourf(bin_centers[var_name][0], bin_centers[var_name][1], cumulative_histograms[var_name].T, levels=20, cmap=cmap)
#                                     ax.set_xlabel(var_name)
#                                     ax.set_ylabel('Height [km]')
#                                     ax.set_title(f'2D Histogram of {var_name} vs Height')
#                                     ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

#                                     if var_name in colorbars:
#                                         colorbars[var_name].remove()  # Remove the previous colorbar
#                                     colorbars[var_name] = fig.colorbar(contour, cax=cax, label='Density')
#                                     colorbars[var_name].ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#                                     cax.yaxis.tick_left()  # Move colorbar ticks and values to the left side
#                                     cax.yaxis.set_label_position('left')  # Move colorbar label to the left side
#                                     plt.pause(0.1)  # Pause to allow the plot to updat
#                             # plt.tight_layout()  # Adjust layout using tight_layout
#                             plt.show()

#                         except Exception as e:
#                             print(f"Error reading {filename}: {e}")

# def read_and_concatenate_datasets(root_folder: str, target_parent_folder: str, file_extension: str = '.nc') -> Dict[str, xr.Dataset]:
#     """
#     Read NetCDF files from subdirectories of a root folder, concatenate them, and return concatenated datasets
#     grouped by the folder before the target parent folder.

#     Args:
#         root_folder (str): The root folder to start the search from.
#         target_parent_folder (str): The name of the target parent folder to concatenate datasets from.
#         file_extension (str, optional): The file extension to filter files. Defaults to '.nc'.

#     Returns:
#         dict: A dictionary where keys are folder names before the target parent folder and values are concatenated xarray datasets.
#     """
#     concatenated_datasets_dict = {}

#     # Walk through the root folder and its subdirectories
#     for parent_folder, _, _ in os.walk(root_folder):
#         parent_folder_name = os.path.basename(parent_folder)

#         if parent_folder_name == target_parent_folder:
#             # Get the folder name before the target parent folder
#             previous_folder_name = os.path.basename(os.path.dirname(parent_folder))

#             concatenated_dataset = None

#             for _, _, filenames in os.walk(parent_folder):
#                 for filename in filenames:
#                     filepath = os.path.join(parent_folder, filename)

#                     # Check if the file has the desired extension
#                     if filename.endswith(file_extension):
#                         try:
#                             # Read the file into an xarray dataset
#                             dataset = xr.open_dataset(filepath)

#                             if concatenated_dataset is None:
#                                 # Initialize the concatenated dataset with the first file's data
#                                 concatenated_dataset = dataset
#                             else:
#                                 # Concatenate the new dataset with the existing concatenated dataset
#                                 concatenated_dataset = xr.concat([concatenated_dataset, dataset], dim='time')

#                         except Exception as e:
#                             print(f"Error reading {filename}: {e}")

#             if concatenated_dataset is not None:
#                 concatenated_datasets_dict[previous_folder_name] = concatenated_dataset.sortby('time')

#     return concatenated_datasets_dict

def reading_dataset_chunking(root_folder: str, target_parent_folder: str, file_extension: str = '.nc') -> Dict[str, xr.Dataset]:
    concatenated_datasets_dict = {}

    for parent_folder, _, _ in os.walk(root_folder):
        parent_folder_name = os.path.basename(parent_folder)

        if parent_folder_name == target_parent_folder:
            previous_folder_name = os.path.basename(os.path.dirname(parent_folder))

            delayed_datasets = []

            for _, _, filenames in os.walk(parent_folder):
                for filename in filenames:
                    filepath = os.path.join(parent_folder, filename)

                    if filename.endswith(file_extension):
                        if file_extension == '.nc':
                            # Create a delayed function to open the file as a Dask-backed xarray dataset
                            delayed_dataset = dask.delayed(xr.open_dataset)(filepath)
                            delayed_datasets.append(delayed_dataset)
                        elif file_extension == '.json':
                            # Read and process JSON file using the json library
                            # Read the JSON file into a pandas DataFrame
                            df = pd.read_json(filepath, orient='index')
                            df.index.name = 'time'
                            delayed_dataset = dask.delayed(xr.Dataset.from_dataframe)(df)
                            delayed_datasets.append(delayed_dataset)

            if delayed_datasets:
                # Use Dask's delayed computation to parallelize the dataset opening
                datasets = dask.compute(*delayed_datasets)

                concatenated_datasets = []

                for dataset in datasets:
                    # Calculate chunking based on the dimensions of the current dataset
                    chunking = {}
                    for dim in dataset.dims:
                        chunking[dim] = dataset.sizes[dim]

                    # Apply chunking to the current dataset
                    chunked_dataset = dataset.chunk(chunking)
                    concatenated_datasets.append(chunked_dataset)

                # Concatenate the chunked datasets along the 'time' dimension
                concatenated_dataset = xr.concat(concatenated_datasets, dim='time')

                concatenated_datasets_dict[previous_folder_name] = concatenated_dataset.sortby('time')

    return concatenated_datasets_dict

def slice_data_by_season(data):
    # Define the seasons and corresponding date ranges
    seasons = {
        'summer': ('06-01', '08-31'),    # from 1st June to 31st August
        'fall': ('09-01', '11-30'),     # from 1st September to 30th November
        'winter': ('12-01', '02-28'),   # from 1st December to 28th February
        'spring': ('03-01', '05-31')    # from 1st March to 31st May
    }

    # Initialize an empty list to store delayed computations for seasonal data
    seasonal_dict = {}
    seasonal_data = []
    # Loop through each season
    for season_name, (start_date, end_date) in seasons.items():
        # Create datetime objects for the start and end dates, assuming the current year
        years = np.unique(data['time.year'].values)
        for current_year in years:
            year_start_date = pd.to_datetime(f'{current_year}-{start_date}', format='%Y-%m-%d')
            year_end_date = pd.to_datetime(f'{current_year}-{end_date}', format='%Y-%m-%d')

            # Use Dask to slice the data for the current season without loading it into memory
            seasonal_data_year = data.sel(time=slice(year_start_date, year_end_date))

            # Append the delayed computation to the list
            seasonal_data.append(seasonal_data_year)
        if seasonal_data:
            # Use Dask to concatenate the delayed computations for the current season
            seasonal_dict[season_name] = xr.concat(seasonal_data, dim='time').sortby('time')
        else:
            seasonal_dict[season_name] = None
    # Combine the delayed computations into a single Xarray DataArray or Dataset
    return seasonal_dict

def add_season_coordinate(data):
    # Define the seasons and corresponding date ranges
    seasons = {
        'summer': ('06-01', '08-31'),    # from 1st June to 31st August
        'fall': ('09-01', '11-30'),     # from 1st September to 30th November
        'winter': ('12-01', '02-28'),   # from 1st December to 28th February
        'spring': ('03-01', '05-31')    # from 1st March to 31st May
    }

    # Create a new DataArray for 'season' with the same dimensions as 'time'
    season_data = np.empty(data['time'].shape, dtype='U3')

    # Loop through each season
    for season_name, (start_date, end_date) in seasons.items():
        # Loop through years
        years = np.unique(data['time.year'].values)
        for current_year in years:
            # Create datetime objects for the start and end dates
            year_start_date = pd.to_datetime(f'{current_year}-{start_date}', format='%Y-%m-%d')
            year_end_date = pd.to_datetime(f'{current_year}-{end_date}', format='%Y-%m-%d')

            # Assign the season name to the corresponding time range
            mask = (data['time.year'] == current_year) & (data['time'] >= year_start_date) & (data['time'] <= year_end_date)
            season_data[mask] = season_name

    set_trace()
    # Create a new DataArray for 'season' and assign it to the dataset
    season_coord = xr.DataArray(season_data, dims='time', coords={'time': data['time']})
    data.coords['season'] = season_coord

    # Return the updated data with 'season' as a coordinate
    return data

def reindex_datasets(chunked_dataset, month=[], chunk_size: int = 1000, freq_index="30S", method=None, tolerance=None):

    time_series = chunked_dataset.indexes['time'] # Convert to pandas DateTimeIndex
    # Calculate the new start time as the first 15 seconds of the day
    if month:
        new_start_time = time_series.min().replace(month = month[0], day=1, hour=0, minute=0, second=15, microsecond=0)
        new_end_time = time_series.max().replace(month = month[1], day=1, hour=23, minute=59, second=59, microsecond=0) + relativedelta(day=31)
    else:
        new_start_time = time_series.min().replace(day=1, hour=0, minute=0, second=15, microsecond=0)
        new_end_time = time_series.max().replace(day=1, hour=23, minute=59, second=59, microsecond=0) + relativedelta(day=31)


    # Create a new time index starting from the new_start_time and ending at the end of the day
    new_time_index = pd.date_range(start=new_start_time, end=new_end_time, freq=freq_index)

    # Reindex the concatenated variable
    reindexed_variable = chunked_dataset.reindex(time=new_time_index, fill_value=np.nan, method=method, tolerance=tolerance)

    return reindexed_variable.chunk({"time": chunk_size})

def get_complete_time(chunked_dataset, chunk_size: int = 1000, freq_index="30S"):
    time_series = chunked_dataset.indexes['time'] # Convert to pandas DateTimeIndex

    # Calculate the new start time as the first 15 seconds of the day
    new_start_time = time_series.min().replace(day=1, hour=0, minute=0, second=15, microsecond=0)
    new_end_time = time_series.max().replace(day=1, hour=23, minute=59, second=59, microsecond=0) + relativedelta(day=31)

    # Create a new time index starting from the new_start_time and ending at the end of the day
    new_time_index = pd.date_range(start=new_start_time, end=new_end_time, freq=freq_index)

    return new_time_index


def sum_variable_and_concatenate(datasets_dict: Dict[str, xr.Dataset], var: str, dim_sum: str, dim_cat: str, dim_sort: str) -> xr.Dataset:
    """
    Process each dataset in the input dictionary by summing over the 'range' dimension of the 'Total' variable,
    and then concatenate all the results along the 'time' dimension.

    Parameters:
        datasets_dict (Dict[str, xr.Dataset]): A dictionary where keys are identifiers and values are xarray datasets.

    Returns:
        xr.Dataset: The concatenated dataset.
    """
    processed_datasets = [ds[var].sum(dim=dim_sum, skipna=False) for ds in datasets_dict.values()]
    concatenated_dataarray = xr.concat(processed_datasets, dim=dim_cat).sortby(dim_sort)
    return concatenated_dataarray

def concatenate_dic_by_time(concatenated_datasets: Dict[str, xr.Dataset]) -> xr.Dataset:
    """
    Concatenates all the datasets in the concatenated_datasets dictionary.

    Parameters:
        concatenated_datasets (Dict[str, xr.Dataset]): A dictionary where keys are identifiers and values are xarray datasets.

    Returns:
        xr.Dataset: The concatenated dataset.
    """
    processed_datasets = list(concatenated_datasets.values())
    concatenated_dataset = xr.concat(processed_datasets, dim='time').sortby('time')
    return concatenated_dataset

def assign_season(ds, seasons):
    """
    Assign season labels to a given xarray dataset based on specified season ranges.

    Parameters:
    ds (xarray.Dataset): The input xarray dataset with a 'time' dimension.
    seasons (dict): A dictionary that defines the season ranges in terms of months.

    Returns:
    xarray.Dataset: A new dataset with an additional 'season' coordinate.
    """

    # Create a mask for each season
    conditions = []
    for name, (start, end) in seasons.items():
        if name == 'winter':
            # Handle the wrap-around between December and January
            cond = (ds['time.month'] >= start) | (ds['time.month'] <= end)
        else:
            cond = (ds['time.month'] >= start) & (ds['time.month'] <= end)
        conditions.append(cond)

    # List of corresponding season names
    season_names = list(seasons.keys())

    # Use np.select to assign season names based on conditions
    season_data = xr.DataArray(
        np.select(conditions, season_names, default='unknown'),
        dims=('time',)
    )

    ds = ds.assign_coords(season=season_data)

    return ds

def plot_cloud_prop_along_time(ds, clouds_single_layer, chirp_integrated_var, variable, save_path):

    time_complete = get_complete_time(ds)
    time_complete_df = pd.DataFrame({'datetime': time_complete})
    time_complete_df['month'] = time_complete_df['datetime'].dt.month
    count_complete = time_complete_df.groupby('month').count()['datetime']

    for var_name in ds.data_vars:
        # Create a new figure and subplots for each var_name
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 12))  # 2 rows, 1 column

        with sns.axes_style("darkgrid"):
            # Line plot (upper subplot)
            cond = clouds_single_layer[var_name].compute() == 1
            var = ds[var_name].where(cond, drop=True)
            new_integrated_var = chirp_integrated_var.where(cond, drop=True)
            sns.lineplot(
                x=var.time.dt.month.values,
                y=var.values/1e3,
                ax=axes[0],
                hue=var.time.dt.year.values,
                palette="Set2",
                errorbar="sd",
                marker="o",
            )
            axes[0].set_title(f"{var_name}")
            axes[0].set_ylabel(f"{variable} [km]")
            axes[0].grid(True)

            # Scatter plot (lower subplot)
            var['month'] = var['time'].dt.month
            monthly_data = var.groupby('month').mean()
            counts = var.groupby('month').count()

            months = monthly_data['month'].values
            means = monthly_data.values / 1e3

            scatter = axes[1].scatter(months, means, c=counts.values, cmap='jet', marker='o', s=100, edgecolor='k')
            cbar = plt.colorbar(scatter, ax=axes[1], orientation='horizontal', pad=0.2, shrink=0.8, aspect=30)
            cbar.set_label('Number of Profiles')

            std_dev = np.sqrt(var.groupby('month').var().values) / 1e3
            print(f"Variable: {var_name}, Mean of Means: {np.nanmean(means)}, Std of Means: {np.nanstd(means)}")
            axes[1].errorbar(months, means, yerr=std_dev, fmt='none', c='k', capsize=4)
            axes[1].set_ylabel(f"{variable} [km]")
            axes[1].set_xlabel("Month")
            axes[1].grid(True)
            # Create a twin axes for the secondary y-axis on the right
            ax2 = axes[1].twinx()

            # Plot the new_integrated_var data on the twin axes
            new_integrated_var['month'] = new_integrated_var['time'].dt.month
            monthly_data_integ_var = new_integrated_var.groupby('month').mean()
            line_color = 'red'  # Get the color of the first line in the palette
            ax2.tick_params(axis='y', colors="black")
            ax2.plot(months, monthly_data_integ_var.LWP, color=line_color, linestyle='--', marker='s', markersize=6, label='LWP')

            # Plot the IWP data on the same twin axes
            ax2.plot(months, monthly_data_integ_var.IWP, color='blue', linestyle='-.', marker='^', markersize=6, label='IWP')

            ax2.set_ylabel(r"LWP/IWP [kg $m^{-2}$]", color="black")
            ax2.legend(loc='upper right')  # Add a legend

        # Adjust subplot layout and spacing
        plt.tight_layout()
        fig.savefig(f"{save_path}_{var_name}.png", dpi=300)
        # Show the plots for the current var_name
        plt.show()

def plot_seasonal_histograms(ds, xname, save_path=None):
    """
    Create and display seasonal histograms for each variable in a dataset.

    Parameters:
    ds (xarray.Dataset): The input xarray dataset containing the data.
    save_path (str, optional): If provided, save the figure to this path.

    Returns:
    None
    """

    unique_season = np.unique(ds.season.values)
    unique_vars = ds.data_vars

    # Define the number of rows and columns for the subplots
    num_seasons = len(unique_season)
    num_cols = 2  # Set the number of columns to be 2

    # Calculate the number of rows needed
    num_rows = (num_seasons + num_cols - 1) // num_cols

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8 * num_rows), sharex=True)
    data = {var_name: [] for var_name in unique_vars}
    for i, season in enumerate(unique_season):
        mask_season = ds.season == season
        data_season = {}

        for var_name in unique_vars:
            cloud_prop_cb_single_layer = ds[var_name].where(clouds_single_layer[var_name].compute() == 1, drop=True)
            data_season[var_name] = cloud_prop_cb_single_layer.where(mask_season, drop=True).values

        ax = axes[i // num_cols, i % num_cols]
        with sns.axes_style("whitegrid"):
            for var_name, values in data_season.items():
                sns.histplot(values, kde=True, bins=30, label=f"{var_name} ({len(values)} Profiles)", ax=ax)
                # print(f"Season: {season}, Variable: {var_name}, Mean: {np.nanmean(values)}, Std: {np.nanstd(values)}, Skew: {stats.skew(np.asarray(values, dtype=np.float64))}, Kurtosis: {stats.kurtosis(np.asarray(values, dtype=np.float64))}")
                # Get the histogram data
                hist, bin_edges = np.histogram(values, bins=30, density=True)
                # Calculate cumulative distribution function (CDF)
                cdf = np.cumsum(hist * np.diff(bin_edges))
                # Find the bin where the CDF crosses 0.5 (median)
                median_bin = np.searchsorted(cdf, 0.5)
                # Estimate the median value
                median_value = (bin_edges[median_bin] + bin_edges[median_bin + 1]) / 2
                if var_name == 'Liquid' and season == 'summer':
                    data[var_name].append(np.nan)
                else:
                    data[var_name].append(values)
                print(f"Season: {season}, Variable: {var_name}, Meadian: {median_value}")
                # print(f"Season: {season}, Variable: {var_name}, Meadian: {np.nanmedian(values)}, 25th Percentile: {np.nanpercentile(values, 25)}, 75th Percentile: {np.nanpercentile(values, 75)}")
        ax.set_title(f"Season - {season}")
        ax.set_xlabel(xname)
        ax.legend()

    for var_name, nested_list in data.items():
        if len(nested_list) > 0:
            # Flatten the list of values
            flattened_list = list(chain.from_iterable(sublist if isinstance(sublist, np.ndarray) else [sublist] for sublist in nested_list if not (isinstance(sublist, float) and np.isnan(sublist))))
            print(f"Variable: {var_name}, Mean: {np.nanmean(flattened_list)/1000}, Std: {np.nanstd(flattened_list)/1000}, Median: {np.nanmedian(flattened_list)/1000}")
            # print(f"Variable: {var_name}, Mean: {np.nanmean(values)/1000}, Std: {np.nanstd(values)/1000}, Median: {np.nanmedian(values)/1000}")


    # Remove any empty subplots
    if num_seasons < num_rows * num_cols:
        for i in range(num_seasons, num_rows * num_cols):
            fig.delaxes(axes[i // num_cols, i % num_cols])

    # Adjust subplot layout and spacing
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.tight_layout()
    plt.show()

def calculate_time_keep(ds_available, ds_reference_complete, freq_rm: str = "M", threshold: float = 0.3):
    """
    Calculates the time periods to keep based on the availability of cloud data.

    Parameters:
    - ds_available: xarray.Dataset
        Dataset containing cloud layers data.
    - ds_reference_complete: xarray.Dataset
        Dataset containing reindexed cloud layers data.
    - freq_rm: str, optional
        Resampling frequency for time aggregation (default is "M" for monthly).
    - threshold: float, optional
        Threshold value for determining the time periods to keep (default is 0.3).

    Returns:
    - time_keep: xarray.DataArray
        DataArray containing the time periods to keep based on the availability of cloud data.
    """
    count_data_available = ds_available.time.resample(time=freq_rm).count(dim='time')
    count_data_total     = ds_reference_complete.time.resample(time=freq_rm).count(dim='time').compute()

    time_rm_intersec = np.intersect1d(count_data_available.time.values, count_data_total.time.values)
    count_data_available = count_data_available.sel(time=time_rm_intersec)
    count_data_total     = count_data_total.sel(time=time_rm_intersec)

    mask_rm = count_data_available / count_data_total > threshold
    time_keep = count_data_available.time[mask_rm]

    return time_keep

def calculate_time_remove(ds_available, ds_reference_complete, freq_rm: str = "M", threshold: float = 0.3):
    """
    Calculates the time periods to keep based on the availability of cloud data.

    Parameters:
    - ds_available: xarray.Dataset
        Dataset containing cloud layers data.
    - ds_reference_complete: xarray.Dataset
        Dataset containing reindexed cloud layers data.
    - freq_rm: str, optional
        Resampling frequency for time aggregation (default is "M" for monthly).
    - threshold: float, optional
        Threshold value for determining the time periods to keep (default is 0.3).

    Returns:
    - time_keep: xarray.DataArray
        DataArray containing the time periods to keep based on the availability of cloud data.
    """
    count_data_available = ds_available.time.resample(time=freq_rm).count(dim='time')
    count_data_total     = ds_reference_complete.time.resample(time=freq_rm).count(dim='time').compute()

    time_rm_intersec = np.intersect1d(count_data_available.time.values, count_data_total.time.values)
    count_data_available = count_data_available.sel(time=time_rm_intersec)
    count_data_total     = count_data_total.sel(time=time_rm_intersec)

    mask_rm = count_data_available / count_data_total < threshold
    time_keep = count_data_available.time[mask_rm]

    return time_keep


hydrometeor_analysis = False

cloud_properties_analysis = True
plot_data_availability = True
plot_cloud_data_availability = False
plot_daily_cloud_availability = True
plot_lwc_iwc_der_ier = False
plot_integrated_water_and_ice_path=False

cloud_macrophysics_analysis = False

radar_variables_analysis = False

target_parent_folder = "number_of_layers"
file_extension = '.nc'
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
mask_cloud_availability = cloud_layers.where(mask_single, other=0.0)
mask_cloud_availability['no_clouds']  = xr.DataArray(mask_w_clouds.astype(np.float64), dims='time')
mask_cloud_availability['multilayer'] = xr.DataArray(mask_multi.astype(np.float64), dims='time')
# -------------------------------------------------------------------------------------------------

clouds_single_layer     = cloud_layers.sel(time=mask_single)   # Select only single layer clouds
clouds_single_layer     = clouds_single_layer.assign_coords(years=clouds_single_layer['time'].dt.year, month=clouds_single_layer['time'].dt.month)
reindexed_clouds_layers = reindex_datasets(mask_cloud_availability, month=[1, 12])
reindexed_clouds_layers['missing'] = reindexed_clouds_layers.multilayer.isnull().astype(np.float64) # ww choose multilayers but could be any other variable. NAns means no data
reindexed_clouds_layers = reindexed_clouds_layers.assign_coords(years=reindexed_clouds_layers['time'].dt.year, month=reindexed_clouds_layers['time'].dt.month)
# # -----------------------------------------------------------------------------------------------
# # Uncomment if want to check mutually exclusive of data
# # -----------------------------------------------------------------------------------------------
# if np.sum(reindexed_clouds_layers.to_dataframe().sum(axis=1) == 1.0) == reindexed_clouds_layers.time.size:
#     print("Data is mutually exclusive")
# # -----------------------------------------------------------------------------------------------
cloud_layers            = cloud_layers.assign_coords(years=cloud_layers['time'].dt.year, month=cloud_layers['time'].dt.month)
mask_cloud_availability = mask_cloud_availability.assign_coords(years=mask_cloud_availability['time'].dt.year, month=mask_cloud_availability['time'].dt.month)

time_for_keeping = calculate_time_keep(cloud_layers, reindexed_clouds_layers, freq_rm="M", threshold=0.5)

if hydrometeor_analysis:
    target_parent_folder = "hydrometeor"
    file_extension = '.nc'
    chirp_hydromet = reading_dataset_chunking(root_folder, target_parent_folder)

    # for key in chirp_hydromet:
    #     dataset = chirp_hydromet[key]
    #     initial_time = dataset.time[0].values
    #     final_time = dataset.time[-1].values
    #     print(f"Key: {key}, Initial Time: {initial_time}, Final Time: {final_time}")

    # keys = list(chirp_hydromet.keys())
    # for i in range(len(keys)):
    #     for j in range(i+1, len(keys)):
    #         key1 = keys[i]
    #         key2 = keys[j]
    #         dataset1 = chirp_hydromet[key1]
    #         dataset2 = chirp_hydromet[key2]
    #         mask = dataset1.time == dataset2.time
    #         print(f"Comparing {key1} and {key2}, Time Overlaping: {mask.sum().item()}")

    # # Create the figure and set the size
    # fig, ax = plt.subplots(figsize=(10, 6))
    # # Iterate over the keys in chirp_hydromet
    # for key in chirp_hydromet:
    #     dataset = chirp_hydromet[key]
    #     time_values = dataset.time.values

    #     # Plot a constant value (key) against the time values
    #     ax.plot(time_values, [key] * len(time_values),'*')
    # # Set the x-axis label
    # plt.xlabel('Time')
    # # Set the y-axis label
    # plt.ylabel('Key')
    # # Set the title
    # plt.title('Key vs Time')
    # # Show the plot
    # fig.savefig(f"{PATH_FIG}key_vs_time.png", dpi=300)
    # plt.show()

    freq_profile_lis = [ds["Total"].mean(dim="range", skipna=False) for ds in chirp_hydromet.values()]
    freq_profile_concat = xr.concat(freq_profile_lis, dim="time").sortby("time")
    freq_pr_reindex = reindex_datasets(freq_profile_concat)

    # freq_list = [.1, .25, .5, .75, .9]
    # freq_pr_grater_than = [(freq_pr_reindex > f).resample(time="D").mean() for f in freq_list]
    # # freq_pr_grater_than.append(freq_pr_reindex.isnull().resample(time="D").mean())
    # freq_labels = [f"f>{f}" for f in freq_list]
    # # freq_labels.append("missing")

    hydro_total_list = [ds["Total"].sum(dim="range", skipna=False) for ds in chirp_hydromet.values()]
    chirp_concatenated_hydrometeors = xr.concat(hydro_total_list, dim="time").sortby("time")
    reindexed_variable = reindex_datasets(chirp_concatenated_hydrometeors)
    # # -----------------------------------------------------------------------------------------------
    # # Specify the start and end dates for the data you're interested in (replace with your desired dates)
    # # -----------------------------------------------------------------------------------------------
    # start_date = "2021-04-01"
    # end_date = "2021-04-30"
    # #sliced_variable = reindexed_variable.sel(time=slice(start_date, end_date))
    # sliced_variable = reindexed_variable
    # freq_str = "M"

    # # colors = ["#28fc21", "#07a8e3", "#ffffff", "#ff0000", "#0000ff", "#000000"]
    # # with sns.axes_style("ticks"):
    # #     create_hydromet_bar_plot(freq_pr_grater_than,
    # #                              freq_str, freq_labels,
    # #                              colors, figname=f"{PATH_FIG}hydrometeor_bar_plot.png")

    # Create a mask where NaN values are True
    nan_mask_no_data = freq_pr_reindex.isnull()
    # not_null_mask = ~nan_mask_no_data

    # for i in range(freq_pr_reindex.shape[0]):
    #     print(f"Time: {freq_pr_reindex.time[i].values}, NaN: {nan_mask_no_data[i]}")

    # Count NaN values by month
    nan_count_by_month = nan_mask_no_data.resample(time='1M').mean(dim='time').values
    month_mask         = nan_count_by_month < .7

    # with sns.axes_style("ticks"):
    #     create_hydromet_box_plot(freq_pr_reindex,figname=f"{PATH_FIG}hydrometeor_boxplot_plot.png")

    # get values comple
    # freq_str = "M"
    # with sns.axes_style("ticks"):
    #     create_data_availability_plot(reindexed_variable, freq_str, figname=f"{PATH_FIG}data_availability.png")

    # # Create a figure and axis
    # fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed

    # # Iterate through variable names and plot each variable using the axis.plot method
    # for i, f in enumerate(freq_pr_grater_than):
    #     ax.plot(f.time, f, label=freq_labels[i])

    # # Add labels, title, and legend
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Frequency [%]")
    # ax.set_title("Variable Plots")
    # ax.grid(True)
    # ax.legend()
    # plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    # # Show the plot
    # plt.show()

    # # Create a figure and subplots
    # fig, axs = plt.subplots(len(freq_pr_grater_than), 1, figsize=(10, 6*len(freq_pr_grater_than)),
    #                         sharex=True)  # Adjust the figure size as needed

    # # Iterate through variable names and plot each variable using the axis.plot method
    # for i, (f, ax) in enumerate(zip(freq_pr_grater_than, axs)):
    #     # Group the data by year and month
    #     years = np.unique(f.time.dt.year)
    #     # Iterate through each group and plot the data
    #     for y in years:
    #         # Extract the month from the group name
    #         data_year = f.sel(time=f.time.dt.year == y)
    #         # Plot the data for the current month
    #         ax.plot(data_year.time.dt.month, data_year, label=y)

    #     # Add labels, title, and legend to each subplot
    #     ax.set_ylabel("Frequency [%]")
    #     # ax.set_title(f"Variable Plot - {freq_labels[i]}")
    #     ax.grid(True)

    # ax.set_xlabel("Month")
    # ax.legend()
    # #ax.set_xticks(np.arange(1, 13))  # Set x-axis ticks for each month
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    # # Rotate x-axis labels for better visibility
    # plt.setp(ax.get_xticklabels(), rotation=45)

    # # Adjust the spacing between subplots
    # plt.tight_layout()

    # # Show the plot
    # plt.show()

    # -----------------------------------------------------------------------------------------------
    # Frequency od occurence for hydrometeors
    # -----------------------------------------------------------------------------------------------
    freq_str = "M"

    chirp_hydromet_list = [ds for ds in chirp_hydromet.values()]
    min_chirp_range = min([var.range.values.min() for var in chirp_hydromet_list])
    max_chirp_range = max([var.range.values.max() for var in chirp_hydromet_list])

    delta_range = 60
    bin_edges   = np.arange(0, max_chirp_range+delta_range, delta_range)

    # Using a generator expression to populate list
    result_generator  = (var.groupby_bins('range', bin_edges, labels=bin_edges[1:]).max() for var in chirp_hydromet_list)

    range_binned_data = []
    for result in result_generator:
        range_binned_data.append(result)

    hydromet_data = xr.concat(range_binned_data, dim='time').sortby('time')
    hydromet_data = hydromet_data.assign_coords(years=hydromet_data['time'].dt.year, month=hydromet_data['time'].dt.month)
    hydromet_data = assign_season(hydromet_data, SEASONS)

    # -----------------------------------------------------------------------------------------------
    # Calculating frequency of available data by month
    # -----------------------------------------------------------------------------------------------
    # time_complete = get_complete_time(hydromet_data)
    # time_complete_df = pd.DataFrame({'datetime': time_complete})
    # time_complete_df['month'] = time_complete_df['datetime'].dt.month
    # count_complete = time_complete_df.groupby('month').count()['datetime']

    # time_hydromet = hydromet_data.time
    # time_hydromet_df = pd.DataFrame({'datetime': time_hydromet})
    # time_hydromet_df['month'] = time_hydromet_df['datetime'].dt.month
    # count_hydromet = time_hydromet_df.groupby('month').count()['datetime']
    # # -----------------------------------------------------------------------------------------------

    # freq_occurence = 100* ( count_hydromet / count_complete )
    set_trace()

    hydromet_to_analyze =['Liquid', 'Ice']
    dy_ticks = 2 #km
    fig                 = plt.figure(figsize=(17, 13))
    # gs                  = fig.add_gridspec(3, 2, height_ratios=[4, 4, 2], width_ratios=[3, .1], wspace=0.08, hspace=0.3)
    gs                  = fig.add_gridspec(3, 3, height_ratios=[5, 5, 2.5], width_ratios=[3, .07, 1], wspace=0.22, hspace=0.15)
    for i, var_name in enumerate(hydromet_to_analyze):
        print(f"Processing {var_name}")
        ds                = hydromet_data[var_name].compute()
        hydromet_by_month       = ds.groupby('month').mean(dim='time')
        hydromet_by_season_mean = ds.groupby('season').mean(dim='time')
        # set_trace()
        ax1 = fig.add_subplot(gs[i,0])
        p1 = ax1.pcolormesh( hydromet_by_month.month, hydromet_by_month.range_bins/1e3, 100*hydromet_by_month.values, shading='nearest', cmap='turbo')
        # ax1.set_title(f"2D Frequency Plot for {var_name}")
        ax1.set_ylabel("Height a.g.l (km)")
        ax1.set_xticks(np.arange(1, 13))
        # hide x-axis label
        ax1.xaxis.set_tick_params(labelbottom=False)
        ax1.set_yticks(np.arange(0, 12+dy_ticks,dy_ticks))
        ax1.grid(True)

        ax2 = fig.add_subplot(gs[i, 1])
        cbar = fig.colorbar(p1, cax=ax2, label=r'F$_{occurence}$(%)')
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.set_ticks_position('left')

        # season profiles for hydrometeors
        ax3 = fig.add_subplot(gs[i, 2], sharey=ax1)

        ax3.plot(100*hydromet_by_season_mean, hydromet_by_season_mean.range_bins/1e3, label=hydromet_by_season_mean.season.values)
        # ax3.set_title(f"Seasonal Profile for {var_name}")

        # hide y-axis label
        ax3.yaxis.set_tick_params(labelleft=False)
        # hide x-axis label
        if i == 0:
           ax3.xaxis.set_tick_params(labelbottom=False)
        else:
            ax3.set_xlabel("Frequency (%)")
        # ax3.set_xticks(np.arange(0, 0.5, 0.1))
        # remove top line of plot
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        ax3.grid(True)
        ax3.legend()

        freq_hydromet_by_month = 100*ds.max(dim='range_bins').groupby('month').mean()
        # Plotting frequncy of available data by month
        if i == 0:
            ax4 = fig.add_subplot(gs[2, 0], sharex=ax1)
        # scat = ax4.scatter(freq_hydromet_by_month['month'], freq_hydromet_by_month.values, c=freq_hydromet_by_month.values, marker='o', cmap='YlGn', s=100, label=var_name)
        line = ax4.plot(freq_hydromet_by_month['month'], freq_hydromet_by_month.values, '-o', linewidth=1, markersize=5, label=var_name)

    ax4.set_xlabel("Month")
    ax4.set_ylabel(r"F$_{hydrometeor}$(%)")
    # remove top line of plot
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_xticks(np.arange(1, 13))
    ax4.set_ylim(0, 100)
    ax4.grid(True)
    ax4.legend()

    # ax5 = fig.add_subplot(gs[2, 1])
    # # Create a colorbar with discrete colors in steps of 10 until 100
    # cbar_ticks = np.arange(0, 101, 10)
    # cbar = fig.colorbar(scat, cax=ax5, label='(%)', ticks=cbar_ticks)
    # cbar.ax.yaxis.set_label_position('left')
    # cbar.ax.yaxis.set_ticks_position('left')
    fig.savefig(f"{PATH_FIG}2d_freq_occurence_Liquid_Ice.png", dpi=300, bbox_inches='tight')
    plt.show()

    set_trace()



    # import concurrent.futures

    # # Define a function to process each variable in parallel
    # def process_variable(var):
    #     return var.groupby_bins('range', bin_edges, labels=bin_edges[1:]).max()

    # # Create a ThreadPoolExecutor with the desired number of threads
    # start_time = time_module.time()
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     # Submit each variable to the executor for processing
    #     futures = [executor.submit(process_variable, var) for var in chirp_hydromet.values()]

    #     # Wait for all the futures to complete and get the results
    #     range_binned_data = [future.result() for future in concurrent.futures.as_completed(futures)]

    # end_time = time_module.time()
    # # Calculate the execution time
    # execution_time = (end_time - start_time) / 60
    # print(f"Execution time: {execution_time} minutes")

    # #-----------------------------------------------------------------------------------------------
    # # Previous code for frequency of occurence for hydrometeors
    # # Uncomment if it is useful for future work
    # #-----------------------------------------------------------------------------------------------
    # range_binned_data = [var.groupby_bins('range', bin_edges, labels=bin_edges[1:]).max() for var in chirp_hydromet.values()]
    # time_binned_data  = [var.resample(time=freq_str).mean().dropna(dim='time', how='all') for var in range_binned_data]
    # hydro_freq_occurence = xr.concat(time_binned_data, dim='time').sortby('time')

    # time_binned_data  = [var.resample(time="1D").mean().dropna(dim='time', how='all') for var in chirp_hydromet.values()]
    # range_binned_data = [var.groupby_bins('range', bin_edges, labels=bin_edges[1:]).mean() for var in time_binned_data]
    # hydro_freq_occurence = xr.concat(range_binned_data, dim='time').sortby('time')
    # #-----------------------------------------------------------------------------------------------

    # with sns.axes_style("ticks"):
    #     plot_2d_and_vertical_frequency(hydro_freq_occurence, month_mask, freq_str, path=f"{PATH_FIG}")
    # set_trace()

# -----------------------------------------------------------------------------------------------
# Testing interpolation
# -----------------------------------------------------------------------------------------------
# x1 = chirp_hydromet['chirp_1']
# x2 = chirp_hydromet['chirp_4']
# y = xr.concat([x1.Ice, x2.Ice], dim='time', join=' left').sortby('time')
# print(y.range.values[:20])
# print(y.values[0,:20])
# print(x1.range.values[:20])
# print(x1.Ice.values[0,:20])

# # Use forward-fill (ffill) and backward-fill (bfill) to replace NaN values
# interp1 = y.interp(range=x2.range.values)
# # interp11 = y.bfill(dim='time').ffill(dim='time')
# print(interp1.values[0,:20])

# interp2 = y.ffill(dim='range').bfill(dim='range')
# print(interp2.values[0,:20])

# y1 = x1.Liquid.astype(float).compute()
# y2 = y1.sel(time = '2018-04-25T14:02:45.000000000')
# y2_nan_mask = y2 == 1
# print(y2.range[y2_nan_mask].values)
# z1 = x1.Liquid.astype(float).interp(range=new_range, method='nearest').compute()
# z2 = z1.sel(time = '2018-04-25T14:02:45.000000000')
# z2_nan_mask = z2 == 1
# print(z2.range[z2_nan_mask].values)
# nan_mask = np.isnan(merged_hydromet_dataset)
# mask_hydromet = (merged_hydromet_dataset == 1)
# month_hydromet = mask_hydromet.where(~nan_mask, np.nan)
# -----------------------------------------------------------------------------------------------
# freq_str = "M"
# hydro_list = [ds.sum(dim='range', skipna=False) for ds in chirp_hydromet.values()]
# hydro_frequency = xr.concat(hydro_list, dim='time').sortby('time')
# reindexed_hydro_frequency = reindex_datasets(hydro_frequency)

# with sns.axes_style("ticks"):
#     # Call the function to plot CFADs
#     plot_2d_and_vertical_frequency_2(chirp_hydromet, freq_str, path=f"{PATH_FIG}")
#     plot_time_evolution_frequency((reindexed_hydro_frequency > 0).resample(time=freq_str).mean(), figname=f"{PATH_FIG}time_evolution_frequency_hydrometeors.png")
# -----------------------------------------------------------------------------------------------
# Specify the folder path where the files are located
# root_folder    = '../../../processed_data/'
# target_parent_folder = "radar_variables"
# file_extension = '.nc'

# chirp_radar = reading_dataset_chunking(root_folder, target_parent_folder)
# bin_edges = {}
# nbins = 20
# bin_edges['Zh'] = [np.linspace(-60, 20, nbins), np.linspace(0, 12, nbins)]
# bin_edges['v']  = [np.linspace(-6, 6, nbins), np.linspace(0, 12, nbins)]

# chirp_radar_keys = list(chirp_radar.keys())
# for key in chirp_radar_keys:
#     plot_cfads2(chirp_radar[key],
#                 bin_edges,
#                 nbins,
#                 figname=f"{PATH_FIG}2d_histogram_chirp_{key}.png")
# -----------------------------------------------------------------------------------------------
# Specify the folder path where the files are located
# -----------------------------------------------------------------------------------------------


if radar_variables_analysis:
    target_parent_folder = "radar_variables"
    chirp_radar      = reading_dataset_chunking(root_folder, target_parent_folder)
    chirp_radar_list = [ds for ds in chirp_radar.values()]
    min_chirp_range = min([var.range.values.min() for var in chirp_radar_list])
    max_chirp_range = max([var.range.values.max() for var in chirp_radar_list])

    delta_range = 60
    bin_edges   = np.arange(0, max_chirp_range+delta_range, delta_range)

    # Using a generator expression to populate list
    result_generator  = (var.groupby_bins('range', bin_edges, labels=bin_edges[1:]).mean() for var in chirp_radar_list)

    range_binned_data = []
    for result in result_generator:
        range_binned_data.append(result)

    radar_data      = xr.concat(range_binned_data, dim='time').sortby('time')
    radar_data      = radar_data.assign_coords(years=radar_data['time'].dt.year, month=radar_data['time'].dt.month)
    radar_data      = assign_season(radar_data, SEASONS)

    radar_single      = radar_data.sel(time=mask_single)

    # # -----------------------------------------------------------------------------------------------
    # # Pice of code for verification, uncomment if needed
    # #  -- analysis of radar data for one day
    # # -----------------------------------------------------------------------------------------------
    # start_date = "2022-04-28 00:00:00"
    # end_date = "2022-04-28 23:59:59"
    # sliced_variable = radar_data.sel(time=slice(start_date, end_date)).compute()
    # freq_str = "M"

    # fig, ax = plt.subplots(figsize=(14, 6))  # Adjust the figure size as needed
    # dbz_mesh = ax.pcolormesh(sliced_variable.time, sliced_variable.range_bins, sliced_variable.Z, shading='nearest', cmap='viridis')
    # cbar = plt.colorbar(dbz_mesh, ax=ax, orientation='vertical', pad=0.05, aspect=20)
    # cbar.set_label('Reflectivity [dBZ]')
    # ax.set_xlabel('Time UTC [HH:MM]')
    # ax.set_ylabel('Height [m]')
    # # axis in HH:MM
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # plt.show()

    # nbins     = 100
    # x_edges   = np.linspace(-70, 20, nbins)
    # y_edges   = np.linspace(0, 14000, 50)

    # # Ploting a 2d histogram for sliced variable
    # x_flat = sliced_variable['Z'].values.T.flatten()
    # y_flat = np.tile(sliced_variable['Z'].range_bins.values, sliced_variable['Z'].time.size)

    # # remove nans
    # nan_mask = np.isnan(x_flat)
    # x_new = x_flat[~nan_mask]
    # y_new = y_flat[~nan_mask]

    # hist, x_edges, y_edges = np.histogram2d(x_new, y_new, bins=(100, 40))
    # bin_area = np.outer(np.diff(x_edges), np.diff(y_edges))
    # hist = hist / (np.sum(hist) * bin_area)

    # fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    # # Calculate bin centers for the contour plot
    # x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    # y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # # Contour plot for the current variable
    # contour = ax.contourf(x_centers, y_centers, hist.T, levels=50, cmap='jet', extend='both')
    # # mesh = ax.pcolormesh(x_centers, y_centers, hist.T, shading='nearest', cmap='jet')

    # # Add a colorbar
    # cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.05, aspect=20)
    # cbar.set_label('Frequency')
    # # Set the x-axis label
    # ax.set_xlabel('Reflectivity [dBZ]')
    # # Set the y-axis label
    # ax.set_ylabel('Height [m]')
    # # Show the plot
    # plt.show()

    # -----------------------------------------------------------------------------------------------
    # Histogram settings
    # -----------------------------------------------------------------------------------------------
    # clouds_to_analyse = ['Liquid', 'Mixed_phase', 'Ice']
    clouds_to_analyse = ['Pre_liquid', 'Pre_mixed_phase']

    var = 'v'
    if var == 'Z':
        x_edges   = np.linspace(-70, 20, 100)
        y_edges   = np.linspace(0, 14, 50)
    # -----------------------------------------------------------------------------------------------
    if var == 'v':
        x_edges   = np.linspace(-5, 5, 100)
        y_edges   = np.linspace(0, 14, 50)
    # -----------------------------------------------------------------------------------------------
    # time interval to reduce the amount of data and time to plot
    # -----------------------------------------------------------------------------------------------
    cut_time = False
    start_date = "2021-04-01 00:00:00"
    end_date   = "2021-04-30 23:59:59"
    # -----------------------------------------------------------------------------------------------
    nplots = len(clouds_to_analyse) + 1
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, nplots, width_ratios=[1]*(nplots-1)+[.05], wspace=0.08, hspace=0.3)
    ax = [None] * (len(clouds_to_analyse) + 1)
    max_hist = []
    for i, var_name in enumerate(clouds_to_analyse):
        print(f"Variable: {var_name}")
        cond           = clouds_single_layer[var_name].compute() == 1
        radar_var_name = radar_single.where(cond, drop=True)

        # -----------------------------------------------------------------------------------------------
        if cut_time:
            radar_var_name = radar_var_name.sel(time=slice(start_date, end_date)).compute()
        # -----------------------------------------------------------------------------------------------

        x_flat = radar_var_name[var].values.T.ravel()
        y_flat = np.tile(radar_var_name[var].range_bins.values/1000, radar_var_name[var].time.size)

        # Get mask for NaN values in x
        nan_mask = np.isnan(x_flat)
        x_new = x_flat[~nan_mask]
        y_new = y_flat[~nan_mask]

        # here, get hist using numpy 2d array
        hist,x_edges,y_edges= np.histogram2d(x_new, y_new, bins=(x_edges, y_edges))

        bin_area = np.outer(np.diff(x_edges), np.diff(y_edges))
        hist = hist / (np.sum(hist) * bin_area)

        # Calculate bin centers for the contour plot
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        ax[i]     = fig.add_subplot(gs[0, i])
        max_hist.append(np.nanmax(hist))
        contour = ax[i].contourf(x_centers, y_centers, hist.T, levels=20, cmap='turbo', extend='both')
        # pmesh = ax[i].pcolormesh(x_centers, y_centers, hist.T, shading='nearest', cmap='turbo')

        if i != 0:
           ax[i].yaxis.set_visible(False)

        # Check if it is the last subplot
        if i == len(clouds_to_analyse) - 1:
            ax[i+1] = fig.add_subplot(gs[0, i+1])
            # Add a colorbar to the last subplot
            cbar = plt.colorbar(contour, cax=ax[i+1], orientation='vertical')
            cbar.set_label('Density')

        # Set the x-axis label
        if var == 'Z':
            ax[i].set_xlabel('Reflectivity (dBZ)')
        if var == 'v':
            ax[i].set_xlabel(r'$\upsilon_{D}$ (m/s)')
        # Set the title for the subplot
        ax[i].set_title(f"{var_name} Clouds")
        # Set the y-axis label
        ax[i].set_ylabel('Height [km] a.m.s.l.')
        # Show the plot
    fig.savefig(f"{PATH_FIG}{var}_clouds_w_rain_2d_histogram.png", dpi=300, bbox_inches='tight')
    plt.show()
    set_trace()

if cloud_properties_analysis:
    print(" Starting cloud properties analysis...")
    # target_parent_folder = "lwp"
    target_parent_folder = "cloud_physical_properties"
    # chirp_lwp = reading_dataset_chunking(root_folder, target_parent_folder)
    chirp_cloud_prop = reading_dataset_chunking(root_folder, target_parent_folder)
    # Remove IWP and LWP variables from chirp_cloud_prop for each dictionary key
    chirp_integrated_var_list = []
    chirp_vertical_var_list   = []
    dropped_vars = ['IWP', 'LWP']
    keep_vars    = ['lwc', 'iwc', 'der', 'ier']
    for key, value in chirp_cloud_prop.items():
        chirp_vertical_var_list.append(value[keep_vars])
        chirp_integrated_var_list.append(value[dropped_vars])

    # Concatenate the modified chirp_cloud_prop datasets
    integrated_var              = xr.concat(chirp_integrated_var_list, dim='time').sortby('time')
    integrated_var              = integrated_var.assign_coords(years=integrated_var['time'].dt.year, month=integrated_var['time'].dt.month)
    integrated_var              = assign_season(integrated_var, SEASONS)
    integrated_var_single_layer = integrated_var.sel(time=mask_single) # Select integrated variables only for single layer clouds
    # -----------------------------------------------------------------------------------------------
    # Gridding vertical cloud properties for concatenating and reindexing
    # -----------------------------------------------------------------------------------------------
    min_chirp_range = min([var.height.values.min() for var in chirp_vertical_var_list])
    max_chirp_range = max([var.height.values.max() for var in chirp_vertical_var_list])

    delta_range = 60
    bin_edges   = np.arange(0, max_chirp_range+delta_range, delta_range)

    # Using a generator expression to populate list
    result_generator  = (var.groupby_bins('height', bin_edges, labels=bin_edges[1:]).mean() for var in chirp_vertical_var_list)

    range_binned_data = []
    for result in result_generator:
        range_binned_data.append(result)

    microphysics       = xr.concat(range_binned_data, dim='time').sortby('time')
    microphysics       = microphysics.assign_coords(years=microphysics['time'].dt.year, month=microphysics['time'].dt.month)
    microphysics       = assign_season(microphysics, SEASONS)
    microphysics_sigle = microphysics.sel(time=mask_single)

    time_for_nans     = calculate_time_remove(cloud_layers, reindexed_clouds_layers, freq_rm="M", threshold=0.5)

    # if plot_cloud_data_availability:
        # # -----------------------------------------------------------------------------------------------
        # # Grouping by month and plotting data availability for cloud layers
        # # -----------------------------------------------------------------------------------------------
        # fig = plt.figure(figsize=(18, 11))  # Increase the size of the plot for better visibility
        # gs  = fig.add_gridspec(2, 1, height_ratios=[4, 2], hspace=0.05)

        # colors = ["#f953d2", "#53d2f9", "#c9b337", "#2521b6", "#fc564f", "#49eb34","#ababab","#ffffff"]
        # data_available = (reindexed_clouds_layers == 1).groupby('month').mean()
        # bar_width = .95
        # # bar_width = pd.Timedelta(days=30)
        # ax1 = fig.add_subplot(gs[0, 0])
        # for i, var_name in enumerate(data_available.data_vars):
        #     freq = data_available[var_name].compute()
        #     if i==0:
        #         bot = np.zeros(freq.month.shape[0])
        #     p = ax1.bar(freq.month, 100*freq.values,
        #             label=var_name.replace('_', '-').title(),
        #             bottom=bot,
        #             color=colors[i],
        #             edgecolor="black",
        #             width=bar_width)  # Set the width of the bars
        #     bot += 100*freq.values
        # # Adding name of the months values
        # ax1.set_xticks(data_available.month)
        # ax1.set_ylabel("F$_{available}$ (%)")
        # # hide x-axis label
        # ax1.xaxis.set_tick_params(labelbottom=False)
        # ax1.spines['top'].set_visible(False)
        # ax1.spines['right'].set_visible(False)
        # ax1.set_ylim([0, 101])
        # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(data_available.data_vars)/2, frameon=False)
        # # -----------------------------------------------------------------------------------------------
        # # Grouping by month and plotting cloud frequency
        # # -----------------------------------------------------------------------------------------------
        # clouds_to_analyse  = ['Liquid', 'Mixed_phase', 'Ice', 'Pre_liquid', 'Pre_mixed_phase']
        # cloud_single_layer_freq     = mask_cloud_availability[clouds_to_analyse].groupby('time.month').mean()
        # cloud_single_layer_variance = cloud_single_layer_freq * (1 - cloud_single_layer_freq)
        # ax2 = fig.add_subplot(gs[1, 0], sharex=ax1) # Share the x-axis with the upper subplot
        # for i, var_name in enumerate(clouds_to_analyse):
        #     ax2.plot(cloud_single_layer_freq['month'], cloud_single_layer_freq[var_name]*100,
        #                 '-s',
        #                 linewidth=1,
        #                 label=f"{var_name.replace('_', '-').title()}",
        #                 markersize=6)
        # ax2.set_ylabel(r"F$_{single-layer}$ (%) ")
        # ax2.set_xlabel("Months")
        # ax2.grid(True)
        # ax2.legend()
        # ax2.set_xticks(np.arange(1, 13))
        # ax2.set_xlim([.5, 12.5])
        # # add name to months
        # fig.savefig(f"{PATH_FIG}cloud_frequency_availability_by_month.png", dpi=300, bbox_inches='tight')
        # plt.show()
    if plot_data_availability:
        # -----------------------------------------------------------------------------------------------
        # Grouping by month and plotting data availability for cloud layers
        # -----------------------------------------------------------------------------------------------
        data_available = (reindexed_clouds_layers == 1).groupby('month').mean()
        fig1 = plt.figure(figsize=(12, 7))  # Increase the size of the plot for better visibility
        gs1  = fig1.add_gridspec(1, 1)
        ax1  = fig1.add_subplot(gs1[0, 0])

        color_data_ava = ["#ababab","#ffffff"]
        label_data_ava = ["Available", "Missing"]
        bar_width = .95
        # bar_width = pd.Timedelta(days=30)
        for i in range(2):
            if i == 0:
                freq = 1 - data_available['missing'].compute()
                bot  = np.zeros(freq.month.shape[0])
                hatch="/"

            else:
                freq = data_available['missing'].compute()
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
        ax1.set_xticks(data_available.month)
        ax1.set_ylabel("F$_{available}$ (%)")
        ax1.set_xlabel("Months")
        # hide x-axis label
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax1.set_ylim([0, 101])
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(data_available.data_vars)/2, frameon=False)
        # add name to months
        fig1.savefig(f"{PATH_FIG}data_availability_by_month.png", dpi=300, bbox_inches='tight')
        plt.show()

    if plot_cloud_data_availability:
        # -----------------------------------------------------------------------------------------------
        # Grouping by month and plotting cloud frequency
        # -----------------------------------------------------------------------------------------------
        fig2 = plt.figure(figsize=(12, 6))  # Increase the size of the plot for better visibility
        gs2  = fig2.add_gridspec(1, 1)
        ax2 = fig2.add_subplot(gs2[0, 0])

        clouds_to_analyse       = ['Liquid', 'Mixed_phase', 'Ice', 'Pre_liquid', 'Pre_mixed_phase']
        cloud_single_layer_freq = mask_cloud_availability[clouds_to_analyse].groupby('time.month').mean()
        cloud_single_layer_variance = cloud_single_layer_freq * (1 - cloud_single_layer_freq)
        for i, var_name in enumerate(clouds_to_analyse):
            ax2.plot(cloud_single_layer_freq['month'], cloud_single_layer_freq[var_name]*100,
                '-o',
                linewidth=1,
                label=f"{var_name.replace('_', '-').title()}",
                markersize=6)
        ax2.set_ylabel(r"F$_{single-layer-clouds}$ (%) ")
        ax2.set_xlabel("Months")
        ax2.grid(True)
        ax2.legend()
        ax2.set_xticks(np.arange(1, 13))
        ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax2.set_xlim([.5, 12.5])
        # add name to months
        fig2.savefig(f"{PATH_FIG}cloud_single_layer_frequency_by_month.png", dpi=300, bbox_inches='tight')
        plt.show()

    if plot_daily_cloud_availability:
        # clouds_to_analyse       = ['Liquid', 'Mixed_phase', 'Ice', 'Pre_liquid', 'Pre_mixed_phase']
        clouds_to_analyse       = ['Mixed_phase', 'Ice']
        freq_time = 'D'
        delta_time = 30
        cloud_single_layer_freq     = mask_cloud_availability[clouds_to_analyse].resample(time=freq_time).mean()
        years = np.unique(mask_cloud_availability['time.year'])

        fig = plt.figure(figsize=(30, 25))  # Increase the size of the plot for better visibility
        gs  = fig.add_gridspec(len(years), 1, hspace=0.05)

        for i, var_name in enumerate(clouds_to_analyse):
            axs  = fig.add_subplot(gs[i, :])
            for j, year in enumerate(years):
                cloud_single_layer_freq_year = cloud_single_layer_freq[var_name].sel(time=cloud_single_layer_freq['time.year'] == year)
                if freq_time == 'D':
                    axs.plot(cloud_single_layer_freq_year.time.dt.dayofyear.values, cloud_single_layer_freq_year*100,
                        '-o',
                        linewidth=1,
                        label=f"{year}",
                        markersize=6)
                    axs.set_xticks(cloud_single_layer_freq_year.time.dt.dayofyear[::delta_time].values)
                    axs.set_xlim([cloud_single_layer_freq_year.time.dt.dayofyear.values[0], cloud_single_layer_freq_year.time.dt.dayofyear.values[-1]])
                if freq_time == 'W':
                    axs.plot(cloud_single_layer_freq_year.time.dt.isocalendar().week.values, cloud_single_layer_freq_year*100,
                        '-o',
                        linewidth=1,
                        label=f"{year}",
                        markersize=6)
                    axs.set_xticks(cloud_single_layer_freq_year.time.dt.isocalendar().week[::delta_time].values)
                    # set_trace()
                    axs.set_xlim([1, 52])
                if freq_time == 'M':
                    axs.plot(cloud_single_layer_freq_year.time.dt.month.values, cloud_single_layer_freq_year*100,
                        '-o',
                        linewidth=1,
                        label=f"{year}",
                        markersize=6)
                    axs.set_xticks(np.arange(1, 13))
                    axs.set_xlim([.5, 12.5])
            
            axs.set_ylabel(f"{var_name.replace('_', '-').title()} (%)")
            axs.grid(True)
            if i == 0:
                axs.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(years), frameon=False)
            if i == len(clouds_to_analyse)-1:
                axs.set_xticklabels(cloud_single_layer_freq_year.time[::delta_time].dt.strftime('%d/%m').values, rotation=45)
                axs.set_xlabel("Time")
            else:
                axs.xaxis.set_tick_params(labelbottom=False)
            # add name to months
            # axs.set_xticks(cloud_single_layer_freq_year.time[::delta_time].values)
            # axs.set_xticklabels(cloud_single_layer_freq_year.time[::delta_time].dt.strftime('%d/%m/%y'))
            # add name to months
        fig.savefig(f"{PATH_FIG}cloud_single_layer_frequency_by_{freq_time}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        fig2 = plt.figure(figsize=(18, 6.5))  # Increase the size of the plot for better visibility
        gs2  = fig2.add_gridspec(len(clouds_to_analyse), 1, hspace=0.05)
        for i, var_name in enumerate(clouds_to_analyse):
            axs  = fig2.add_subplot(gs2[i, :])
            cond = clouds_single_layer[var_name].compute() == 1
            data = integrated_var_single_layer.sel(time=cond).resample(time=freq_time).mean()    
            for j, year in enumerate(years):
                iwp_year = data['IWP'].sel(time=data['time.year'] == year)
                if freq_time == 'D':
                    axs.plot(iwp_year.time.dt.dayofyear.values, iwp_year,
                        '-o',
                        linewidth=2,
                        label=f"{year}",
                        markersize=6)
                    
                    axs.set_xticks(iwp_year.time.dt.dayofyear[::delta_time].values)
                    axs.set_xlim([iwp_year.time.dt.dayofyear.values[0], iwp_year.time.dt.dayofyear.values[-1]])
                
            axs.set_ylabel(f"{var_name.replace('_', '-').title()}"+r", g m$^{-2}$", fontsize=15)
            axs.grid(True)
            if i == 0:
                axs.legend(loc='upper center', ncol=len(years)//2, frameon=False, prop={'size': 14})
            
            if i == len(clouds_to_analyse)-1:
                axs.set_xticklabels(iwp_year.time[::delta_time].dt.strftime('%d/%m').values, rotation=30)
                axs.set_xlabel("Day/Month")
            else:
                axs.xaxis.set_tick_params(labelbottom=False)
            # add name to months
            # axs.set_xticks(cloud_single_layer_freq_year.time[::delta_time].values)
            # axs.set_xticklabels(cloud_single_layer_freq_year.time[::delta_time].dt.strftime('%d/%m/%y'))
            # add name to months
        fig2.savefig(f"{PATH_FIG}IWP_by_{freq_time}.png", dpi=300, bbox_inches='tight')
        plt.show()
        breakpoint()

    if plot_lwc_iwc_der_ier:
        # ylim_violin = [0, 70 ]
        # dreff =
        # set_trace()
        micro_var = 'lwc'
        integ_var = 'LWP'
        data              = microphysics_sigle[micro_var].compute() * 1000
        if integ_var == 'LWP':
            data1 = data.sum(dim='height_bins')*delta_range/1000
        else:
            data1 = integrated_var_single_layer[integ_var].compute()
        clouds_to_analyse = ['Ice', 'Liquid', 'Mixed_phase', 'Pre_liquid', 'Pre_mixed_phase']
        for var_name in clouds_to_analyse:
            print(f"Variable: {var_name}")
            cond                  = clouds_single_layer[var_name].compute() == 1
            microphysics_var_name = data.sel(time=cond)
            integrated_var_name   = data1.sel(time=cond)*1000

            # Rechunk the 'microphysics_var_name' array along the 'time' dimension into a single chunk
            # microphysics_var_name = microphysics_var_name.chunk({'time': -1})
            median_by_month = microphysics_var_name.dropna(dim='time', how = "all" ).groupby('month').quantile(.5, skipna=True)


            if np.sum(median_by_month) == 0:
                print(f"\nMonthly median is only zero for {var_name} clouds\n")
            mean_by_month = microphysics_var_name.dropna(dim='time', how="all").groupby('month').mean()
            std_by_month  = microphysics_var_name.dropna(dim='time', how="all").groupby('month').std()

            fig = plt.figure(figsize=(18, 9))
            gs = fig.add_gridspec(2, 3, width_ratios=[1.8, 5, .15], height_ratios=[3, 1.2], hspace=0.05, wspace=0.4)
            # -----------------------------------------------------------------------------------------------
            # Mesh plot (upper subplot)
            # -----------------------------------------------------------------------------------------------
            ax1 = fig.add_subplot(gs[0, 1])
            # max_value = np.nanmax(mean_by_month)

            mesh = ax1.pcolormesh( mean_by_month.month.values,  mean_by_month.height_bins/1000,
                        mean_by_month, shading='nearest', cmap='turbo', norm=mpl.colors.LogNorm(vmin=0.001, vmax=1))
            ax1.set_ylabel("Height (km) a.g.l.")
            ax1.set_xlabel("Time")
            ax1.set_title(f"{micro_var.upper()} for {var_name} Clouds")
            ax1.set_ylim([.0, np.max(mean_by_month.height_bins/1000)])
            ax1.grid(True)
            ax1.xaxis.set_visible(False)
            ax1.set_xticks(np.arange(1, 13))
            # -----------------------------------------------------------------------------------------------
            # Colorbar (right subplot)
            # -----------------------------------------------------------------------------------------------
            cax = fig.add_subplot(gs[0, 2])
            cbar1 = plt.colorbar(mesh, cax=cax, orientation='vertical', label="g m$^{-3}$", aspect=10)
            cbar1.ax.yaxis.set_label_position('left')
            # -----------------------------------------------------------------------------------------------
            # Mean profile by season (lower subplot)
            # -----------------------------------------------------------------------------------------------
            ax2 = fig.add_subplot(gs[:, 0])

            mean_by_season = microphysics_var_name.dropna(dim='time', how = "all" ).groupby('season').mean()
            std_by_season = microphysics_var_name.dropna(dim='time', how = "all" ).groupby('season').std()/10
            mean_std_by_season = std_by_season/np.sqrt(microphysics_var_name.dropna(dim='time', how = "all" ).groupby('season').count(dim='time'))

            median_by_season = microphysics_var_name.dropna(dim='time', how = "all" ).groupby('season').quantile(dim='time', q=.5, skipna=True)
            if np.sum(median_by_season) == 0:
                 print(f"\nSeasonal median is only zero for {var_name} clouds\n")
            # inf_quantile = microphysics_var_name.groupby('season').quantile(dim='time', q=.25, skipna=True)
            # sup_quantile = microphysics_var_name.groupby('season').quantile(dim='time', q=.75, skipna=True)

            max_value = np.nanmax(mean_by_season)

            for season in mean_by_season.season.values:
                mean_profile = mean_by_season.sel(season=season)
                std_profile = mean_std_by_season.sel(season=season)
                # qinf_season = inf_quantile.sel(season=season)
                # qsup_season = sup_quantile.sel(season=season)

                ax2.plot(mean_profile, mean_profile.height_bins/1000, label=f"{season}")
                # ax2.fill_betweenx(mean_profile.height_bins/1000, qinf_season, qsup_season, alpha=0.3)
                ax2.fill_betweenx(mean_profile.height_bins/1000, mean_profile - std_profile, mean_profile + std_profile, alpha=0.3)
            ax2.set_xlabel(f"{micro_var.upper()}, g m" + r"$^{-3}$ ($\sigma$/10)")
            ax2.set_ylabel("Height (km) a.g.l.")
            ax2.set_xlim([0, max_value + .1])
            ax2.set_ylim([0, np.max(mean_by_season.height_bins/1000)])
            ax2.grid(True)
            ax2.legend()
            # -----------------------------------------------------------------------------------------------
            # Violin plot for size distribution for each month
            # -----------------------------------------------------------------------------------------------

            ax3 = fig.add_subplot(gs[1,1], sharex=ax1)  # Share the x-axis with ax1
            for month in mean_by_month.month.values:
                mask_month = integrated_var_name['month'] == month
                month_data = integrated_var_name.sel(time=mask_month).values
                # month_data =
                # removing nans from month data
                month_data = month_data[(~np.isnan(month_data)) & (month_data >= 0)]
                # mean_profile = monthly_data.sel(month=month).dropna(dim='height_bins')
                # ax3.violinplot(month_data, positions=[month], showmeans=False, showmedians=True, showextrema=False)
                ax3.boxplot(month_data, positions=[month], showfliers=False, showmeans=True, patch_artist=True, widths=0.8,
                             meanprops=dict(marker='*', markerfacecolor='black', markeredgecolor='black'),
                             medianprops=dict(color='red', linewidth=1.5), boxprops=dict(facecolor='lightblue', color='black'))

            ax3.set_ylabel(f"{integ_var} (g m" + r"$^{-2}$)")
            ax3.set_xlabel("Months")
            ax3.set_xticks(np.arange(1, 13))
            ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            # if var_name == 'Pre_liquid' or var_name == 'Pre_mixed_phase':
            #     ax3.set_ylim([0, 1000])
            # else:
            #     ax3.set_ylim([0, 300])
            # ax3.set_yticks(np.arange(0, np.max(microphysics_var_name), dreff))
            # ax3.set_ylim(ylim_violin)
            ax3.grid(True, axis='y')

            number_profiles = microphysics_var_name.month.groupby('month').count(dim='time')
            ax3_right = ax3.twinx()
            ax3_right.plot(number_profiles.month, number_profiles.values, '--', linewidth=1, color='blue')  # Set zorder to 0
            ax3_right.set_ylabel(r"N$_{Profiles}$", color="blue")
            ax3_right.tick_params(axis='y', colors='blue')
            # ax3_right.set_yticks(np.arange(np.min(number_profiles), np.max(number_profiles), 1000))

            ax3.spines['top'].set_visible(False)  # Remove the top spine
            ax3_right.spines['top'].set_visible(False)  # Remove the top spine
            ax3_right.spines['right'].set_color('blue')  # Set the color of the right spine to blue

            # # -----------------------------------------------------------------------------------------------
            # # LWP and IWP evolution
            # # -----------------------------------------------------------------------------------------------
            # ax3 = fig.add_subplot(gs[1,1], sharex=ax1)
            # ax3.spines['top'].set_visible(False)  # Remove the top spine

            # # grouped_by_month = integrated_var_name.groupby('time.month').mean()
            # number_profiles = integrated_var_name.groupby('month').count()
            # # std_by_month = integrated_var_name.groupby('time.month').std()
            # # ax3.errorbar(grouped_by_month.month, grouped_by_month, yerr=std_by_month, fmt='--s', capsize=5, capthick=2, color='black')

            # # ax3_right = ax3.twinx()
            # # ax3_right.plot(number_profiles.month, number_profiles, 'r--o')
            # # ax3_right.set_ylabel("N Profiles", color="red")
            # # ax3_right.tick_params(axis='y', colors='red')
            # # ax3_right.set_yticks(np.linspace(np.min(number_profiles), np.max(number_profiles), 5))

            # ax3.plot(number_profiles.month, number_profiles, 'r--o', linewidth=1, markersize=5)
            # ax3.set_ylabel("N Profiles")
            # ax3.set_yticks(np.linspace(np.min(number_profiles), np.max(number_profiles), 5))
            # ax3.grid(True)

            # # ax3.set_ylabel(r"LWP g m$^{-3}$")
            # ax3.set_xlabel("Month")
            # ax3.set_xticks(np.arange(1, 13))
            # ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            # # ax3.set_yscale('log')
            fig.savefig(f"{PATH_FIG}grouped_by_month_evolution_for_{micro_var}_{var_name}.png", dpi=300, bbox_inches='tight')
            plt.show()
        set_trace()

        micro_var = 'der'
        data = microphysics_sigle[micro_var].compute() * 1e6
        # -----------------------------------------------------------------------------------------------
        norm_color = [0, 15] # for der
        ylim_violin = np.array([0, 31 ]) # for der
        dreff = 10
        # -----------------------------------------------------------------------------------------------
        # norm_color = [0, 70] # for ier
        # ylim_violin = np.array([0, 70 ]) # for ier
        # dreff = 20
        # -----------------------------------------------------------------------------------------------
        # clouds_to_analyse = ['Mixed_phase', 'Ice', 'Pre_mixed_phase']
        clouds_to_analyse = ['Liquid', 'Mixed_phase', 'Ice', 'Pre_liquid', 'Pre_mixed_phase']
        for var_name in clouds_to_analyse:
            print(f"Variable: {var_name}")
            cond                  = clouds_single_layer[var_name].compute() == 1
            microphysics_var_name = data.sel(time=cond)
            # Rechunk the 'microphysics_var_name' array along the 'time' dimension into a single chunk
            # microphysics_var_name = microphysics_var_name.chunk({'time': -1})
            # monthly_data = microphysics_var_name.groupby('month').quantile(.5)
            monthly_data = microphysics_var_name.groupby('month').quantile(.5, skipna=True)

            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 3, width_ratios=[1.8, 6, .15], height_ratios=[3, 1.2], hspace=0.05, wspace=0.3)
            # -----------------------------------------------------------------------------------------------
            # Mesh plot (upper subplot)
            # -----------------------------------------------------------------------------------------------
            ax1 = fig.add_subplot(gs[0, 1])

            max_value = np.nanmax(monthly_data)
            cbar_max =  max_value
            if var_name == 'Liquid':
                mesh = ax1.pcolormesh(monthly_data.month.values, monthly_data.height_bins/1000,
                            monthly_data, shading='nearest', cmap='jet', vmin=norm_color[0], vmax=norm_color[1])
            else:
                mesh = ax1.pcolormesh(monthly_data.month.values, monthly_data.height_bins/1000,
                        monthly_data, shading='nearest', cmap='jet')
            ax1.set_ylabel("Height [km] a.g.l.")
            ax1.set_xlabel("Time")
            ax1.set_title(f"Droplet effective radius for {var_name} clouds")
            ax1.set_ylim([.0, 15])
            ax1.grid(True)
            ax1.xaxis.set_visible(False)
            ax1.set_xticks(np.arange(1, 13))
            # -----------------------------------------------------------------------------------------------
            # Colorbar (right subplot)
            # -----------------------------------------------------------------------------------------------
            cax = fig.add_subplot(gs[0, 2])
            cbar1 = plt.colorbar(mesh, cax=cax, orientation='vertical', label=r"$\tilde{R_{eff}}$ ($\mu$m)", aspect=10)
            cbar1.ax.yaxis.set_label_position('left')
            # -----------------------------------------------------------------------------------------------
            # Mean profile by season (lower subplot)
            # -----------------------------------------------------------------------------------------------
            ax2 = fig.add_subplot(gs[:, 0])

            mean_profile_by_season = microphysics_var_name.groupby('season').quantile(dim='time', q=.5, skipna=True)
            std_profile_by_season = microphysics_var_name.groupby('season').std(dim='time')

            inf_quantile = microphysics_var_name.groupby('season').quantile(dim='time', q=.25, skipna=True)
            sup_quantile = microphysics_var_name.groupby('season').quantile(dim='time', q=.75, skipna=True)

            max_value = np.nanmax(mean_profile_by_season)

            for season in mean_profile_by_season.season.values:
                mean_profile = mean_profile_by_season.sel(season=season)
                std_profile = std_profile_by_season.sel(season=season)
                qinf_season = inf_quantile.sel(season=season)
                qsup_season = sup_quantile.sel(season=season)
                ax2.plot(mean_profile, mean_profile.height_bins/1000, '--',label=f"{season.title()}")
                # ax2.fill_betweenx(mean_profile.height_bins/1000, mean_profile - std_profile, mean_profile + std_profile, alpha=0.3)
                ax2.fill_betweenx(mean_profile.height_bins/1000, qinf_season, qsup_season, alpha=0.3)
            ax2.set_xlabel(r"$\tilde{R_{eff}}$ ($\mu$m)")
            ax2.set_ylabel("Height (km) a.g.l.")
            ax2.set_xlim([np.nanmin(mean_profile_by_season), np.nanmax(mean_profile_by_season)])
            ax2.set_xticks(np.arange(0, np.max(mean_profile_by_season), dreff))
            ax2.set_ylim([.0, 15])
            ax2.grid(True)
            ax2.legend()
            # -----------------------------------------------------------------------------------------------
            # Violin plot for size distribution for each month
            # -----------------------------------------------------------------------------------------------
            ax3 = fig.add_subplot(gs[1,1], sharex=ax1)  # Share the x-axis with ax1
            for month in monthly_data.month.values:
                mask_month = microphysics_var_name['month'] == month
                month_data = microphysics_var_name.sel(time=mask_month).values.ravel()
                # removing nans from month data
                month_data = month_data[~np.isnan(month_data)]
                # mean_profile = monthly_data.sel(month=month).dropna(dim='height_bins')
                ax3.violinplot(month_data, positions=[month], 
                               showmeans=False, showmedians=True, showextrema=False)

            ax3.set_ylabel(r"R$_{eff}$ ($\mu$m)")
            ax3.set_xlabel("Months")
            ax3.set_xticks(np.arange(1, 13))
            ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            # ax3.set_yticks(np.arange(0, np.max(microphysics_var_name), dreff))
            if var_name == 'Liquid':
                ax3.set_ylim([0, 30])
            elif var_name == 'Pre_liquid':
                ax3.set_ylim([0, 40])
            else:
                ax3.set_ylim(ylim_violin*2)
            ax3.grid(True, axis='y')

            number_profiles = microphysics_var_name.month.groupby('month').count(dim='time')
            ax3_right = ax3.twinx()
            ax3_right.plot(number_profiles.month, number_profiles.values, '--s', linewidth=1, color='blue')  # Set zorder to 0
            ax3_right.set_ylabel(r"N$_{Profiles}$", color="blue")
            ax3_right.tick_params(axis='y', colors='blue')
            # ax3_right.set_yticks(np.linspace(np.min(number_profiles), np.max(number_profiles), 5))

            ax3.spines['top'].set_visible(False)  # Remove the top spine
            ax3_right.spines['top'].set_visible(False)  # Remove the top spine
            ax3_right.spines['right'].set_color('blue')  # Set the color of the right spine to blue
            fig.savefig(f"{PATH_FIG}grouped_by_month_evolution_for_{micro_var}_{var_name}.png", dpi=300, bbox_inches='tight')
            plt.show()
    # -----------------------------------------------------------------------------------------------
    target_parent_folder = "geometric_cloud_thickness"
    file_extension = '.json'
    chirp_cloud_thickness = reading_dataset_chunking(root_folder, target_parent_folder, file_extension=file_extension)
    target_parent_folder = "height_cloud_base"
    file_extension = '.json'
    chirp_height_cb = reading_dataset_chunking(root_folder, target_parent_folder, file_extension=file_extension)
    target_parent_folder = "height_cloud_top"
    file_extension = '.json'

    chirp_cloud_thickness_list = [ds for ds in chirp_cloud_thickness.values()]
    cloud_thickness = xr.concat(chirp_cloud_thickness_list, dim='time').sortby('time')
    cloud_thickness = cloud_thickness.assign_coords(years=cloud_thickness['time'].dt.year, month=cloud_thickness['time'].dt.month)
    cloud_thickness = assign_season(cloud_thickness, SEASONS)
    cloud_thickness_single = cloud_thickness.sel(time=mask_single)

    chirp_height_cloud_base_list = [ds for ds in chirp_height_cb.values()]
    height_cloud_base = xr.concat(chirp_height_cloud_base_list, dim='time').sortby('time')
    height_cloud_base = height_cloud_base.assign_coords(years=height_cloud_base['time'].dt.year, month=height_cloud_base['time'].dt.month)
    height_cloud_base = assign_season(height_cloud_base, SEASONS)
    height_cloud_base_single = height_cloud_base.sel(time=mask_single)

    chirp_height_cloud_top = reading_dataset_chunking(root_folder, target_parent_folder, file_extension=file_extension)
    chirp_height_cloud_top_list = [ds for ds in chirp_height_cloud_top.values()]
    height_cloud_top = xr.concat(chirp_height_cloud_top_list, dim='time').sortby('time')
    height_cloud_top = height_cloud_top.assign_coords(years=height_cloud_top['time'].dt.year, month=height_cloud_top['time'].dt.month)
    height_cloud_top = assign_season(height_cloud_top, SEASONS)
    height_cloud_top_single = height_cloud_top.sel(time=mask_single)

    if plot_integrated_water_and_ice_path:
        # -----------------------------------------------------------------------------------------------
        # LWP and IWP for single layer clouds analysis
        # -----------------------------------------------------------------------------------------------
        bin_width=250
        for integ_var in integrated_var_single_layer.data_vars:
            print(f"Variable: {integ_var}")
            fig = plt.figure(figsize=(20, 9))
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1], hspace=0.05, wspace=0.2)

            if integ_var == 'LWP':
                clouds_to_analyse = ['Pre_liquid', 'Pre_mixed_phase']
            else:
                clouds_to_analyse = ['Ice', 'Mixed_phase']

            ax1 = fig.add_subplot(gs[:, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
            for var_name in clouds_to_analyse:
                # -----------------------------------------------------------------------------------------------
                # Montlhy median of LWP and IWP for single layer clouds for each cloud type
                # -----------------------------------------------------------------------------------------------
                cond                    = clouds_single_layer[var_name].compute() == 1
                integrated_var_name     = integrated_var_single_layer.sel(time=cond).compute()
                new_integrated_var_name = integrated_var_name.dropna(dim='time')*1000

                max_value = np.nanmax(new_integrated_var_name[integ_var])
                bin_edges = np.arange(0, max_value + bin_width, bin_width)
                # Calculate histograms for each dataset
                histogram = np.histogram(new_integrated_var_name[integ_var], bins=bin_edges)[0]

                # Get the number of profiles for each dataset
                num_profiles = new_integrated_var_name[integ_var].sizes["time"]

                bin_widths = bin_edges[1] - bin_edges[0]
                norm_histogram = 100 * (histogram / num_profiles)

                sigma_n = np.sqrt(histogram)
                sigma_f = 100 * (sigma_n/num_profiles)

                ax1.hist(bin_edges[:-1], bins=bin_edges,
                            weights=norm_histogram,
                            histtype='step',
                            label=f"{var_name.replace('_', '-').title()} ({num_profiles} profiles)",
                            linewidth=2.0)
                ax1.errorbar(bin_edges[:-1] + bin_widths/2, norm_histogram, yerr=sigma_f, fmt='none', capsize=2, color='black')

                # -----------------------------------------------------------------------------------------------
                # Montlhy median of LWP and IWP for single layer clouds for each cloud type
                # -----------------------------------------------------------------------------------------------

                monthly_data = new_integrated_var_name[integ_var].groupby('month').quantile(.5)
                inf_quantile = new_integrated_var_name[integ_var].groupby('month').quantile(.25)
                sup_quantile = new_integrated_var_name[integ_var].groupby('month').quantile(.75)

                scat_lwc = ax2.plot(monthly_data['month'], monthly_data.values, linestyle='--', marker='s', markersize=5, label=var_name.replace('_', '-').title())
                ax2.fill_between(monthly_data['month'], inf_quantile.values, sup_quantile.values, alpha=0.3)

                # -----------------------------------------------------------------------------------------------
                # Data availability for LWP and IWP for each cloud type
                # -----------------------------------------------------------------------------------------------
                number_profiles       = integrated_var_name[integ_var].groupby('time.month').count()
                data_complete         = reindex_datasets(integrated_var_name[integ_var])
                number_profiles_total = data_complete.time.dt.month.groupby('time.month').count().values

                freq = 100 * number_profiles / number_profiles_total

                scat_freq = ax3.plot(number_profiles['month'], freq, linestyle='--', marker='s', markersize=5)

            ax1.set_yscale('log')
            ax1.legend()
            ax1.grid()
            ax1.set_xticks(np.arange(min(bin_edges), max(bin_edges) + bin_width+10, bin_width+10))
            ax1.set_xlim([0, 180])
            # ax1.set_ylim([0.1, 70])
            # ax1.tick_params(axis='x', rotation=40)
            ax1.set_xlabel(f"{integ_var}"+r" [g m$^{-2}$]")
            ax1.set_ylabel("Frequency [%]")

            ax2.set_ylabel(f"{integ_var}"+r" [g m$^{-2}$]")
            ax2.grid(True)
            ax2.legend()
            ax2.set_ylim([.0, 180])
            ax2.set_yticks(np.arange(0, 180+bin_width, bin_width))
            ax2.xaxis.set_visible(False)

            ax3.set_ylabel("F$_{occurence}$")
            ax3.set_xlabel("Month")
            ax3.grid(True)
            ax3.set_xticks(np.arange(1, 13))
            ax3.set_xlim([1, 12])
            ax3.spines['top'].set_visible(False)  # Remove the top spine
            fig.savefig(f"{PATH_FIG}lwp_iwp_single_layer_{integ_var}_w_precipitation.png", dpi=300, bbox_inches='tight')
            plt.show()
        set_trace()
    # # -----------------------------------------------------------------------------------------------
    # # Comaring LWP and IWP for single layer clouds analysis
    # # -----------------------------------------------------------------------------------------------
    # fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))  # 2 rows, 1 column
    # for var_name in clouds_single_layer.data_vars:

    #     if var_name == 'Liquid' or var_name == 'Mixed_phase' or var_name == 'Ice':
    #         print(f"Variable: {var_name}")
    #         cond                             = clouds_single_layer[var_name].compute() == 1
    #         new_integrated_var_name          = integrated_var_single_layer.sel(time=cond).compute().dropna(dim='time')

    #         new_integrated_var_name['month'] = new_integrated_var_name['time'].dt.month
    #         monthly_data = new_integrated_var_name.groupby('month').quantile(.5)

    #         inf_quantile = new_integrated_var_name.groupby('month').quantile(.25)
    #         sup_quantile = new_integrated_var_name.groupby('month').quantile(.75)

    #         scat_lwc = ax[0].plot(monthly_data['month'], monthly_data['LWP'], linestyle='--', marker='o', markersize=5, label=var_name)
    #         ax[0].fill_between(monthly_data['month'], inf_quantile['LWP'], sup_quantile['LWP'], alpha=0.3)
    #         ax[0].set_ylabel(r"LWP [kg m$^{-2}$]")
    #         ax[0].grid(True)
    #         ax[0].legend()

    #         scat_iwc = ax[1].plot(monthly_data['month'], monthly_data['IWP'], linestyle='--', marker='o', markersize=5)
    #         ax[1].fill_between(monthly_data['month'], inf_quantile['IWP'], sup_quantile['IWP'], alpha=0.3)
    #         ax[1].set_ylabel(r"IWP [kg m$^{-2}$]")
    #         ax[1].set_xlabel("Month")
    #         ax[1].grid(True)
    #         ax[1].set_ylim([-0.008, 0.25])
    #         ax[1].set_xticks(np.arange(1, 13))
    #         # ax[1].set_yscale('log')
    # fig.savefig(f"{PATH_FIG}lwp_iwp_single_layer.png", dpi=300)
    # plt.show()

    test_mask = clouds_single_layer.compute() == 1
    new_cloud_thickness_var = cloud_thickness_single.where(test_mask).compute().astype(float)
    new_height_cloud_base    = height_cloud_base_single.where(test_mask).compute().astype(float)
    new_height_cloud_top     = height_cloud_top_single.where(test_mask).compute().astype(float)

    # Create a DataFrame with only the data variables
    df_cloud_thickness = (new_cloud_thickness_var/1000).to_dataframe().reset_index() # convert from m to km
    df_height_cloud_base = (new_height_cloud_base/1000).to_dataframe().reset_index()
    df_height_cloud_top = (new_height_cloud_top/1000).to_dataframe().reset_index()

    # Melt the DataFrame to long format for easier plotting
    # melted_cloud_thickness = pd.melt(df_cloud_thickness.drop(columns=['time', 'years', 'month', 'Pre_liquid', 'Pre_mixed_phase']), id_vars=['season'], var_name='cloud_type', value_name='cloud_thickness')
    # melted_cloud_base = pd.melt(df_height_cloud_base.drop(columns=['time', 'years', 'month', 'Pre_liquid', 'Pre_mixed_phase']), id_vars=['season'], var_name='cloud_type', value_name='cloud_base')
    # melted_cloud_top = pd.melt(df_height_cloud_top.drop(columns=['time', 'years', 'month', 'Pre_liquid', 'Pre_mixed_phase']), id_vars=['season'], var_name='cloud_type', value_name='cloud_top')

    melted_cloud_thickness = pd.melt(df_cloud_thickness.drop(columns=['time', 'years', 'month']), id_vars=['season'], var_name='cloud_type', value_name='cloud_thickness')
    melted_cloud_base = pd.melt(df_height_cloud_base.drop(columns=['time', 'years', 'month']), id_vars=['season'], var_name='cloud_type', value_name='cloud_base')
    melted_cloud_top = pd.melt(df_height_cloud_top.drop(columns=['time', 'years', 'month']), id_vars=['season'], var_name='cloud_type', value_name='cloud_top')

    n_profiles        = new_cloud_thickness_var.groupby('season').count(dim='time')
    n_profiles        = n_profiles.to_dataframe().reset_index()
    melted_n_profiles = pd.melt(n_profiles.drop(columns=['Pre_liquid', 'Pre_mixed_phase']), id_vars=['season'], var_name='cloud_type', value_name='N_Profiles')

    fig = plt.figure(figsize=(25, 11))
    # gs = fig.add_gridspec(4, 1, height_ratios=[2, 2, 2, 1.5], hspace=0.05)
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 2], hspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.violinplot(x='season', y='cloud_thickness', hue='cloud_type', quantiles=[0.25, 0.5, 0.75], data=melted_cloud_thickness, ax=ax1, inner_kws=dict(box_width=7, whis_width=.2), scale='width')
    ax1.set_ylim([-1, 10])
    ax1.set_ylabel(r"$\Delta$Z (km)")
    ax1.spines['top'].set_visible(False)  # Remove the top spine
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.set_visible(False)
    ax1.set_yticks(np.arange(0, 10, 2))
    ax1.legend().set_visible(True)  # Show the legend for the first violin plot
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)  # Place the legend above the figure on the horizontal
    ax1.grid(True)

    median_values_cloud_thickness = melted_cloud_thickness.groupby(['season', 'cloud_type'])['cloud_thickness'].median().reset_index()
    percentile25_values_cloud_thickness = melted_cloud_thickness.groupby(['season', 'cloud_type'])['cloud_thickness'].quantile(0.25).reset_index()
    percentile75_values_cloud_thickness = melted_cloud_thickness.groupby(['season', 'cloud_type'])['cloud_thickness'].quantile(0.75).reset_index()
    df_cloud_thickness_stats = pd.merge(median_values_cloud_thickness, percentile25_values_cloud_thickness, on=['season', 'cloud_type'], suffixes=('_median', '_25_percentile'))
    df_cloud_thickness_stats = pd.merge(df_cloud_thickness_stats, percentile75_values_cloud_thickness, on=['season', 'cloud_type'], suffixes=('', '_75_percentile'))
    df_cloud_thickness_stats.to_csv('/home/matheustolen/Documentos/matheus_doctorado/processed_data/files_to_analyse/df_cloud_thickness.txt', sep='\t', index=False, decimal=',')

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    sns.violinplot(x='season', y='cloud_base', hue='cloud_type', quantiles=[0.25, 0.5, 0.75], data=melted_cloud_base, ax=ax2, inner_kws=dict(box_width=7, whis_width=.2), scale='width')
    # ax2.set_ylim([-1, 10])
    ax2.set_ylabel(r"Cloud Base (km)")
    ax2.spines['top'].set_visible(False)  # Remove the top spine
    ax2.spines['right'].set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.legend().set_visible(False)  # Hide the legend for the second violin plot
    ax2.set_yticks(np.arange(0, 14, 2))
    ax2.grid(True)

    median_values_cloud_base = melted_cloud_base.groupby(['season', 'cloud_type'])['cloud_base'].median().reset_index()
    percentile25_values_cloud_base = melted_cloud_base.groupby(['season', 'cloud_type'])['cloud_base'].quantile(0.25).reset_index()
    percentile75_values_cloud_base = melted_cloud_base.groupby(['season', 'cloud_type'])['cloud_base'].quantile(0.75).reset_index()
    df_cloud_base_stats = pd.merge(median_values_cloud_base, percentile25_values_cloud_base, on=['season', 'cloud_type'], suffixes=('_median', '_25_percentile'))
    df_cloud_base_stats = pd.merge(df_cloud_base_stats, percentile75_values_cloud_base, on=['season', 'cloud_type'], suffixes=('', '_75_percentile'))
    df_cloud_base_stats.to_csv('/home/matheustolen/Documentos/matheus_doctorado/processed_data/files_to_analyse/df_cloud_base.txt', sep='\t', index=False, decimal=',')

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1, sharey=ax2)
    sns.violinplot(x='season', y='cloud_top', hue='cloud_type', quantiles=[0.25, 0.5, 0.75], data=melted_cloud_top, ax=ax3, inner_kws=dict(box_width=7, whis_width=.2), scale='width')
    # ax3.set_ylim([-1, 10])
    ax3.set_ylabel(r"Cloud Top (km)")
    ax3.spines['top'].set_visible(False)  # Remove the top spine
    ax3.spines['right'].set_visible(False)
    ax3.set_xlabel("Seasons")
    ax3.legend().set_visible(False)  # Hide the legend for the third violin plot
    ax3.grid(True)

    median_values_cloud_top = melted_cloud_top.groupby(['season', 'cloud_type'])['cloud_top'].median().reset_index()
    percentile25_values_cloud_top = melted_cloud_top.groupby(['season', 'cloud_type'])['cloud_top'].quantile(0.25).reset_index()
    percentile75_values_cloud_top = melted_cloud_top.groupby(['season', 'cloud_type'])['cloud_top'].quantile(0.75).reset_index()
    df_cloud_top_stats = pd.merge(median_values_cloud_top, percentile25_values_cloud_top, on=['season', 'cloud_type'], suffixes=('_median', '_25_percentile'))
    df_cloud_top_stats = pd.merge(df_cloud_top_stats, percentile75_values_cloud_top, on=['season', 'cloud_type'], suffixes=('', '_75_percentile'))
    df_cloud_top_stats.to_csv('/home/matheustolen/Documentos/matheus_doctorado/processed_data/files_to_analyse/df_cloud_top.txt', sep='\t', index=False, decimal=',')


    # ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    # sns.lineplot(x='season', y='N_Profiles', hue='cloud_type', data=melted_n_profiles, ax=ax4, marker='o',style='cloud_type', markersize=10)
    # ax4.set_ylabel("N$_{Profiles}$")
    # ax4.spines['top'].set_visible(False)  # Remove the top spine
    # ax4.spines['right'].set_visible(False)
    # ax4.legend().set_visible(False)
    # ax4.grid(True)
    fig.savefig(f"{PATH_FIG}cloud_thickness_base_top_violin_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    set_trace()
    # -----------------------------------------------------------------------------------------------
    # Repeting above clou for the cloud thickness
    # -----------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))  # 2 rows, 1 column
    clouds_to_analyse = ['Liquid', 'Mixed_phase', 'Ice']
    for var_name in clouds_to_analyse:
        print(f"Variable: {var_name}")
        cond                             = clouds_single_layer[var_name].compute() == 1
        new_cloud_thickness_var          = cloud_thickness_single[var_name].sel(time=cond).compute().dropna(dim='time').astype(float)

        monthly_data = new_cloud_thickness_var.groupby('month').quantile(.5)
        inf_quantile = new_cloud_thickness_var.groupby('month').quantile(.25)
        sup_quantile = new_cloud_thickness_var.groupby('month').quantile(.75)

        scat_lwc = ax.plot(monthly_data['month'], monthly_data.values, linestyle='--', marker='o', markersize=5, label=var_name.replace('_', '-').title())
        ax.fill_between(monthly_data['month'].values, inf_quantile.values, sup_quantile.values, alpha=0.3)
        ax.set_ylabel(r"$\Delta$Z (m)")
        ax.grid(True)
        ax.legend()
        # ax[1].set_yscale('log')
    fig.savefig(f"{PATH_FIG}comparison_dz_clouds_wo_precipitation", dpi=300)
    plt.show()

    # -----------------------------------------------------------------------------------------------
    # plot_cloud_frequency2(reindexed_clouds_layers,
    #                      freq_str,
    #                      figname=f"{PATH_FIG}cloud_frequency_subplots_by_year.png")

    # plot_cloud_frequency3(reindexed_clouds_layers,
    #                      freq_str,
    #                      figname=f"{PATH_FIG}cloud_freq_plot_by_year")

    # plot_histograms_with_profiles(datasets=[integrated_var_single_layer.LWP.where(clouds_single_layer.Liquid.compute() == 1, drop=True).dropna(dim='time'),
    #                                         integrated_var_single_layer.LWP.where(clouds_single_layer.Mixed_phase.compute() == 1, drop=True).dropna(dim='time')],
    #                              bin_width=.025,
    #                              labels=["Liquid", "Mixed-phase"],
    #                              x_label= r"LWP ($\Delta$LWP = .025 [kg m$^{-2}$])",
    #                              y_label="Frequency [%]",
    #                              xticks_resolution=.050,
    #                              xlim=[0, .5],
    #                              ylim=(0.1, 100),
    #                              figname=f"{PATH_FIG}lwp_histograms.png")

    # plot_histograms_with_profiles(datasets=[integrated_var_single_layer.IWP.where(clouds_single_layer.Ice.compute() == 1, drop=True).dropna(dim='time'),
    #                                         integrated_var_single_layer.IWP.where(clouds_single_layer.Mixed_phase.compute() == 1, drop=True).dropna(dim='time')],
    #                              bin_width=.025,
    #                              labels=["Ice", "Mixed-phase"],
    #                              x_label= r"IWP ($\Delta$IWP = .025 kg m$^{-2}$)",
    #                              y_label="Frequency [%]",
    #                              xticks_resolution=.050,
    #                              xlim=[0, 1.],
    #                              ylim=(0.1, 100),
    #                              figname=f"{PATH_FIG}ice_histograms.png")

    # plot_histograms_with_profiles(datasets=[cloud_thickness.Liquid.where(clouds_single_layer.Liquid.compute() == 1, drop=True).dropna(dim='time'),
    #                                         cloud_thickness.Ice.where(clouds_single_layer.Ice.compute() == 1, drop=True).dropna(dim='time'),
    #                                         cloud_thickness.Mixed_phase.where(clouds_single_layer.Mixed_phase.compute() == 1, drop=True).dropna(dim='time')],
    #                              bin_width=200,
    #                              labels=["Liquid", "Ice", "Mixed-phase"],
    #                              x_label= r"Cloud Thickness ($\Delta$Z = 200 m)",
    #                              y_label="Frequency [%]",
    #                              xticks_resolution=500,
    #                              ylim=(0.1, 100),
    #                              figname=f"{PATH_FIG}cloud_thickness_histograms.png")

    # -----------------------------------------------------------------------------------------------
    # Cloud Geometric Properties Analysis
    # ----------------------------------------------------------------------------------------------_
if cloud_macrophysics_analysis:
    print(" Starting cloud macrophysics analysis...")
    target_parent_folder = "mwr_profiles"
    file_extension = '.nc'
    mwr_profiles   = reading_dataset_chunking(root_folder, target_parent_folder, file_extension=file_extension)['processed_data']
    # time_grid      = cloud_layers.time.values
    # mwr_rebined    = mwr_profiles.compute().groupby_bins('time', time_grid, labels=time_grid[1:]).mean().dropna(dim='time', how='all')

    # reindexing mwr_profiles in the same time grid as cloud_layers:
    mwr_reindexed = reindex_datasets(mwr_profiles, method='nearest', tolerance='2min').dropna(dim='time', how='all')
    # # -----------------------------------------------------------------------------------------------
    # # Testing interpolation
    # # -----------------------------------------------------------------------------------------------
    # start_date    = "2023-01-01"
    # end_date      = "2023-12-30"
    # # sliced_mwr_pr  = mwr_profiles.sel(time=slice(start_date, end_date))
    # sliced_mwr_pr  = mwr_reindexed.sel(time=slice(start_date, end_date))
    # resampled_mwr  = sliced_mwr_pr.resample(time="M").mean()

    # mwr_var_to_analise = ['relative_humidity','humidity','temperature']
    # for var_name in sliced_mwr_pr.data_vars:
    #     if var_name in mwr_var_to_analise:
    #         print(f"Variable: {var_name}")
    #         fig, ax = plt.subplots(figsize=(10, 7))  # 2 rows, 1 column
    #         mesh = ax.pcolormesh(resampled_mwr.time, resampled_mwr.altitude, resampled_mwr[var_name].T, cmap='rainbow')
    #         ax.set_ylabel("Height [m] a.g.l")
    #         ax.set_title("MWR Profiles")
    #         fig.colorbar(mesh, ax=ax, label=f"{var_name} [kg m$^{-2}$]")
    #         ax.grid(True)
    #         plt.show()
    # -----------------------------------------------------------------------------------------------
    target_parent_folder = "height_cloud_base"
    file_extension = '.json'
    chirp_height_cb = reading_dataset_chunking(root_folder, target_parent_folder, file_extension=file_extension)
    target_parent_folder = "height_cloud_top"
    file_extension = '.json'
    chirp_height_ct = reading_dataset_chunking(root_folder, target_parent_folder, file_extension=file_extension)
    target_parent_folder = "geometric_cloud_thickness"
    file_extension = '.json'
    chirp_height_cg = reading_dataset_chunking(root_folder, target_parent_folder, file_extension=file_extension)

    freq_str = "M"

    chirp_height_cb_list = [ds for ds in chirp_height_cb.values()]
    cloud_prop_cb = xr.concat(chirp_height_cb_list, dim='time').sortby('time')

    chirp_height_ct_list = [ds for ds in chirp_height_ct.values()]
    cloud_prop_ct = xr.concat(chirp_height_ct_list, dim='time').sortby('time')

    chirp_height_cg_list = [ds for ds in chirp_height_cg.values()]
    cloud_prop_cg = xr.concat(chirp_height_cg_list, dim='time').sortby('time')

    cloud_prop_cb = assign_season(cloud_prop_cb, SEASONS)
    cloud_prop_ct = assign_season(cloud_prop_ct, SEASONS)
    cloud_prop_cg = assign_season(cloud_prop_cg, SEASONS)

    cloud_base_single      = cloud_prop_cb.sel(time=mask_single) # Select only single layer clouds
    cloud_top_single       = cloud_prop_ct.sel(time=mask_single) # Select only single layer clouds
    cloud_thickness_single = cloud_prop_cg.sel(time=mask_single) # Select only single layer clouds

    # unique_seasons = np.unique(cloud_base_single.season.values)
    # fig, ax = plt.subplots(figsize=(10, 7))  # 2 rows, 1 column
    # season_positions = {season: i for i, season in enumerate(unique_seasons)}  # Mapping from season to numerical position

    # for i, season in enumerate(unique_seasons):
    #     for j, var_name in enumerate(cloud_base_single.data_vars):
    #         mask_season = cloud_base_single[var_name].season == season
    #         var_season  = cloud_base_single[var_name].where(mask_season, drop=True).compute()
    #         var_season  = var_season.dropna(dim='time', how='all')
    #         result = np.array([x for x in var_season.values if isinstance(x, float)])

    #         # Calculate the position for each var_name with increased spacing
    #         position = season_positions[season] + j * 0.4

    #         # Violin plot
    #         ax.violinplot(result, positions=[position], showmeans=False, showmedians=True, showextrema=False)

    # # Set xticks and labels
    # ax.set_xticks(range(len(unique_seasons)))
    # ax.set_xticklabels(unique_seasons)

    # plt.show()

    # -----------------------------------------------------------------------------------------------
    # Thermodinamic conditions for Non Clouds periods
    # -----------------------------------------------------------------------------------------------
    mask_no_clouds = reindexed_clouds_layers['no_clouds'].compute() == 1
    mwr_no_clouds  = mwr_reindexed['temperature'].where(mask_no_clouds, drop=True).compute()

    if len(mwr_no_clouds.time) > 0:
        fig, ax = plt.subplots(figsize=(10, 7))  # 2 rows, 1 column
        mwr_resampled = mwr_no_clouds.resample(time="M").mean()

        # mesh = ax.pcolormesh(mwr_resampled.time.values, mwr_resampled.altitude.values/1000, mwr_resampled.T - 273.15, cmap='rainbow')
        mesh = ax.contourf(mwr_resampled.time.values, mwr_resampled.altitude.values/1000, mwr_resampled.T - 273.15, cmap='rainbow', levels = 8)

        ax.set_ylabel("Height [km] a.g.l")
        fig.colorbar(mesh, ax=ax, label='Temperature (°C)')
        ax.set_xlim([mwr_resampled.time.values[0], mwr_resampled.time.values[-1]])
        # Customize x-axis tick labels
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.xaxis.set_tick_params(rotation=30)
        ax.set_ylim([0, 10])

        ax.grid(True)
        plt.show()
        fig.savefig(f"{PATH_FIG}mwr_profiles_temperature_no_clouds.png", dpi=300)
    # -----------------------------------------------------------------------------------------------
    # Thermodinamic conditions for each cloud type separately
    # -----------------------------------------------------------------------------------------------
    clouds_to_analyse = ['Liquid', 'Mixed_phase', 'Ice', 'Pre_liquid', 'Pre_mixed_phase']
    # clouds_to_analyse = ['Liquid']
    termo_to_analyse  = ['temperature', 'humidity', 'relative_humidity']
    number_of_clouds_to_analyse = len(clouds_to_analyse)

    for var_name in clouds_single_layer.data_vars:
        if var_name in clouds_to_analyse:
            print(f"Variable: {var_name}")
            mask_var_single_layer      = clouds_single_layer[var_name].compute() == 1
            mwr_reindexed_single_layer = mwr_reindexed['temperature'].where(mask_var_single_layer, drop=True).compute()
            cb_single_layer            = cloud_prop_cb[var_name].where(mask_var_single_layer, drop=True).compute()
            ct_single_layer            = cloud_prop_ct[var_name].where(mask_var_single_layer, drop=True).compute()

            # Removing months with low cloud occurence
            # mask_var_single_layer = mask_var_single_layer.assign_coords(month=mask_var_single_layer['time'].dt.month)
            clouds_single_layer   = clouds_single_layer.assign_coords(month=clouds_single_layer['time'].dt.month)
            count_cloud_type      = mask_var_single_layer.resample(time="M").sum()       # Counting the number of clouds by month
            count_cloud_total     = clouds_single_layer.month.resample(time='M').count() # Counting the total number of clouds by month

            mask_to_remove = (count_cloud_type / count_cloud_total) < .1
            # time_to_keep   = mask_to_remove.where(~mask_to_remove, drop=True).time.values
            time_to_keep   = calculate_time_keep(cloud_layers, reindexed_clouds_layers, freq_rm="M", threshold=0.5)

            if len(mwr_reindexed_single_layer.time) > 0:
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [1]}, figsize=(12, 14))  # 2 rows, 1 column

                mwr_resampled = mwr_reindexed_single_layer.resample(time="M").mean()
                cb_resampled_median  = cb_single_layer.resample(time="M").quantile(.5)
                ct_resampled_median  = ct_single_layer.resample(time="M").quantile(.5)

                time_to_keep = np.intersect1d(mwr_resampled.time.values, np.intersect1d(cb_resampled_median.time.values, time_to_keep))
                mwr_resampled = mwr_resampled.sel(time=time_to_keep)
                cb_resampled_median = cb_resampled_median.sel(time=time_to_keep)
                ct_resampled_median = ct_resampled_median.sel(time=time_to_keep)

                mesh = ax1.contourf(mwr_resampled.time.values, mwr_resampled.altitude.values/1000, mwr_resampled.T - 273.15, cmap='rainbow', levels = 8)
                scat_cb = ax1.plot(cb_resampled_median.time.values, cb_resampled_median/1000, color='black',linestyle='-', marker='o', markersize=5, label=var_name)
                scat_ct = ax1.plot(ct_resampled_median.time.values, ct_resampled_median/1000, color='dimgray',linestyle='-', marker='o', markersize=5, label=var_name)

                ax1.set_ylabel(f"z [km] a.g.l ({var_name.replace('_', '-').title()})")
                ax1.set_xlim([mwr_resampled.time.values[0], mwr_resampled.time.values[-1]])
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))

                ax1.set_ylim([0, 10])
                ax1.grid(True)

                bar_width = (count_cloud_type.time.max().values - count_cloud_type.time.min().values) / count_cloud_type.time.shape[0]
                ax2.bar(count_cloud_type.time.values, count_cloud_type, width=bar_width, color='blue')
                ax2.set_ylabel(r"N$_{profiles}$")
                ax2.set_xlim([mwr_resampled.time.values[0], mwr_resampled.time.values[-1]])
                ax2.xaxis.set_tick_params(rotation=30)
                ax2.set_yticks(np.linspace(0, count_cloud_type.max().values, 5))

                # Add colorbar above the figure horizontally
                cbar = fig.colorbar(mesh, ax=(ax1,ax2), label='Temperature (°C)', extend='both',  orientation='horizontal', pad=0.15)
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.xaxis.set_label_position('top')

                plt.show()
                # plt.tight_layout()
                plt.show()
                fig.savefig(f"{PATH_FIG}mwr_profiles_temperature_{var_name}_cloud_base.png", dpi=300)
    # -----------------------------------------------------------------------------------------------
    # Thermodynamic conditions for each cloud type at the same figure
    # -----------------------------------------------------------------------------------------------
    clouds_to_analyse = ['Liquid', 'Mixed_phase', 'Ice']
    termo_to_analyse  = ['temperature', 'humidity', 'relative_humidity']
    number_of_clouds_to_analyse = len(clouds_to_analyse)
    fig, ax = plt.subplots(number_of_clouds_to_analyse, 1, sharex=True, sharey=True, figsize=(12, 5*number_of_clouds_to_analyse))  # 2 rows, 1 column
    for i, var_name in enumerate(clouds_to_analyse):

        print(f"Variable: {var_name}")
        mask_var_single_layer      = clouds_single_layer[var_name].compute() == 1 # Mask with only single layer clouds for var_name (cloud type)
        mwr_reindexed_single_layer = mwr_reindexed['temperature'].where(mask_var_single_layer, drop=True).compute()
        cb_single_layer            = cloud_prop_cb[var_name].where(mask_var_single_layer, drop=True).compute()
        ct_single_layer            = cloud_prop_ct[var_name].where(mask_var_single_layer, drop=True).compute()

        # Removing months with low cloud occurence
        clouds_single_layer   = clouds_single_layer.assign_coords(month=clouds_single_layer['time'].dt.month) # Assigning month coordinate for all sigle layer clouds
        count_cloud_type      = mask_var_single_layer.resample(time="M").sum()       # Counting the number of clouds by month
        count_cloud_total     = clouds_single_layer.month.resample(time='M').count() # Counting the total number of clouds by month

        mask_to_remove = (count_cloud_type / count_cloud_total) < .1
        # time_to_keep   = mask_to_remove.where(~mask_to_remove, drop=True).time.values
        time_to_keep   = calculate_time_keep(cloud_layers, reindexed_clouds_layers, freq_rm="M", threshold=0.5)

        if len(mwr_reindexed_single_layer.time) > 0:
            mwr_resampled = mwr_reindexed_single_layer.resample(time="M").mean()

            cb_resampled_median  = cb_single_layer.resample(time="M").quantile(.5)
            ct_resampled_median  = ct_single_layer.resample(time="M").quantile(.5)

            time_to_keep  = np.intersect1d(mwr_resampled.time.values, np.intersect1d(cb_resampled_median.time.values, time_to_keep))
            mwr_resampled = mwr_resampled.sel(time=time_to_keep)
            cb_resampled_median = cb_resampled_median.sel(time=time_to_keep)
            ct_resampled_median = ct_resampled_median.sel(time=time_to_keep)

            mesh = ax[i].contourf(mwr_resampled.time.values, mwr_resampled.altitude.values/1000, mwr_resampled.T - 273.15, cmap='rainbow', levels = 8)
            scat_cb = ax[i].plot(cb_resampled_median.time.values, cb_resampled_median/1000, color='black',linestyle='-', marker='o', markersize=5, label=var_name)
            scat_ct = ax[i].plot(ct_resampled_median.time.values, ct_resampled_median/1000, color='dimgray',linestyle='-', marker='o', markersize=5, label=var_name)

            ax[i].set_ylabel(f"z (km) a.g.l ({var_name.replace('_', '-').title()})")
            fig.colorbar(mesh, ax=ax[i], label='Temperature (°C)')
            ax[i].set_xlim([mwr_resampled.time.values[0], mwr_resampled.time.values[-1]])
            # Customize x-axis tick labels
            ax[i].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
            ax[i].xaxis.set_tick_params(rotation=30)
            ax[i].set_ylim([0, 10])
            # reduce space between subplots
            plt.subplots_adjust(hspace=0.1)

            ax[i].grid(True)
    plt.show()
    fig.savefig(f"{PATH_FIG}mwr_profiles_temperature_cloud_types.png", dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(number_of_clouds_to_analyse, 1, sharex=True, sharey=True, figsize=(12, 5*number_of_clouds_to_analyse))  # 2 rows, 1 column
    for i, var_name in enumerate(clouds_to_analyse):

        print(f"Variable: {var_name}")
        mask_var_single_layer      = clouds_single_layer[var_name].compute() == 1
        mwr_reindexed_single_layer = mwr_reindexed['humidity'].where(mask_var_single_layer, drop=True).compute()
        cb_single_layer            = cloud_prop_cb[var_name].where(mask_var_single_layer, drop=True).compute()
        ct_single_layer            = cloud_prop_ct[var_name].where(mask_var_single_layer, drop=True).compute()

        # Removing months with low cloud occurence
        clouds_single_layer   = clouds_single_layer.assign_coords(month=clouds_single_layer['time'].dt.month) # Assigning month coordinate for all sigle layer clouds
        count_cloud_type      = mask_var_single_layer.resample(time="M").sum()       # Counting the number of clouds by month
        count_cloud_total     = clouds_single_layer.month.resample(time='M').count() # Counting the total number of clouds by month

        mask_to_remove = (count_cloud_type / count_cloud_total) < .1
        # time_to_keep   = mask_to_remove.where(~mask_to_remove, drop=True).time.values
        time_to_keep   = calculate_time_keep(cloud_layers, reindexed_clouds_layers, freq_rm="M", threshold=0.5)

        if len(mwr_reindexed_single_layer.time) > 0:
            mwr_resampled = mwr_reindexed_single_layer.resample(time="M").mean()

            cb_resampled_median  = cb_single_layer.resample(time="M").quantile(.5)
            ct_resampled_median  = ct_single_layer.resample(time="M").quantile(.5)

            time_to_keep  = np.intersect1d(mwr_resampled.time.values, np.intersect1d(cb_resampled_median.time.values, time_to_keep))
            mwr_resampled = mwr_resampled.sel(time=time_to_keep)
            cb_resampled_median = cb_resampled_median.sel(time=time_to_keep)
            ct_resampled_median = ct_resampled_median.sel(time=time_to_keep)

            mesh = ax[i].contourf(mwr_resampled.time.values, mwr_resampled.altitude.values/1000, mwr_resampled.T, cmap='rainbow', levels = 8)
            scat_cb = ax[i].plot(cb_resampled_median.time.values, cb_resampled_median/1000, color='black',linestyle='-', marker='o', markersize=5, label=var_name)
            scat_ct = ax[i].plot(ct_resampled_median.time.values, ct_resampled_median/1000, color='dimgray',linestyle='-', marker='o', markersize=5, label=var_name)

            ax[i].set_ylabel(f"z (km) a.g.l ({var_name.replace('_', '-').title()})")
            fig.colorbar(mesh, ax=ax[i], label='Humidity (g m$^{-3}$)')
            ax[i].set_xlim([mwr_resampled.time.values[0], mwr_resampled.time.values[-1]])
            # Customize x-axis tick labels
            ax[i].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
            ax[i].xaxis.set_tick_params(rotation=30)
            ax[i].set_ylim([0, 10])
            # reduce space between subplots
            plt.subplots_adjust(hspace=0.1)

            ax[i].grid(True)
    plt.show()
    fig.savefig(f"{PATH_FIG}mwr_profiles_humidity_cloud_types.png", dpi=300, bbox_inches='tight')
    # -----------------------------------------------------------------------------------------------

    # for var_name in clouds_single_layer.data_vars:
    #     if var_name in clouds_to_analyse:
    #         print(f"Variable: {var_name}")
    #         mask_var_single_layer      = clouds_single_layer[var_name].compute() == 1
    #         mwr_reindexed_single_layer = mwr_reindexed['humidity'].where(mask_var_single_layer, drop=True).compute()
    #         cb_single_layer            = cloud_prop_cb[var_name].where(mask_var_single_layer, drop=True).compute()

    #         # Removing months with low cloud occurence
    #         clouds_single_layer   = clouds_single_layer.assign_coords(month=clouds_single_layer['time'].dt.month) # Assigning month coordinate for all sigle layer clouds
    #         count_cloud_type      = mask_var_single_layer.resample(time="M").sum()       # Counting the number of clouds by month
    #         count_cloud_total     = clouds_single_layer.month.resample(time='M').count() # Counting the total number of clouds by month

    #         mask_to_remove = (count_cloud_type / count_cloud_total) < .1
    #         time_to_keep   = mask_to_remove.where(~mask_to_remove, drop=True).time.values

    #         if len(mwr_reindexed_single_layer.time) > 0:
    #             fig, ax = plt.subplots(figsize=(10, 7))  # 2 rows, 1 column
    #             mwr_resampled = mwr_reindexed_single_layer.resample(time="M").mean()

    #             cb_resampled_median  = cb_single_layer.resample(time="M").quantile(.5)
    #             ct_resampled_median  = ct_single_layer.resample(time="M").quantile(.5)

    #             time_to_keep  = np.intersect1d(mwr_resampled.time.values, np.intersect1d(cb_resampled_median.time.values, time_to_keep))
    #             mwr_resampled = mwr_resampled.sel(time=time_to_keep)
    #             cb_resampled_median = cb_resampled_median.sel(time=time_to_keep)
    #             ct_resampled_median = ct_resampled_median.sel(time=time_to_keep)

    #             # inf_quantile = cb_single_layer.resample(time="M").quantile(.25)
    #             # sup_quantile = cb_single_layer.resample(time="M").quantile(.75)

    #             # mesh = ax.pcolormesh(mwr_resampled.time.values, mwr_resampled.altitude.values/1000, mwr_resampled.T - 273.15, cmap='rainbow')
    #             mesh = ax.contourf(mwr_resampled.time.values, mwr_resampled.altitude.values/1000, mwr_resampled.T, cmap='rainbow', levels = 8)
    #             # scat = ax.errorbar(cb_resampled_mean.time.values, cb_resampled_mean/1000, yerr=cb_resampled_std/1000, fmt='-', label=var_name, c='k', alpha=0.8)
    #             scat_cb = ax.plot(cb_resampled_median.time.values, cb_resampled_median/1000, color='black',linestyle='-', marker='o', markersize=5, label=var_name)
    #             scat_ct = ax.plot(ct_resampled_median.time.values, ct_resampled_median/1000, color='dimgray',linestyle='-', marker='o', markersize=5, label=var_name)

    #             ax.set_ylabel("Height [km] a.g.l")
    #             ax.set_title(f"Cloud Type - {var_name.replace('_', '-').title()}")
    #             fig.colorbar(mesh, ax=ax, label='Humidity (g m$^{-3}$)')
    #             ax.set_xlim([mwr_resampled.time.values[0], mwr_resampled.time.values[-1]])
    #             # Customize x-axis tick labels
    #             ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    #             ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    #             ax.xaxis.set_tick_params(rotation=30)
    #             ax.set_ylim([0, 10])

    #             ax.grid(True)
    #             plt.show()
    #             fig.savefig(f"{PATH_FIG}mwr_profiles_humidity_{var_name}_cloud_base.png", dpi=300)

    # -----------------------------------------------------------------------------------------------
    # Tests
    # -----------------------------------------------------------------------------------------------

    # for i, cloud_type in enumerate(ds.data_vars):
    #     fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    #     cloud_prop_cb_single_layer = ds[cloud_type].where(clouds_single_layer[cloud_type].compute() == 1, drop=True)
    #     sns.boxplot(x=cloud_prop_cb_single_layer.season.values, y=cloud_prop_cb_single_layer.values, orient='v', ax=ax)
    #     # set_trace()
    #     ax.set_ylabel(f"Cloud Base Height - {cloud_type} [m]")
    #     plt.show()

    # unique_season = np.unique(ds.season.values)
    # for season in unique_season:
    #     fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    #     mask_season = ds.season == season
    #     cloud_prop_cb_single_layer = ds.Pre_liquid.where(clouds_single_layer.Pre_liquid.compute() == 1, drop=True)
    #     data_season = cloud_prop_cb_single_layer.where(mask_season, drop=True)
    #     with sns.axes_style("whitegrid"):
    #         sns.boxplot(x=data_season.values, y=data_season.time.dt.year.values, orient='h', ax=ax)

    #     ax.set_title(f"Cloud Base Height - {season}")
    #     plt.show()

    # Define the number of rows and columns for the subplots based on the number of variables
    # num_rows = len(unique_vars)
    # num_cols = len(unique_season)

    # # Create a figure with subplots
    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15), sharey=True)

    # for i, season in enumerate(unique_season):
    #     mask_season = ds.season == season

    #     for j, var_name in enumerate(unique_vars):
    #         ax = axes[j, i]  # Select the appropriate subplot

    #         cloud_prop_cb_single_layer = ds[var_name].where(clouds_single_layer[var_name].compute() == 1, drop=True)
    #         data_season = cloud_prop_cb_single_layer.where(mask_season, drop=True)

    #         with sns.axes_style("whitegrid"):
    #             sns.histplot(data_season.values, kde=True, ax=ax)

    #         ax.set_title(f"{var_name} - {season}")
    #         ax.set_xlabel("Values")

    #     # Add a common y-axis label on the leftmost subplot in each column
    #     axes[0, i].set_ylabel("Frequency")

    # # Adjust subplot layout and spacing
    # plt.tight_layout()
    # plt.show()

    # -----------------------------------------------------------------------------------------------
    # END Tests
    # -----------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------
    # Plotting Histograms and Montlhy time evolution of cloud grometrics properties
    # -----------------------------------------------------------------------------------------------

    # cloud_base_single         = cloud_prop_cb.sel(time=mask_single) # Select only single layer clouds
    # cloud_top_single          = cloud_prop_ct.sel(time=mask_single) # Select only single layer clouds
    # cloud_thickness_single    = cloud_thickness_indexed_byseason.sel(time=mask_single) # Select only single layer clouds

    plot_seasonal_histograms(cloud_base_single,
                            "Cloud base height [m]" ,
                            save_path=f"{PATH_FIG}cloud_base_height_histograms.png")
    plot_seasonal_histograms(cloud_top_single,
                            "Cloud top height [m]" ,
                            save_path=f"{PATH_FIG}cloud_top_height_histograms.png")
    plot_seasonal_histograms(cloud_thickness_single,
                            "Cloud thickness [m]",
                            save_path=f"{PATH_FIG}cloud_thickness_histograms.png")
    # # -----------------------------------------------------------------------------------------------

    plot_cloud_prop_along_time(cloud_base_single,
                               clouds_single_layer,
                               integrated_var_single_layer, r"CB Height a.g.l", save_path=f"{PATH_FIG}cloud_base_height")
    plot_cloud_prop_along_time(cloud_top_single,
                               clouds_single_layer,
                               integrated_var_single_layer, r"CT Height a.g.l", save_path=f"{PATH_FIG}cloud_top_height")
    plot_cloud_prop_along_time(cloud_thickness_single,
                               clouds_single_layer,
                               integrated_var_single_layer, r"$\Delta$Z", save_path=f"{PATH_FIG}cloud_thickness")

# -----------------------------------------------------------------------------------------------

# # if __name__ == "__main__":
# #     main()