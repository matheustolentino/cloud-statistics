
#-------------------------------------------------------------------------------------------------------
# import classes
#-------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
import dask.dataframe as dd
import dask.array as da
from dask.array.core import Array
from datetime import timedelta
from datetime import datetime, timedelta
import dask.config
import time as time_module
from dateutil.relativedelta import relativedelta
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, griddata
from itertools import chain
# import seaborn as sns 

# Set the option to split large chunks
dask.config.set(**{'array.slicing.split_large_chunks': True})
#dask.config.set(num_workers=4)

plt.ion()
plt.close('all')
locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
# Set up Seaborn for better visualization
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.5})
# Customize tick parameters to have black markers only at the axis
# ---------------------------------------------------------------------------------------------
# Paths to the data files and save figures
# ---------------------------------------------------------------------------------------------
PATH_FIG = '../figures/'
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

def create_frequency_cloud_layers_plot(frequency_cloud_layers):
    # List of markers for the plot
    markers = ["o", "*", "s", "<", "X"]
    
    # Create a subplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Loop over each variable and create a plot
    for i, variable in enumerate(frequency_cloud_layers):
        p = ax.plot(frequency_cloud_layers['time'], 
                    frequency_cloud_layers[variable],
                    label=variable,
                    color=np.random.rand(3),  # Generate random color
                    marker=markers[i])
    
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set x-axis tick locator to show ticks by month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))  # Set x-axis tick formatter to show month and year
    ax.set_xticks(dataset["time"][::3])  # Set x-axis ticks at every 3 months
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    # Add a legend
    ax.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    
    # Display the plot
    plt.show()

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

def reindex_datasets(chunked_dataset, month=[], chunk_size: int = 1000):
    
    time_series = chunked_dataset.indexes['time'] # Convert to pandas DateTimeIndex
    # Calculate the new start time as the first 15 seconds of the day
    if month:
        new_start_time = time_series.min().replace(month = month[0], day=1, hour=0, minute=0, second=15, microsecond=0)
        new_end_time = time_series.max().replace(month = month[1], day=1, hour=23, minute=59, second=59, microsecond=0) + relativedelta(day=31)
    else:
        new_start_time = time_series.min().replace(day=1, hour=0, minute=0, second=15, microsecond=0)
        new_end_time = time_series.max().replace(day=1, hour=23, minute=59, second=59, microsecond=0) + relativedelta(day=31)
    

    # Create a new time index starting from the new_start_time and ending at the end of the day
    new_time_index = pd.date_range(start=new_start_time, end=new_end_time, freq="30S")

    # Reindex the concatenated variable
    reindexed_variable = chunked_dataset.reindex(time=new_time_index, fill_value=np.nan)

    return reindexed_variable.chunk({"time": chunk_size})

def get_complete_time(chunked_dataset, chunk_size: int = 1000):
    time_series = chunked_dataset.indexes['time'] # Convert to pandas DateTimeIndex

    # Calculate the new start time as the first 15 seconds of the day
    new_start_time = time_series.min().replace(day=1, hour=0, minute=0, second=15, microsecond=0)
    new_end_time = time_series.max().replace(day=1, hour=23, minute=59, second=59, microsecond=0) + relativedelta(day=31)

    # Create a new time index starting from the new_start_time and ending at the end of the day
    new_time_index = pd.date_range(start=new_start_time, end=new_end_time, freq="30S")

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

root_folder    = '../../../processed_data/'
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
# Plotting 2D Histograms
# -----------------------------------------------------------------------------------------------
# freq_str = "M"

# # hydromet_dataset_list   = list(chirp_hydromet.values())

# min_chirp_range = min([var.range.values.min() for var in chirp_hydromet.values()])
# max_chirp_range = max([var.range.values.max() for var in chirp_hydromet.values()])
# max_size_chirp_range = max([var.range.values.size for var in chirp_hydromet.values()])

# delta_range = 60
# bin_edges = np.arange(0, max_chirp_range+delta_range, delta_range) 

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

# hydro_freq_occurence = xr.concat(range_binned_data, dim='time').sortby('time')

# # range_binned_data = [var.groupby_bins('range', bin_edges, labels=bin_edges[1:]).max() for var in chirp_hydromet.values()]
# # time_binned_data  = [var.resample(time=freq_str).mean().dropna(dim='time', how='all') for var in range_binned_data]
# # hydro_freq_occurence = xr.concat(time_binned_data, dim='time').sortby('time')

# # time_binned_data  = [var.resample(time="1D").mean().dropna(dim='time', how='all') for var in chirp_hydromet.values()]
# # range_binned_data = [var.groupby_bins('range', bin_edges, labels=bin_edges[1:]).mean() for var in time_binned_data]
# # hydro_freq_occurence = xr.concat(range_binned_data, dim='time').sortby('time')


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
root_folder    = '../../../processed_data/'
target_parent_folder = "number_of_layers"
file_extension = '.nc'
chirp_layers = reading_dataset_chunking(root_folder, target_parent_folder)
# target_parent_folder = "lwp"
target_parent_folder = "integrated_variables"
# chirp_lwp = reading_dataset_chunking(root_folder, target_parent_folder)
chirp_integrated_var = reading_dataset_chunking(root_folder, target_parent_folder)
target_parent_folder = "geometric_cloud_thickness"
file_extension = '.json'
chirp_cloud_thickness = reading_dataset_chunking(root_folder, target_parent_folder, file_extension=file_extension)
freq_str = "M"

layer_list   = [ds for ds in chirp_layers.values()]
cloud_layers = xr.concat(layer_list, dim='time').sortby('time')

df_layers = cloud_layers.to_dataframe()
mask_single = (df_layers.sum(axis=1) == 1.0).to_numpy() # Mask with single layer clouds
mask_multi  = (df_layers.sum(axis=1) > 1.0) # Mask with multi layer clouds
mask_w_clouds = (df_layers.sum(axis=1) == 0.0) # Mask with no clouds
del df_layers

integrated_var_list = [ds for ds in chirp_integrated_var.values()]
integrated_var = xr.concat(integrated_var_list, dim='time').sortby('time')

clouds_single_layer         = cloud_layers.sel(time=mask_single) # Select only single layer clouds
integrated_var_single_layer = integrated_var.sel(time=mask_single) # Select integrated variables only for single layer clouds

chirp_cloud_thickness_list = [ds for ds in chirp_cloud_thickness.values()]
cloud_thickness = xr.concat(chirp_cloud_thickness_list, dim='time').sortby('time')

reindexed_clouds_layers = reindex_datasets(clouds_single_layer, month=[1, 12])

reindexed_clouds_layers['multilayer'] = xr.DataArray(mask_multi.astype(np.float64), dims='time')
reindexed_clouds_layers['no_clouds'] = xr.DataArray(mask_w_clouds.astype(np.float64), dims='time')
reindexed_clouds_layers['missing'] = reindexed_clouds_layers.multilayer.isnull().astype(np.float64)

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

cloud_base_indexed_byseason = assign_season(cloud_prop_cb, SEASONS)
cloud_top_indexed_byseason  = assign_season(cloud_prop_ct, SEASONS)
cloud_thickness_indexed_byseason = assign_season(cloud_prop_cg, SEASONS)
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
    
# cloud_base_by_season         = cloud_base_indexed_byseason.sel(time=mask_single) # Select only single layer clouds
# cloud_top_by_season          = cloud_top_indexed_byseason.sel(time=mask_single) # Select only single layer clouds
# cloud_thickness_by_season    = cloud_thickness_indexed_byseason.sel(time=mask_single) # Select only single layer clouds

plot_seasonal_histograms(cloud_base_indexed_byseason.sel(time=mask_single),
                         "Cloud base height [m]" , 
                         save_path=f"{PATH_FIG}cloud_base_height_histograms.png")
plot_seasonal_histograms(cloud_top_indexed_byseason.sel(time=mask_single),
                         "Cloud top height [m]" , 
                         save_path=f"{PATH_FIG}cloud_top_height_histograms.png")
plot_seasonal_histograms(cloud_thickness_indexed_byseason.sel(time=mask_single),
                         "Cloud thickness [m]", 
                         save_path=f"{PATH_FIG}cloud_thickness_histograms.png")
# # -----------------------------------------------------------------------------------------------
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

cloud_base_by_season         = cloud_base_indexed_byseason.sel(time=mask_single) # Select only single layer clouds
cloud_top_by_season          = cloud_top_indexed_byseason.sel(time=mask_single) # Select only single layer clouds
cloud_thickness_by_season    = cloud_thickness_indexed_byseason.sel(time=mask_single) # Select only single layer clouds

# plot_cloud_prop_along_time(cloud_base_by_season, 
#                            clouds_single_layer, 
#                            integrated_var_single_layer, r"CB Height a.g.l", save_path=f"{PATH_FIG}cloud_base_height")
# plot_cloud_prop_along_time(cloud_top_by_season, 
#                            clouds_single_layer, 
#                            integrated_var_single_layer, r"CT Height a.g.l", save_path=f"{PATH_FIG}cloud_top_height")
# plot_cloud_prop_along_time(cloud_thickness_by_season, 
#                            clouds_single_layer, 
#                            integrated_var_single_layer, r"$\Delta$Z", save_path=f"{PATH_FIG}cloud_thickness")

# -----------------------------------------------------------------------------------------------

# # if __name__ == "__main__":
# #     main()