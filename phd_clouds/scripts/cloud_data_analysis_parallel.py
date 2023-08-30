
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
# import seaborn as sns

# Set the option to split large chunks
dask.config.set(**{'array.slicing.split_large_chunks': True})
#dask.config.set(num_workers=4)

plt.ion()
plt.close('all')
locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
# Set up Seaborn for better visualization
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
# Customize tick parameters to have black markers only at the axis

def create_data_availability_plot(reindexed_variable: xr.DataArray, freq_str: str):
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

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency [%]")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(data_availability), frameon=False)
    ax.set_xticks(freq.time[::3])
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def plot_2d_and_vertical_frequency(dataset1, dataset2):
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
        variable2 = dataset2[var_name]
        ax_lineplot.plot(variable2.values, variable2.range/1e3, label=var_name, color=np.random.rand(3), marker="o")
        ax_lineplot.set_title(f"Line Plot for {var_name}")
        ax_lineplot.set_xlabel("Frequency [%]")
        ax_lineplot.grid(True)
        
        # Adjust vertical spacing between subplots
        # plt.subplots_adjust(wspace=horizontal_space)
        
        # Adjust layout for the subplots
        plt.tight_layout()  # Adjust the left subplot to occupy most of the figure space
        
        plt.show()

def plot_2d_and_vertical_frequency_2(dataset, freq_str: str):
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
    
    plt.show()

def plot_time_evolution_frequency(dataset):
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))  # Set x-axis tick formatter to show month and year
    ax.set_xticks(dataset["time"][::3])  # Set x-axis ticks at every 3 months
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    # Show the plot
    plt.show()

def plot_cfads2(dataset, bin_edges, nbins=50):
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
        var_values = dataset[var_name]

        # Get the chunk size of the variable
        chunk_size = var_values.chunks
        # Create a 2D histogram for each chunk
        hist = np.zeros((nbins-1,nbins-1))
        # Initialize a starting value
        start_value = 0
        for step in chunk_size[0]:
            chunked_values = var_values[start_value:start_value + step, :].values.ravel()
            nan_mask       = np.isnan(chunked_values)
            flattened_var  = chunked_values[~nan_mask]
            flattened_range_var = np.tile(range_values, step)[~nan_mask]
            chunk_hist, x_edges, y_edges = np.histogram2d(flattened_var, flattened_range_var, bins=bin_edges[var_name])
            # Update the value using the current step
            start_value += step
            hist += chunk_hist

        # Calculate bin centers for the contour plot
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

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
                                  xticks_resolution: float = 1.0) -> None:
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
        axes.bar(bin_edges[:-1], norm_hist, width=bin_widths, alpha=0.5, label=label)
    
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_yscale('log')
    axes.legend()
    axes.grid()
    # Set x-axis ticks to bin_edges with specified resolution
    x_ticks = np.arange(min(bin_edges), max(bin_edges) + xticks_resolution, xticks_resolution)
    axes.set_xticks(x_ticks)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
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

def reindex_datasets(chunked_dataset):
    time_series = pd.to_datetime(chunked_dataset.indexes['time'])  # Convert to pandas DateTimeIndex

    # Calculate the new start time as the first 15 seconds of the day
    new_start_time = time_series.min().replace(second=15, microsecond=0)
    new_end_time = time_series.max().replace(hour=23, minute=59, second=59)

    # Create a new time index starting from the new_start_time and ending at the end of the day
    new_time_index = pd.date_range(start=new_start_time, end=new_end_time, freq="30S")

    # Reindex the concatenated variable
    reindexed_variable = chunked_dataset.reindex(time=new_time_index, fill_value=np.nan)

    return reindexed_variable

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

root_folder    = '../../../processed_data/'
target_parent_folder = "hydrometeor"

file_extension = '.nc'
chirp_hydromet = reading_dataset_chunking(root_folder, target_parent_folder)

for key in chirp_hydromet:
    dataset = chirp_hydromet[key]
    initial_time = dataset.time[0].values
    final_time = dataset.time[-1].values
    print(f"Key: {key}, Initial Time: {initial_time}, Final Time: {final_time}")

keys = list(chirp_hydromet.keys())
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        key1 = keys[i]
        key2 = keys[j]
        dataset1 = chirp_hydromet[key1]
        dataset2 = chirp_hydromet[key2]
        mask = dataset1.time == dataset2.time
        print(f"Comparing {key1} and {key2}, Time Overlaping: {mask.sum().item()}")

# Create the figure and set the size
fig, ax = plt.subplots(figsize=(10, 6))
# Iterate over the keys in chirp_hydromet
for key in chirp_hydromet:
    dataset = chirp_hydromet[key]
    time_values = dataset.time.values
    
    # Plot a constant value (key) against the time values
    ax.plot(time_values, [key] * len(time_values),'*')
# Set the x-axis label
plt.xlabel('Time')
# Set the y-axis label
plt.ylabel('Key')
# Set the title
plt.title('Key vs Time')
# Show the plot
plt.show()

hydro_total_list = [ds["Total"].sum(dim="range", skipna=False) for ds in chirp_hydromet.values()]
chirp_concatenated_hydrometeors = xr.concat(hydro_total_list, dim="time").sortby("time")
reindexed_variable = reindex_datasets(chirp_concatenated_hydrometeors)
# -----------------------------------------------------------------------------------------------
# Specify the start and end dates for the data you're interested in (replace with your desired dates)
# -----------------------------------------------------------------------------------------------
start_date = "2021-04-01"
end_date = "2021-04-30"
#sliced_variable = reindexed_variable.sel(time=slice(start_date, end_date))
sliced_variable = reindexed_variable
freq_str = "M"
with sns.axes_style("ticks"):
    create_data_availability_plot(sliced_variable, freq_str)
# -----------------------------------------------------------------------------------------------
#merged_hydromet_dataset = concatenate_dic_by_time(chirp_hydromet).sel(time=slice(start_date, end_date))
#nan_mask = np.isnan(merged_hydromet_dataset)
#mask_hydromet = (merged_hydromet_dataset == 1)
#month_hydromet = mask_hydromet.where(~nan_mask, np.nan)

hydro_list = [ds.sum(dim='range', skipna=False) for ds in chirp_hydromet.values()]
hydro_frequency = xr.concat(hydro_list, dim='time').sortby('time')
#with sns.axes_style("ticks"):
#    # Call the function to plot CFADs
#    plot_2d_and_vertical_frequency(month_hydromet.resample(time=freq_str).mean(dim='time'),
#                                   month_hydromet.mean(dim='time'))
with sns.axes_style("ticks"):
    # Call the function to plot CFADs
    # plot_2d_and_vertical_frequency_2(chirp_hydromet, freq_str)
    plot_time_evolution_frequency((hydro_frequency > 0).resample(time=freq_str).mean())
# -----------------------------------------------------------------------------------------------
# Specify the folder path where the files are located
root_folder    = '../../../processed_data/'
target_parent_folder = "radar_variables"
file_extension = '.nc'

chirp_radar = reading_dataset_chunking(root_folder, target_parent_folder)
bin_edges = {}
nbins = 20
bin_edges['Zh'] = [np.linspace(-60, 20, nbins), np.linspace(0, 10, nbins)]
bin_edges['v']  = [np.linspace(-6, 6, nbins), np.linspace(0, 10, nbins)]

plot_cfads2(chirp_radar['chirp_1'], bin_edges, nbins)
#for key in chirp_radar:
#    dataset = chirp_radar[key]
#    plot_cfads(dataset)
# -----------------------------------------------------------------------------------------------
# # Specify the folder path where the files are located
# -----------------------------------------------------------------------------------------------
root_folder    = '../../../processed_data/'
target_parent_folder = "number_of_layers"
file_extension = '.nc'
chirp_layers = reading_dataset_chunking(root_folder, target_parent_folder)
target_parent_folder = "lwp"
chirp_lwp = reading_dataset_chunking(root_folder, target_parent_folder)
target_parent_folder = "geometric_cloud_thickness"
file_extension = '.json'
chirp_cloud_thickness = reading_dataset_chunking(root_folder, target_parent_folder, file_extension=file_extension)

layer_list   = [ds for ds in chirp_layers.values()]
cloud_layers = xr.concat(layer_list, dim='time').sortby('time')

df_layers = cloud_layers.to_dataframe()
arr_mask = (df_layers.sum(axis=1) == 1.0).to_numpy() # Mask with single layer clouds
del df_layers

lwp_list = [ds for ds in chirp_lwp.values()]
lwp = xr.concat(lwp_list, dim='time').sortby('time')

clouds_single_layer = cloud_layers.sel(time=arr_mask)
clouds_multi_layer  = cloud_layers.sel(time=~arr_mask)
lwp_single_layer    = lwp.sel(time=arr_mask)

chirp_cloud_thickness_list = [ds for ds in chirp_cloud_thickness.values()]
cloud_thickness = xr.concat(chirp_cloud_thickness_list, dim='time').sortby('time')

plot_histograms_with_profiles(datasets=[lwp_single_layer.value.where(clouds_single_layer.Liquid.compute() == 1, drop=True).dropna(dim='time'),
                                        lwp_single_layer.value.where(clouds_single_layer.Mixed_phase.compute() == 1, drop=True).dropna(dim='time')],
                             bin_width=25,
                             labels=["Liquid", "Mixed-phase"],
                             x_label= r"LWP ($\delta$ LWP = 25 [g $m^{-2}$])",
                             y_label="Frequency [%]",
                             xticks_resolution=500)



plot_histograms_with_profiles(datasets=[cloud_thickness.Liquid.where(clouds_single_layer.Liquid.compute() == 1, drop=True).dropna(dim='time'),
                                        cloud_thickness.Ice.where(clouds_single_layer.Ice.compute() == 1, drop=True).dropna(dim='time'),
                                        cloud_thickness.Mixed_phase.where(clouds_single_layer.Mixed_phase.compute() == 1, drop=True).dropna(dim='time')],
                             bin_width=200,
                             labels=["Liquid", "Ice", "Mixed-phase"],
                             x_label= r"CB ($\delta$ CTHICKNESS = 200 [m])",
                             y_label="Frequency [%]",
                             xticks_resolution=500)

# # if __name__ == "__main__":
# #     main()