
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
# import seaborn as sns

plt.ion()
plt.close('all')
locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
# Set up Seaborn for better visualization
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
# Customize tick parameters to have black markers only at the axis

def create_data_availability_plot(data, freq_str):
    
    # Convert time coordinate to pandas Series
    time_series = pd.Series(data.time.to_index())
    
    # Create a new time index with the desired frequency
    new_time_index = pd.date_range(start=time_series.min(), end=time_series.max(), freq="30S")
    # set_trace()
    # Reindex the dataset to fill time gaps with NaN values
    new_dataset = data.reindex(time=new_time_index)
    
    # Calculate data availability metrics
    data_availability = [
        (new_dataset.sum(dim='range', skipna=False) > 0).resample(time=freq_str).mean(),
        (new_dataset.sum(dim='range', skipna=False) == 0).resample(time=freq_str).mean(),
        (np.sum(np.isnan(new_dataset), axis=1) > 0).resample(time=freq_str).mean()
    ]
    
    # Labels for the data availability categories
    data_availability_lab = ["Hydrometeors", "Clear Sky", "No Data"]

    # set_trace()
    # Create a subplot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initialize variables for stacking bars
    bot = np.zeros(data_availability[0].shape[0])
    width = 1.  # Bar width
    colors = ["#28fc21", "#07a8e3", "#ffffff"]  # Improved color scheme
    
    # Loop over each data availability category
    for i, freq in enumerate(data_availability):
        # Create stacked bar plot
        p = ax.bar(freq['time'], 100*freq,
                   width,
                   label=data_availability_lab[i],
                   bottom=bot,
                   color=colors[i],
                   edgecolor="black")
        bot += 100*freq
    
    # Format the x-axis date labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency [%]")
    # Add a legend above the figure, out of the graph, and spread
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(data_availability_lab), frameon=False)
    # ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set tick frequency to months
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

    # Remove top and right spines
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    # Display the plot
    plt.tight_layout()  # Improve layout spacing
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

def plot_time_evolution_frequency(dataset):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed

    # Iterate through variable names and plot each variable using the axis.plot method
    for var_name in dataset.data_vars:
        ax.plot(dataset["time"], dataset[var_name], label=var_name)

    # Add labels, title, and legend
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Variable Plots")
    ax.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

    # Show the plot
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
            chunked_values = var_values[start_value:start_value + step, :].compute()
            flattened_var = chunked_values.values[~np.isnan(chunked_values.values)]
            flattened_range_var = np.tile(range_values, step)[~np.isnan(chunked_values.values.ravel())]
            hist, x_edges, y_edges = np.histogram2d(flattened_var, flattened_range_var, bins=(50, 50), density=True)
            histograms.append(hist)
            # Update the value using the current step
            start_value += step
        hist = np.sum(histograms, axis=0)

        # Calculate bin centers for the contour plot
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2 / 1e3  # Convert to km

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
                        # Create a delayed function to open the file as a Dask-backed xarray dataset
                        delayed_dataset = dask.delayed(xr.open_dataset)(filepath)
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

# # Specify the folder path where the files are located
# root_folder    = '../../../processed_data/'
# target_parent_folder = "hydrometeor"
# file_extension = '.nc'
# concatenated_datasets = read_and_concatenate_datasets(root_folder, target_parent_folder, file_extension)['chirp_0']

# # Specify the specific day you're interested in (replace with your desired date)
# specific_day = "2021-04-07"

# # Select the data for the specific day using .sel
# concatenated_datasets = concatenated_datasets.sel(time=specific_day)

# # Set up the figure
# freq_str = "1D"
# with sns.axes_style("ticks"):
#     create_data_availability_plot(concatenated_datasets.Total, freq_str)

# with sns.axes_style("ticks"):
#     # Call the function to plot CFADs
#     plot_2d_and_vertical_frequency((concatenated_datasets > 0).resample(time=freq_str).mean(dim='time'),
#                                  (concatenated_datasets > 0))
#     plot_time_evolution_frequency((concatenated_datasets.sum(dim='range') > 0).resample(time=freq_str).mean())

# Specify the folder path where the files are located
root_folder    = '../../../processed_data/'
target_parent_folder = "radar_variables"
file_extension = '.nc'

concatenated_datasets = reading_dataset_chunking(root_folder, target_parent_folder)
plot_cfads(concatenated_datasets['chirp_0'])

# plot_cfads(concatenated_datasets['chirp_0'], chunks)
# plot_cfads(concatenated_datasets['chirp_0'], ['Zh', 'v'])
# incremental_cfads(root_folder, target_parent_folder)

# Specify the specific day you're interested in (replace with your desired date)
# specific_day = "2021-04-07"

# # Select the data for the specific day using .sel
# concatenated_datasets = concatenated_datasets.sel(time=specific_day)
# with sns.axes_style("ticks"):
#     # Call the function to plot CFADs
#     plot_cfads(concatenated_datasets['chirp_0'], ['Zh', 'v'])