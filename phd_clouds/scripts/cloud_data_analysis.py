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
# import seaborn as sns

plt.ion()
plt.close('all')
locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
# Set up Seaborn for better visualization
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
# Customize tick parameters to have black markers only at the axis


def read_files_and_convert(root_folder):
    # Dictionary to store xarray datasets with filenames as keys
    data_dict = {}

    # Walk through the root folder and its subdirectories
    for folder_path, _, filenames in os.walk(root_folder):
        for filename in filenames:
            filepath = os.path.join(folder_path, filename)

            # For NetCDF (.nc) files
            if filename.endswith('.nc'):
                try:
                    # Read the file into an xarray dataset
                    dataset = xr.open_dataset(filepath)

                    # Extract the filename without extension
                    filename_without_extension = os.path.splitext(filename)[0]

                    # Store the dataset in the dictionary
                    data_dict[filename_without_extension] = dataset

                except Exception as e:
                    print(f"Error reading {filename}: {e}")

            # For JSON files
            elif filename.endswith('.json'):
                try:
                    # Read the JSON file into a pandas DataFrame
                    df = pd.read_json(filepath, orient='index')

                    # Convert the DataFrame to an xarray dataset
                    dataset = xr.Dataset.from_dataframe(df)

                    # Set the xarray dataset index name using the DataFrame's index name
                    dataset = dataset.rename({'index': 'time'})

                    # Extract the filename without extension
                    filename_without_extension = os.path.splitext(filename)[0]

                    # Store the dataset in the dictionary
                    data_dict[filename_without_extension] = dataset

                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    return data_dict

def create_data_availability_plot(data, freq_str):
    # Calculate data availability metrics
    data_availability = [
        (data.sum(dim='range', skipna=False) > 0).resample(time=freq_str).mean(),
        (data.sum(dim='range', skipna=False) == 0).resample(time=freq_str).mean(),
        (np.sum(np.isnan(data), axis=1) > 0).resample(time=freq_str).mean()
    ]
    
    # Labels for the data availability categories
    data_availability_lab = ["Hydrometeors", "Clear Sky", "No Data"]
    
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

def plot_cfads(dataset, variables):
    """
    Plot 2D histograms with frequency-based contour plots for given variables.
    
    Parameters:
    - dataset (xarray.Dataset): The dataset containing the variables.
    - variables (list): List of variable names to plot.
    """

    # Extract variables from the dataset
    range_values = dataset.range

    # Define a Seaborn color palette for the colormap with inverted colors
    cmap = sns.color_palette("turbo", as_cmap=True)

    # Create subplots for each variable using gridspec
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, len(variables) + 1, width_ratios=[1] * len(variables) + [0.05])

    for i, var_name in enumerate(variables):
        ax = plt.subplot(gs[i])
        var_values = dataset[var_name]

        # Flatten the arrays and remove NaN values
        flattened_var = var_values.values[~np.isnan(var_values.values)]
        flattened_range_var = np.tile(range_values, var_values.shape[0])[~np.isnan(var_values.values.ravel())]

        # Create a 2D histogram
        hist, x_edges, y_edges = np.histogram2d(flattened_var, flattened_range_var, bins=(50, 50), density=True)
        # Calculate bin centers for the contour plot
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2 / 1e3  # Convert to km

        # Define levels for contour plot
        levels = np.linspace(0, hist.max(), 12)

        # Contour plot for the current variable
        contour = ax.contourf(x_centers, y_centers, hist.T, levels=levels, cmap=cmap, extend='both')
        ax.set_xlabel(var_name)
        ax.set_ylabel('Height [km]')
        ax.set_title(f'2D Histogram of {var_name} vs Height')

        # Increase resolution of x-axis
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

        # Add grid
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # Add a colorbar on the right using the last subplot's position
    cbar_ax = plt.subplot(gs[len(variables)])
    cbar = plt.colorbar(contour, cax=cbar_ax, label='Density', format='%.0e', extend='both')  # Scientific notation
    cbar.set_ticks(levels[:-1] + np.diff(levels)/2)  # Set colorbar ticks at bin centers
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')

    plt.tight_layout()
    plt.show()


def create_frequency_cloud_layers_plot(number_of_layers, freq_str):
    # Calculate frequency of cloud layers
    frequency_cloud_layers = (number_of_layers > 0).resample(time=freq_str).mean()
    
    # List of markers for the plot
    markers = ["o", "*", "s", "<", "X"]
    
    # Create a subplot
    fig, ax = plt.subplots()
    
    # Loop over each variable and create a plot
    for i, variable in enumerate(frequency_cloud_layers):
        p = ax.plot(frequency_cloud_layers['time'], 
                    frequency_cloud_layers[variable],
                    label=variable,
                    color=np.random.rand(3),  # Generate random color
                    marker=markers[i])
    
    # Add a legend
    ax.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    
    # Display the plot
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

def plot_kde(flatten_cloudtop, title="Kernel Density Estimation for Cloud Top Height",
                      ymin=1e-6, ymax=1e-2):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine common x_vals for all plots
    all_flattened_arrays = [flattened_array for flattened_array in flatten_cloudtop.values()]
    min_value = min([min(flattened_array) for flattened_array in all_flattened_arrays])
    max_value = max([max(flattened_array) for flattened_array in all_flattened_arrays])
    x_vals = np.linspace(min_value, max_value, 100)

    for var_name, flattened_array in flatten_cloudtop.items():
        kde = gaussian_kde(flattened_array)
        y_vals = kde(x_vals)
        ax.plot(x_vals, y_vals, label=f"{var_name}")

    ax.set_title(title)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("Density")
    ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(min_value, max_value)
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_histograms(flatten_cloudtop, title):
    fig, ax = plt.subplots(figsize=(9, 5))

    for var_name, flattened_array in flatten_cloudtop.items():
        ax.hist(flattened_array, bins=15, label=var_name, histtype='step')

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_yscale('log')
    ax.legend()
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
    
    plt.tight_layout()
    plt.show()

def flatten_array(dataset: xr.Dataset) -> Dict[str, Union[np.ndarray, None]]:
    """
    Flattens arrays within an xarray dataset.

    Args:
        dataset (xr.Dataset): The input xarray dataset.

    Returns:
        dict: A dictionary containing flattened arrays for each variable in the dataset.
              The dictionary maps variable names to flattened arrays or None if no valid arrays are present.
    """
    flattened_arrays = {}

    for var_name in dataset.data_vars:
        variable = dataset[var_name]
        variable_np = variable.values
        
        flattened_var = []
        for arr in variable_np:
            if arr is not None:
                if np.ndim(arr) == 0:  # Handle scalar values
                    flattened_var.append(np.array([arr]))
                else:
                    flattened_var.append(arr)
        
        if flattened_var:
            flattened_var = np.concatenate(flattened_var)
        else:
            flattened_var = None
        
        flattened_arrays[var_name] = flattened_var
        
    return flattened_arrays

# TODO: def merge_xarray_datasets_by_prefix(data_dict):
    
# Specify the folder path where the files are located
root_folder    = '../../../processed_data/'
processed_data = read_files_and_convert(root_folder)
# ----------------------------------------------------------------------------------------------
# Getting only single layer clouds data from chirp_0
# ----------------------------------------------------------------------------------------------
df_layers = processed_data['chirp_0_number_of_layers'].to_dataframe()
time_mask = (df_layers.sum(axis=1) == 1.0).to_numpy() # Mask with single layer clouds
# Get following variables where is only one layer of cloud
single_layer_mask = processed_data['chirp_0_number_of_layers'].sel(time=time_mask)
single_layer_lwp = processed_data['chirp_0_lwp'].sel(time=time_mask)
single_layer_cbase = processed_data["chirp_0_height_cloud_base"].sel(time=time_mask)
single_layer_ctop = processed_data["chirp_0_height_cloud_top"].sel(time=time_mask)
single_layer_cmean = processed_data["chirp_0_height_cloud_mean"].sel(time=time_mask)
single_layer_cthickness = processed_data["chirp_0_geometric_cloud_thickness"].sel(time=time_mask)
# ----------------------------------------------------------------------------------------------
# Test if the data is read correctly
# Specify the time range you want to slice
start_time = '2021-04-21T00:00:00'  # Replace with your desired start time
end_time   = '2021-04-22T00:00:00'  # Replace with your desired end time

# Slice the data
flatten_cloudtop = flatten_array(processed_data["chirp_0_height_cloud_top"].sel(time=slice(start_time, end_time)))
flatten_cloudbase = flatten_array(processed_data["chirp_0_height_cloud_base"].sel(time=slice(start_time, end_time)))
flatten_cloudmean = flatten_array(processed_data["chirp_0_height_cloud_mean"].sel(time=slice(start_time, end_time)))
flatten_cloudthickness = flatten_array(processed_data["chirp_0_geometric_cloud_thickness"].sel(time=slice(start_time, end_time)))

with sns.axes_style("ticks"):
    # Call the function to plot CFADs
    plot_cfads(processed_data['chirp_0_radar'], ['Zh', 'v'])

freq_str = "1D"
with sns.axes_style("ticks"):
    create_data_availability_plot(processed_data['chirp_0_hydrometeor'].Total, freq_str)

# plot_histograms(flatten_cloudtop, title="Histograms for Cloud Top Height")
# plot_histograms(flatten_cloudbase, title="Histograms for Cloud Base Height")
# plot_histograms(flatten_cloudmean, title="Histograms for Cloud Mean Height")
# plot_histograms(flatten_cloudthickness, title="Histograms for Cloud Thickness")

with sns.axes_style("ticks"):
    # Call the function to plot CFADs
    
    plot_2d_and_vertical_frequency((processed_data['chirp_0_hydrometeor'] > 0).resample(time=freq_str).mean(dim='time'),
                                 (processed_data['chirp_0_hydrometeor'].sel(time=slice(start_time, end_time)).resample(time="60S").sum(dim='time') > 0))
    plot_time_evolution_frequency((processed_data['chirp_0_hydrometeor'].sum(dim='range') > 0).resample(time='1H').mean())


create_frequency_cloud_layers_plot(processed_data['chirp_0_number_of_layers'].where(time_mask, np.nan), 
                                   freq_str)

plot_histograms_with_profiles(datasets=[single_layer_lwp.value.where(single_layer_mask.Liquid == 1, drop=True).dropna(dim='time'),
                                        single_layer_lwp.value.where(single_layer_mask.Mixed_phase == 1, drop=True).dropna(dim='time')],
                             bin_width=25,
                             labels=["Liquid", "Mixed-phase"],
                             x_label= r"LWP ($\delta$ LWP = 25 [g $m^{-2}$])",
                             y_label="Frequency [%]",
                             xticks_resolution=25)

plot_histograms_with_profiles(datasets=[single_layer_cthickness.Liquid.where(single_layer_mask.Liquid == 1, drop=True).dropna(dim='time'),
                                        single_layer_cthickness.Ice.where(single_layer_mask.Ice == 1, drop=True).dropna(dim='time'),
                                        single_layer_cthickness.Mixed_phase.where(single_layer_mask.Mixed_phase == 1, drop=True).dropna(dim='time')],
                             bin_width=200,
                             labels=["Liquid", "Ice", "Mixed-phase"],
                             x_label= r"CB ($\delta$ CTHICKNESS = 200 [m])",
                             y_label="Frequency [%]",
                             xticks_resolution=1000)


# plot_kde(flatten_cloudtop, title="Kernel Density Estimation for Cloud Top Height",
#                       ymin=1e-6, ymax=1e-2)
# plot_kde(flatten_cloudbase, title="Kernel Density Estimation for Cloud Base Height",
#                         ymin=1e-10, ymax=1e-2)
# plot_kde(flatten_cloudmean, title="Kernel Density Estimation for Cloud Mean Height",
#                         ymin=1e-10, ymax=1e-2)
# plot_kde(flatten_cloudthickness, title="Kernel Density Estimation for Cloud Thickness",
#                         ymin=1e-6, ymax=1e-2)
