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

class Mwr:
    def __init__(self):
        pass
    def loadTPC(self, fileName: str):
        # Open the file
        with open(fileName, 'rb') as fid:
            # Read the data from the file
            tpcCode = int.from_bytes(fid.read(4), byteorder='little', signed=True)
            N = int.from_bytes(fid.read(4), byteorder='little', signed=True)
            tpcMin = struct.unpack('f', fid.read(4))[0]
            tpcMax = struct.unpack('f', fid.read(4))[0]
            tpcTimeRef = int.from_bytes(fid.read(4), byteorder='little', signed=True)
            tpcRetrieval = int.from_bytes(fid.read(4), byteorder='little', signed=True)
            nH = int.from_bytes(fid.read(4), byteorder='little', signed=True)
            range_values = np.frombuffer(fid.read(nH*4), dtype=np.int32)
            data = np.zeros((N, nH+2))

            for i in range(N):
                data[i, 0] = int.from_bytes(fid.read(4), byteorder='little', signed=True) / 60 / 60 / 24 + datetime.datetime(2001, 1, 1).toordinal()
                data[i, 1] = int.from_bytes(fid.read(1), byteorder='little', signed=True)
                data[i, 2:] = np.frombuffer(fid.read(nH*4), dtype=np.float32)

        header = ['Date/Time', 'RainFlag', 'T (K)']

        return data, range_values, header, tpcRetrieval

    def loadHPC(self, fileName: str) -> tuple:
        """
        Load HPC data from a file.

        Args:
            fileName (str): Path to the file.

        Returns:
            tuple: A tuple containing the loaded data and information.
        """
        with open(fileName, 'rb') as fid:
            hpcCode = int.from_bytes(fid.read(4), byteorder='little', signed=True)
            N = int.from_bytes(fid.read(4), byteorder='little', signed=True)
            hpcMin = struct.unpack('f', fid.read(4))[0]
            hpcMax = struct.unpack('f', fid.read(4))[0]
            hpcTimeRef = int.from_bytes(fid.read(4), byteorder='little', signed=True)
            hpcRetrieval = int.from_bytes(fid.read(4), byteorder='little', signed=True)
            nH = int.from_bytes(fid.read(4), byteorder='little', signed=True) # Number of altitude levels
            range_values = np.frombuffer(fid.read(nH*4), dtype=np.int32)
            dataH = np.zeros((N, nH+2))


            for i in range(N):
                dataH[i, 0] = int.from_bytes(fid.read(4), byteorder='little', signed=True) / 60 / 60 / 24 + datetime.datetime(2001, 1, 1).toordinal() # Convert day fractions to datetime objects
                dataH[i, 1] = int.from_bytes(fid.read(1), byteorder='little', signed=True)
                dataH[i, 2:] = np.frombuffer(fid.read(nH*4), dtype=np.float32)

            headerH = ['Date/Time', 'RainFlag', 'Humidity (g/m^3)']

            if hpcCode == 117343673:
                rhMin = np.fromfile(fid, dtype=np.float32, count=1)[0]
                rhMax = np.fromfile(fid, dtype=np.float32, count=1)[0]
                dataRH = np.zeros((N, nH+2))

                headerRH = ['Date/Time', 'RainFlag', 'Humidity (%)']
                for i in range(N):
                    dataRH[i, 0] = int.from_bytes(fid.read(4), byteorder='little', signed=True) / 60 / 60 / 24 + datetime.datetime(2001, 1, 1).toordinal()
                    dataRH[i, 1] = int.from_bytes(fid.read(1), byteorder='little', signed=True)

                    try:
                        # NOTE: The read method reads a specified number of bytes from the current file position,
                        # and the file position is then advanced by that amount in the next read operation.
                        dataRH[i, 2:] = np.frombuffer(fid.read(nH * 4), dtype=np.float32, count=nH)

                    except Exception as e:
                        print(f"Filename: {fileName} raised the following exception: {e}")
                        # dataRH[i, 2:] = np.full(nH, np.nan)
                        dataRH   = []  # Create an empty array with the correct shape
                        headerRH = []  # Create an empty list
                        dataH    = []  # Create an empty array with the correct shape
                        headerH  = []  # Create an empty list
                        return dataH, dataRH, range_values, headerH, headerRH, hpcRetrieval
            else:
                dataRH   = []  # Create an empty array with the correct shape
                headerRH = []  # Create an empty list

        return dataH, dataRH, range_values, headerH, headerRH, hpcRetrieval

    def convert_day_fractions_to_datetime(self, arr_ordinal_dates: np.ndarray):

        # Convert day fractions to timedelta objects and add to the base date
        datetime_array = [datetime.datetime.fromordinal(int(ordinal_date)) + datetime.timedelta(days=ordinal_date % 1) for ordinal_date in arr_ordinal_dates]

        return datetime_array

    def concatenate_mwr_files_tpc(self, filenames: list):
        """
        Load MWR data from multiple files and concatenate them into a single xarray dataset.

        Args:
            filenames (list): List of file paths.

        Returns:
            xr.Dataset: Concatenated dataset containing MWR data.
        """
        # Initialize the dataset
        ds_mwr = None

        # Iterate over the files
        for i, file in enumerate(filenames):
            # Load the data from the file
            data, altitude_range, header, tpcRetrieval = self.loadTPC(file)

            # Create a new dataset for the first file
            if ds_mwr == None:
                ds_mwr = xr.Dataset(
                    {
                        'temperature': (["time", "altitude"], data[:, 2:]),
                        header[1]: (["time"], data[:, 1]),
                    },
                    coords={
                        "time": self.convert_day_fractions_to_datetime(data[:, 0]),
                        "altitude": altitude_range,
                    },
                )
            else:
                # Concatenate the data to the existing dataset
                ds_mwr = xr.concat(
                    [
                        ds_mwr,
                        xr.Dataset(
                            {
                                'temperature': (["time", "altitude"], data[:, 2:]),
                                header[1]: (["time"], data[:, 1]),
                            },
                            coords={
                                "time": self.convert_day_fractions_to_datetime(data[:, 0]),
                                "altitude": altitude_range,
                            },
                        )
                    ],
                    dim="time",
                )

        return ds_mwr.sortby('time')

    def read_and_convert_tpc_to_dataset(self, file: str):
        """
        Read a MWR file and convert it to an xarray dataset.

        Args:
            file (str): Path to the file.

        Returns:
            xr.Dataset: Dataset containing MWR data.
        """
        # Load the data from the file
        data, altitude_range, header, tpcRetrieval = self.loadTPC(file)

        # Create a new dataset for the first file
        ds_mwr = xr.Dataset(
            {
                'temperature': (["time", "altitude"], data[:, 2:]),
                header[1]: (["time"], data[:, 1]),
            },
            coords={
                "time": self.convert_day_fractions_to_datetime(data[:, 0]),
                "altitude": altitude_range,
            },
        )

        return ds_mwr

    def read_and_convert_hpc_to_dataset(self, file: str):
        """
        Read a MWR file and convert it to an xarray dataset.

        Args:
            file (str): Path to the file.

        Returns:
            xr.Dataset: Dataset containing MWR data.
        """
        # Load the data from the file
        dataH, dataRH, altitude_range, headerH, headerRH, hpcRetrieval = self.loadHPC(file)

        #Initialize the dataset
        ds_mwr_rh = None
        ds_mwr_h  = None

        if len(dataRH) > 0:
            # Create a new dataset for the first file
            ds_mwr_rh = xr.Dataset(
                {
                    'relative_humidity': (["time", "altitude"], dataRH[:, 2:]),
                    headerRH[1]: (["time"], dataRH[:, 1]),
                },
                coords={
                    "time": self.convert_day_fractions_to_datetime(dataRH[:, 0]),
                    "altitude": altitude_range,
                },
            )
        else:
            print("Problem with relativity humidity or empty values: ", file)
            # print("Time for RH equal time for H ? : "dataRH[0, 0] == dataH[0, 0])

        if len(dataH) > 0:
            # Create a new dataset for the first file
            ds_mwr_h = xr.Dataset(
                {
                    'humidity': (["time", "altitude"], dataH[:, 2:]),
                    headerH[1]: (["time"], dataH[:, 1]),
                },
                coords={
                    "time": self.convert_day_fractions_to_datetime(dataH[:, 0]),
                    "altitude": altitude_range,
                },
            )
        else:
            print("Problem with humidity or empty values: ", file)
            # print("Time for RH equal time for H ? : "dataRH[0, 0] == dataH[0, 0])

        return ds_mwr_rh, ds_mwr_h


    def concatenate_mwr_files_tpc_2(self, filenames: list):
        """
        Load MWR data from multiple files and concatenate them into a single xarray dataset.

        Args:
            filenames (list): List of file paths.

        Returns:
            xr.Dataset: Concatenated dataset containing MWR data.
        """
        # Initialize the dataset
        ds_mwr = None
        delayed_datasets = []

        # Iterate over the files
        for i, file in enumerate(filenames):
            # Load the data from the file
            ds_mwr = dask.delayed(self.read_and_convert_tpc_to_dataset)(file)
            delayed_datasets.append(ds_mwr)

        if delayed_datasets:
            # Chunk dataset along the time dimension and concatenate them into a single dataset
            ds_mwr = xr.concat(dask.compute(*delayed_datasets), dim='time')

        return ds_mwr.sortby('time')

    def concatenate_mwr_files_hpc_2(self, filenames: list):
        """
        Load MWR data from multiple files and concatenate them into a single xarray dataset.

        Args:
            filenames (list): List of file paths.

        Returns:
            xr.Dataset: Concatenated dataset containing MWR data.
        """
        # Initialize the dataset
        ds_mwr_rh = None
        ds_mwr_h = None
        delayed_datasets_rh = []
        delayed_datasets_h  = []
        # Iterate over the files
        for i, file in enumerate(filenames):
            # Load the data from the file
            ds_mwr_rh, ds_mwr_h  = self.read_and_convert_hpc_to_dataset(file)

            # cheking if the dataset is empty
            if ds_mwr_rh is not None:
                delayed_datasets_rh.append(ds_mwr_rh)
                delayed_datasets_h.append(ds_mwr_h)

        # set_trace()
        if delayed_datasets_rh:
            # Chunk dataset along the time dimension and concatenate them into a single dataset
            ds_mwr_rh_concat = xr.concat((delayed_datasets_rh), dim='time')

        if delayed_datasets_h:
            # Chunk dataset along the time dimension and concatenate them into a single dataset
            ds_mwr_h_concat = xr.concat((delayed_datasets_h), dim='time')

        return ds_mwr_rh_concat.sortby('time'), ds_mwr_h_concat.sortby('time')

    def concatenate_mwr_files_hpc(self, filenames: list):
        """
        Load MWR data from multiple files and concatenate them into a single xarray dataset.

        Args:
            filenames (list): List of file paths.

        Returns:
            xr.Dataset: Concatenated dataset containing MWR data.
        """
        # Initialize the dataset
        ds_mwr = None

        # Iterate over the files
        for i, file in enumerate(filenames):
            # Load the data from the file
            dataH, dataRH, altitude_range, headerH, headerRH, hpcRetrieval = self.loadHPC(file)

            # Create a new dataset for the first file
            if ds_mwr == None:
                ds_mwr = xr.Dataset(
                    {
                        'humidity': (["time", "altitude"], dataH[:, 2:]),
                        'relative_humidity': (["time", "altitude"], dataRH[:, 2:]),
                        headerH[1]: (["time"], dataH[:, 1]),
                    },
                    coords={
                        "time": self.convert_day_fractions_to_datetime(dataH[:, 0]),
                        "altitude": altitude_range,
                    },
                )
            else:
                # Concatenate the data to the existing dataset
                ds_mwr = xr.concat(
                    [
                        ds_mwr,
                        xr.Dataset(
                            {
                                'humidity': (["time", "altitude"], dataH[:, 2:]),
                                'relative_humidity': (["time", "altitude"], dataRH[:, 2:]),
                                headerH[1]: (["time"], dataH[:, 1]),
                            },
                            coords={
                                "time": self.convert_day_fractions_to_datetime(dataH[:, 0]),
                                "altitude": altitude_range,
                            },
                        )
                    ],
                    dim="time",
                )

        return ds_mwr.sortby('time')


    def merge_mwr_ds(self, ds_list: list, dim: str):
        """
        Merge a list of xarray datasets along a given dimension.

        Args:
            ds_list (list): List of xarray datasets.
            dim (str): Dimension along which to merge the datasets.

        Returns:
            xr.Dataset: Merged dataset.
        """
        # Initialize the dataset
        ds = None

        # Iterate over the datasets
        for i, ds_ in enumerate(ds_list):
            # Create a new dataset for the first file
            if i == 0:
                ds = ds_
            else:
                # Concatenate the data to the existing dataset
                ds = xr.merge([ds, ds_], join='outer')

        return ds.sortby(dim)
    
    def remove_months(self, resample_freq: str, threshold_freq: float, ds: xr.Dataset):
        """
        Remove months with a number of observations below a given threshold.

        Args:
            resample_freq (str): Frequency to resample the dataset.
            threshold_freq (str): Threshold frequency.
            ds (xr.Dataset): Dataset containing the data.

        Returns:
            xr.Dataset: Dataset with months removed.
        """
        set_trace()
        # Define the initial time grid and resample the dataset
        ds_resampled = ds.resample(time=resample_freq).mean()

        # Get the total frequency of observations in a month according to the resample frequency
        # NOTE: should check this
        total_freq = ds_resampled['time'].dt.days_in_month.values

        
        
        # Remove months with a number of observations below the threshold
        ds_resampled = ds_resampled.where(ds_resampled['nobs'] >= threshold_freq, drop=True)

        # Remove the number of observations variable
        ds_resampled = ds_resampled.drop('nobs')

        return ds_resampled
    

    def standardAtmosphere(self, Hb, dH, Tb, Pb):
        # altura en metros
        # temperatura en Kelvin
        # presion en hectoPascales

        H = np.arange(Hb + dH/2, 20000, dH)
        T = np.zeros_like(H)
        P = np.zeros_like(H)

        for i in range(len(H)):
            if H[i] < 11000:   # standar atmosphere [ground to 11000 m]
                T[i] = Tb - 0.006545 * H[i]
                P[i] = Pb * ((Tb / T[i]) ** -5.2199)
            else:              # standar atmosphere [11000 to 20000 m]
                T[i] = Tb - 71.995   # 71.995 = 0.006545 * 11000
                P[i] = (Pb * ((Tb / T[i]) ** -5.2199)) * np.exp(-0.034164 * (H[i] - 11000) / T[i])

        return H, T, P

# -----------------------------------------------------------------------------------------------
# Input directory containing the MWR data
#------------------------------------------------------------------------------------------

input_directory = '/home/matheustolen/shared/NAS_raw_data/UGR/mwr/'
# -----------------------------------------------------------------------------------------------
quicklook_temperature = True
quicklook_humidity = True
# -----------------------------------------------------------------------------------------------
# initialize the Mwr class
# -----------------------------------------------------------------------------------------------
mwr_data = Mwr()
# -----------------------------------------------------------------------------------------------
# Use glob to find files matching the pattern '*.CMP.TPC' in all subdirectories of input_directory
mwr_files_tpc = glob.glob(input_directory + '**/*.TPC', recursive=True)

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

folder_to_save_mwr_ds = f"../../../processed_data/mwr_profiles/"
if not os.path.exists(folder_to_save_mwr_ds):
    os.makedirs(folder_to_save_mwr_ds)
    # Save mwr_profiles as NetCDF file
    mwr_profiles.to_netcdf(folder_to_save_mwr_ds + 'mwr_profiles.nc')

# Resampling the dataset to monthly frequency
resampled_merged_mwr = mwr_profiles.resample(time='1M').mean()

# Reindexed dataset for montly frequency with nan in months with no data
start_time     = resampled_merged_mwr['time'].min().values
end_time       = resampled_merged_mwr['time'].max().values
new_time_index = pd.date_range(start=start_time, end=end_time, freq='M', normalize=True)

reindexed_mwr = resampled_merged_mwr.reindex(time=new_time_index)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.pcolormesh(reindexed_mwr.time.values, reindexed_mwr.altitude.values/1e3, reindexed_mwr.temperature.values.T-273.15, shading='auto', cmap='rainbow')
fig.colorbar(im, ax=ax, label='Temperature (C)')
ax.set_xlabel('Month/Year')  # Increase font size of x-axis label
ax.set_ylabel('Altitude (km)')  # Increase font size of y-axis label
# ax.set_title('Temperature Variation')  # Increase font size of title
ax.grid(True)  # Add grid lines

# Customize x-axis tick labels
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
ax.xaxis.set_tick_params(rotation=30)
# ax.set_xlim([reindexed_mwr['time'].min().values, datetime.datetime(2024, 2, 1)])
fig.savefig(PATH_FIG + 'mwr_time_series_T.png', dpi=300, bbox_inches='tight')
plt.show()

# Plotting the Temperature variable using seaborn
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.pcolormesh(reindexed_mwr.time.values, reindexed_mwr.altitude.values/1e3, reindexed_mwr.relative_humidity.values.T, shading='auto',
                    cmap='rainbow',
                    vmin=0,
                    vmax=70)
fig.colorbar(im, ax=ax, label='Relative Humidity (%)')
ax.set_xlabel('Month/Year')  # Increase font size of x-axis label
ax.set_ylabel('Altitude (km)')  # Increase font size of y-axis label
# ax.set_title('Temperature Variation')  # Increase font size of title
ax.grid(True)  # Add grid lines

# Customize x-axis tick labels
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
ax.xaxis.set_tick_params(rotation=30)

# ax.set_xlim([reindexed_mwr['time'].min().values, datetime.datetime(2024, 2, 1)])
fig.savefig(PATH_FIG + 'mwr_time_series_RH.png', dpi=300, bbox_inches='tight')
plt.show()


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
