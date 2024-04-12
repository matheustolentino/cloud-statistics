import struct
import numpy as np
from pdb import set_trace
import datetime
import xarray as xr
import matplotlib.pyplot as plt
import dask

SEASONS = {
        'summer': (6, 8),   # from 1st June to 31st August
        'fall': (9, 11),    # from 1st September to 30th November
        'winter': (12,2),   # from 1st December to 28th February
        'spring': (3, 5)    # from 1st March to 31st May
    }

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
    
    def loadLWP(self, fileName: str) -> tuple:
        """
        Load LWP data from a file.

        Args:
            fileName (str): Path to the file.

        Returns:
            tuple: A tuple containing the loaded data, header, and lwpRetrieval.
        """
        with open(fileName, 'rb') as fid:
            lwpCode = int.from_bytes(fid.read(4), byteorder='little', signed=True)
            N = int.from_bytes(fid.read(4), byteorder='little', signed=True)
            lwpMin = struct.unpack('f', fid.read(4))[0]
            lwpMax = struct.unpack('f', fid.read(4))[0]
            lwpTimeRef = int.from_bytes(fid.read(4), byteorder='little', signed=True)
            lwpRetrieval = int.from_bytes(fid.read(4), byteorder='little', signed=True)
            data = np.zeros((N, 5))
            for i in range(N):
                data[i, 0] = int.from_bytes(fid.read(4), byteorder='little', signed=True) / 60 / 60 / 24 + datetime.datetime(2001, 1, 1).toordinal()
                data[i, 1] = int.from_bytes(fid.read(1), byteorder='little', signed=True)
                data[i, 2] = struct.unpack('f', fid.read(4))[0]
                angle = struct.unpack('f', fid.read(4))[0]
                angleStr = f"{angle:+017.8f}"
                if angleStr[0] == '-':
                    data[i, 3] = -float(angleStr[6:])
                else:
                    data[i, 3] = float(angleStr[6:])
                data[i, 4] = int(angleStr[2:5])
        header = ['Date/Time', 'RainFlag', 'LWP (g/m^2)', 'Elev. Angle (º)', 'Azi. Angle (º)']
        return data, header, lwpRetrieval

    def convert_day_fractions_to_datetime(self, arr_ordinal_dates: np.ndarray):

        # Convert day fractions to timedelta objects and add to the base date
        datetime_array = [datetime.datetime.fromordinal(int(ordinal_date)) + datetime.timedelta(days=ordinal_date % 1) for ordinal_date in arr_ordinal_dates]

        return datetime_array

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
    
    def read_and_convert_lwp_to_dataset(self, file: str):
        """
        Read a MWR file and convert it to an xarray dataset.

        Args:
            file (str): Path to the file.

        Returns:
            xr.Dataset: Dataset containing MWR data.
        """
        # Load the data from the file
        data, header, lwpRetrieval = self.loadLWP(file)

        # Create a new dataset for the first file
        ds_mwr = xr.Dataset(
            {
                'lwp': (["time"], data[:, 2]),
                'elevation_angle': (["time"], data[:, 3]),
                'azimuth_angle': (["time"], data[:, 4]),
                header[1]: (["time"], data[:, 1]),
            },
            coords={
                "time": self.convert_day_fractions_to_datetime(data[:, 0]),
            },
        )

        return ds_mwr

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

    def concatenate_mwr_files_lwp_2(self, filenames: list):
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
            ds_mwr = dask.delayed(self.read_and_convert_lwp_to_dataset)(file)
            delayed_datasets.append(ds_mwr)

        if delayed_datasets:
            # Chunk dataset along the time dimension and concatenate them into a single dataset
            ds_mwr = xr.concat(dask.compute(*delayed_datasets), dim='time')

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