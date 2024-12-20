import shutil
import psutil
import os
import numpy as np
import netCDF4 as nc
import datetime
import glob
from pdb import set_trace
import time as time_module

# Define the path to the radar data
PATH_RADAR  = '/media/matheustolen/Seagate Basic/cloudnet/radar/'

def get_date_from_files_wo_ldr(path):
    """
    Reads all netCDF files in a given directory using xarray.
    
    Args:
    path (str): Path to directory containing netCDF files.
    
    Returns:
    list: List of xarray.Dataset objects, one for each netCDF file in the directory.
    
    The following code was copied from Cloudnet quality check github repository:
    https://github.com/actris-cloudnet/cloudnetpy-qc/blob/v1.13.6/cloudnetpy_qc/quality.py
    
    class TestLDR(Test):
    def run(self):
        has_ldr = "ldr" in self.nc.variables or "sldr" in self.nc.variables
        has_v = "v" in self.nc.variables
        if has_v and has_ldr:
            v = self.nc["v"][:]
            ldr = (
                self.nc["ldr"][:] if "ldr" in self.nc.variables else self.nc["sldr"][:]
            )
            v_count = ma.count(v)
            ldr_count = ma.count(ldr)
            if v_count > 0 and (ldr_count / v_count * 100) < 0.1:
                self._add_warning("LDR exists in less than 0.1 % of pixels.")

    """
    files = [f for f in os.listdir(path) if f.endswith('.nc')]
    date = []
    for file in files:
        
        dataset = nc.Dataset(os.path.join(path, file))
        ldr_count = np.ma.count(dataset['ldr']) 
        v_count   = np.ma.count(dataset['v'])
        if v_count > 0 and (ldr_count / v_count * 100) < 0.1:
            print(f"File: {file[:8]} - LDR exists in less than 0.1 % of pixels.")
            date.append(datetime.datetime.strptime(file[:8], '%Y%m%d'))
    return date

def get_total_folder_size(path: str):
    """
    Calculates the total size of folders in a given path.
    
    Args:
        path (str): The path to the folder.
        
    Returns:
        None
    """
    # Get the dates without LDR from the files in the given path
    dates_without_ldr = sorted(get_date_from_files_wo_ldr(path))
    
    # Define the path to the radar NAS folder
    path_radar_nas = "/home/matheustolen/shared/NAS_raw_data/UGR/nephele"
    
    # Create a list of folder paths in the NAS folder corresponding to the dates without LDR
    folders_path_nephele_nas = [os.path.join(path_radar_nas, date.strftime('%Y/%m/%d')) for date in dates_without_ldr]

    # Initialize the total size variable
    total_size = 0
    
    # Iterate over each folder path
    for folder_path in folders_path_nephele_nas:
        try:
            # Calculate the size of each file in the folder and sum them up
            folder_size = sum(os.path.getsize(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)))
            
            # Convert the size to gigabytes
            folder_size_gb = folder_size / (1024**3)
            
            # Add the folder size to the total size
            total_size += folder_size_gb
            
            # Print the folder size
            print(f"Folder {folder_path} size: {folder_size_gb:.3f} GB")
        except FileNotFoundError as e:
            print(f"Folder {folder_path} does not exist")
    
    # Print the total size
    print(f"Total size: {total_size:.3f} GB")

def get_data_without_ldr(path: str, path_download_1: str, path_download_2: str):
    """
    Downloads data from NAS without ldr to a specified folder

    """
    path_to_download = path_download_1

    # Get the dates without LDR from the files in the given path
    dates_without_ldr = sorted(get_date_from_files_wo_ldr(path))
    
    # Define the path to the radar NAS folder
    path_radar_nas = "/home/matheustolen/shared/NAS_raw_data/UGR/nephele"
    
    # Create a list of folder paths in the NAS folder corresponding to the dates without LDR
    folders_path_nephele_nas = [os.path.join(path_radar_nas, date.strftime('%Y/%m/%d')) for date in dates_without_ldr]
    
    # Iterate over each folder path
    print("Starting download...")
    for folder_path in folders_path_nephele_nas:
        try:
            # Get the available disk space in percentage
            disk_space = psutil.disk_usage(path_to_download).percent
            print(f"Used disk space: {disk_space}% in {path_to_download}")

            # Check if the available disk space is less than 2%
            if disk_space > 98 and path_to_download == path_download_1:
                print("Not enough free disk space in external hard drive. Changing to second download path (should be the second hard drive).")
                path_to_download = path_download_2
                set_trace()
            elif disk_space > 98:
                print("Not enough free disk space. Stopping download.")
                return
            
            path_to_copy = os.path.join(path_to_download, '/'.join(folder_path.split('/')[-3:]))
            # check if folder already exists in the download path and then rewrite it
            if os.path.exists(path_to_copy):
                shutil.rmtree(path_to_copy)
                print(f"Folder {folder_path} already exists in {path_to_download}. Rewriting it.")
            # Download the data from the NAS folder to specified path 
            shutil.copytree(folder_path, path_to_copy)
            print(f"Folder {folder_path} downloaded to {os.path.join(path_to_download, os.path.basename(folder_path))}")
        except Exception as e:
            print(f"Error while downloading folder {folder_path}: {e}")

def get_data(filepaths, path_download_1: str, path_download_2: str):
    """
    Downloads data from NAS to a specified folder

    """
    path_to_download = path_download_1
    
    # Iterate over each file path
    print("Starting download...")
    for filepath in filepaths:
        try:
            # Get the available disk space in percentage
            disk_space = psutil.disk_usage(path_to_download).percent
            print(f"Used disk space: {disk_space}% in {path_to_download}")
           
            # Check if the available disk space is less than 2%
            if disk_space > 98 and path_to_download == path_download_1:
                print("Not enough free disk space in external hard drive. Changing to second download path (should be the second hard drive).")
                path_to_download = path_download_2
                set_trace()
            elif disk_space > 98:
                print("Not enough free disk space. Stopping download.")
                return
            
            path_to_copy = os.path.join(path_to_download, '/'.join(filepath.split('/')[-4:-1]))
            # check if file already exists in the download path and then rewrite it
            if not os.path.exists(path_to_copy):
                os.makedirs(path_to_copy)
            # Download the data from the NAS folder to specified path
            print(f"Copying {filepath} to {path_to_copy}")
            shutil.copy(filepath, path_to_copy)
            print(f"Folder {filepath} downloaded to {os.path.join(path_to_download, os.path.basename(filepath))}")
        except Exception as e:
            print(f"Error while downloading folder {filepath}: {e}")

if __name__ == "__main__":
   
    # start_time = time_module.time()
    # path_to_download_1 = "/media/matheustolen/EXTERNAL_USB/raw_data"
    # path_to_download_2 = "/media/matheustolen/EXTERNAL_USB1/raw_data"
    # get_data_without_ldr(PATH_RADAR, path_to_download_1, path_to_download_2)
    # end_time = time_module.time()
    # print(f"Execution time: {(end_time - start_time)/60:.2f} minutes")

    start_time = time_module.time()
    days_to_get_from_nas = [3, 9, 10, 24, 25, 26, 30]
    path_nas             = '/home/matheustolen/shared/NAS_raw_data/UGR/nebula_ka/2024/03/'
    # # get all files in the path_nas ended with *.LV1, including in the subdirectories 
    # filepaths = [f for f in glob.glob(path_nas + "**/*.LV1", recursive=True)]
    # same but for the days in days_to_get_from_nas
    filepaths =  [filepath for day in days_to_get_from_nas for filepath in glob.glob(path_nas + f"{day:02d}/*ZEN.LV1")]
    path_to_download_1   = "/mnt/cloudnet_external/data_to_send"
    get_data(filepaths, path_to_download_1, path_to_download_1)  
    end_time = time_module.time()
    print(f"Execution time: {(end_time - start_time)/60:.2f} minutes")


    # OBS: last executioin time: 49 hours
