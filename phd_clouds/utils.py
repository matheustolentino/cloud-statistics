#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
various utilities
"""
import os
import datetime
import pandas as pd
import requests
from pdb import set_trace
from scipy import ndimage
import numpy as np

def groupSequence(lst, cloud_bins=0):
    # Initialize a result list with the first element of the input list.
    res = [[lst[0]]]

    # Loop through the input list starting from the second element.
    for i in range(1, len(lst)):
        # Check if the current element is consecutive to the previous element.
        if lst[i - 1] + 1 == lst[i]:
            # If consecutive, add the current element to the last sub-list in 'res'.
            res[-1].append(lst[i])
        else:
            # If not consecutive, create a new sub-list in 'res' with the current element.
            res.append([lst[i]])

    # Filter out sub-lists with length less than 'cloud_bins'.
    filtered_res = [sublist for sublist in res if len(sublist) >= cloud_bins]

    # Return the final list of grouped sequences with minimum length 'cloud_bins'.
    return filtered_res

def common_prefix_of_filenames(paths, extension):
    # Get the filenames from each path with the given extension
    filenames_per_path = []
    for path in paths:
        filenames = [file[:8] for file in os.listdir(path) if file.endswith(extension)]
        filenames_per_path.append(set(filenames))

    # Find the common prefixes among the filenames in both paths
    common_prefixes = set.intersection(*filenames_per_path)

    # Convert common prefixes to datetime objects
    datetime_prefixes = []
    for prefix in common_prefixes:
        try:
            datetime_obj = datetime.datetime.strptime(prefix, '%Y%m%d')
            datetime_prefixes.append(datetime_obj)
        except ValueError:
            # If a prefix is not in the expected format, it will be skipped
            pass

    datetime_prefixes.sort()

    return pd.DatetimeIndex(datetime_prefixes)

def download_cloudnet_products(date_ini, date_end, path_output, product='model', site='granada'):
    
    # Define the folder path where you want to save the downloaded files
    # Download the files from the cloudnetpy API
    url = 'https://cloudnet.fmi.fi/api/files'
    payload = {
        'product': product,
        'site': site,
        'dateFrom': date_ini,
        'dateTo': date_end
    }
    metadata = requests.get(url, params=payload).json()

    # Ensure the output folder exists; create it if not.
    os.makedirs(path_output, exist_ok=True)
    
    for row in metadata:
        res = requests.get(row['downloadUrl'])
        
        # Create the full file path by joining the output folder and the filename
        file_path = os.path.join(path_output, row['filename'])
        
        with open(file_path, 'wb') as f:
            f.write(res.content)

def calculate_cloud_composition(mask, cloud_cassification, type_hydrometeors):
    # Label connected components
    labels, num_features = ndimage.label(mask)

    # Initialize a dictionary to store indices for each label
    indices_dict = {}

    # Iterate over each unique label
    for label_val in range(1, num_features+1):
        # Get indices where mask equals the current label
        indices = np.argwhere(labels == label_val)
        # Add indices to the dictionary
        indices_dict[label_val] = indices

    cloud_composition = {}

    # Iterate over each label and calculate cloud composition
    for cloud_number, indx in indices_dict.items():
        n_pixels = len(indx)

        if n_pixels > 1:
            hydro_cloud = cloud_cassification.values[indx[:, 0], indx[:, 1]]
            hydromet_freq = {}
            for hydromet_name, hydromet_val in type_hydrometeors.items():
                freq = np.count_nonzero(hydro_cloud == hydromet_val)
                hydromet_freq[hydromet_name] = freq/n_pixels
            cloud_composition[cloud_number] = hydromet_freq

    return cloud_composition