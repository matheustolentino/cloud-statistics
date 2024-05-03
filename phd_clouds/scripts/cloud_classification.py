from phd_clouds.constants import CLEAR_SKY, CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS, ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE, MELTING_ICE_LIQUID_DROPLETS, AERO_NO_CLOUD, INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD, CLASSIFICATION_TICK_LABELS
from phd_clouds.constants import TARG_BET_HYDRO
from phd_clouds.clouds import CloudProcess
import xarray as xr
from phd_clouds.mwr_class import Mwr
from phd_clouds.utils import download_cloudnet_products, calculate_cloud_composition
import matplotlib.pyplot as plt
from pdb import set_trace
import os 

from scipy import ndimage
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

fontsize = 14
# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize

plt.ion()
plt.close('all')

HYDRO_TYPES = {"CLOUD_LIQUID": CLOUD_LIQUID,
               "DRIZZLE_OR_RAIN": DRIZZLE_OR_RAIN,
               "DRIZZLE_OR_RAIN_LIQUID_DROPLETS": DRIZZLE_OR_RAIN_LIQUID_DROPLETS,
               "ICE_PARTICLES": ICE_PARTICLES,
               "ICE_WITH_SUP_WATER": ICE_WITH_SUP_WATER,
               "MELTING_ICE": MELTING_ICE,
               "MELTING_ICE_LIQUID_DROPLETS": MELTING_ICE_LIQUID_DROPLETS}

path_output = "/home/matheustolen/Documentos/matheus_doctorado/phd-clouds/tests/data"

site= 'granada'
product = 'classification'
date_ini = "2022-11-16"
date_end = "2022-11-16"
date_end_new = date_end.replace("-", "")

download_cloudnet_products(date_ini, date_end, path_output, product=product, site=site)
# Load the downloaded with xarray pandas:

# Load the NetCDF file
data = xr.open_dataset(os.path.join(path_output, f'{date_end_new}_{site}_{product}.nc'))

# Pass data['target_classification'] to a DataFrame
df_aux = pd.DataFrame(data['target_classification'].values, columns=data['target_classification']['height'].values, index=data['target_classification']['time'].values)

hydro = [CLOUD_LIQUID, DRIZZLE_OR_RAIN, DRIZZLE_OR_RAIN_LIQUID_DROPLETS,\
                                        ICE_PARTICLES, ICE_WITH_SUP_WATER, MELTING_ICE,\
                                                MELTING_ICE_LIQUID_DROPLETS]
classification_filter = CloudProcess(df_aux,
                                     hydro,
                                     TARG_BET_HYDRO,
                                     1)

# cloud_mask = xr.open_dataset("./ds_mask_reindexed.nc")
cloud_mask          = classification_filter.cloud_mask().to_numpy(dtype=int)
cloud_cassification = data['target_classification']
# set_trace()
# struct = ndimage.generate_binary_structure(2, 2)
# erode  = ndimage.binary_erosion(mask, struct)
# edges = mask ^ erode

# # the indices of the non-zero locations and their corresponding values
# nonzero_idx = np.vstack(np.where(mask)).T
# nonzero_vals = mask[mask]

# # use it to find the indices of all non-zero values that are at most 1 pixel
# # away from each edge pixel
# edge_idx = np.vstack(np.where(edges)).T

# fig, ax = plt.subplots(figsize=(15, 6))
# sc = ax.scatter(edge_idx[:, 0], edge_idx[:, 1], c='r', s=1)
# # sc = ax.scatter(nonzero_idx[:, 1], nonzero_idx[:, 0], c='b', s=1)
# plt.show()

# # Plot the edges
# fig, ax = plt.subplots(figsize=(15, 6))
# # pc2 = ax.pcolormesh(data['cloud_classification']['time'], data['cloud_classification']['height'], data['cloud_classification'].T, cmap='tab20')
# pc = ax.pcolormesh(cloud_mask['time'], cloud_mask['height'], edges.T, cmap='Blues')
# # scat = ax.scatter(data['cloud_classification']['time'][edge_idx[:, 1]], edge_idx[:, 0], c='r', s=1)
# ax.set_xlabel('Time')
# ax.set_ylabel('Height (m)')
# # Convert time values to HH:MM format
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# ax.grid()
# plt.colorbar(pc, ax=ax)
# plt.show()

# Label connected components
labels, num_features = ndimage.label(cloud_mask)

# Initialize a dictionary to store indices for each label
indices_dict = {}

# Iterate over each unique label
for label_val in range(1, num_features+1):
    # Get indices where mask equals the current label
    indices = np.argwhere(labels == label_val)
    # Add indices to the dictionary
    indices_dict[label_val] = indices

cloud_composition = {}
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[1, .02], height_ratios=[1, 1], wspace=0.05, hspace=0.08)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)

# Plot the cloud composition
v = 1
for cloud_number, indx in indices_dict.items():
    n_pixels = len(indx)

    if n_pixels > 200:
        x = data.time.values[indx[:, 0]]
        y = data.height.values[indx[:, 1]]/1e3
        sc = ax1.scatter(x, y, s=1, color=np.random.rand(3,))
    
        hydro_cloud = cloud_cassification.values[indx[:, 0], indx[:, 1]]
        hydromet_freq = {}
        for hydromet_name, hydromet_val in HYDRO_TYPES.items():
            count = np.count_nonzero(hydro_cloud == hydromet_val)
            hydromet_freq[hydromet_name] = count/n_pixels
            cloud_composition[v]     = hydromet_freq
        v += 1
ax1.set_ylabel('Height (km)')
ax1.grid()
ax1.set_xlim(data.time.values[0], data.time.values[-1])

# Get cloud classification for the mask
cloud_classification_mask = cloud_cassification.values * cloud_mask
# Plot the cloud classification
pc = ax2.pcolormesh(cloud_cassification['time'], cloud_cassification['height']/1e3, cloud_classification_mask.T, cmap='tab10', vmin=1, vmax=7)
ax2.set_xlabel('Time')
ax2.set_ylabel('Height (km)')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.grid()

ax3 = fig.add_subplot(gs[1, 1])
cbar = plt.colorbar(pc, cax=ax3)
cbar.set_ticks(np.arange(1, 8))
plt.show()

# Print number of valid clouds
print(f"Valid clouds: {cloud_composition.keys()}")

# distances, indices = tree.query(edge_idx, k=2, eps=1)
# close_islands = np.unique(indices[:, 0])
# print(f"Close islands: {close_islands}")
# # Find the indices of the closed islands
# closed_islands_idx = np.where(np.isin(nonzero_idx[:, 0], close_islands))[0]
# print(f"Indices of closed islands: {closed_islands_idx}")