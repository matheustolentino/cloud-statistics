import xarray as xr
from pdb import set_trace
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy import ndimage
from scipy.spatial import cKDTree

def dfs(matrix, visited, x, y, n, m):
     
    # If the land is already visited
    # or there is no land or the
    # coordinates gone out of matrix
    # break function as there
    # will be no islands
    if (x < 0 or y < 0 or
        x >= n or y >= m or
        visited[x][y] == True or
        matrix[x][y] == 0):
        return
         
    # Mark land as visited
    visited[x][y] = True
 
    # Traverse to all adjacent elements
    dfs(matrix, visited, x + 1, y, n, m);
    dfs(matrix, visited, x, y + 1, n, m);
    dfs(matrix, visited, x - 1, y, n, m);
    dfs(matrix, visited, x, y - 1, n, m);


def plot(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    plt.tight_layout()


fontsize = 14
# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = fontsize

plt.ion()
plt.close('all')

# Load the NetCDF file
data = xr.open_dataset('./ds_mask_reindexed.nc')

# Access the cloud mask variable
cloud_mask = data['cloud_mask']

# # Plot the cloud mask using pcolormesh
# fig = plt.figure(figsize=(17, 10))
# gs = fig.add_gridspec(2, 2, width_ratios=[3, .05], wspace=0.05, height_ratios=[3, 1])

# ax1 = fig.add_subplot(gs[0, 0])
# pc  = ax1.pcolormesh(cloud_mask['time'], cloud_mask['height'], cloud_mask.T, cmap='Blues')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Height (m)')
# # Convert time values to HH:MM format
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# ax1.grid()
# # Add colorbar and labels

# cax=fig.add_subplot(gs[0, 1])
# cbar = plt.colorbar(pc, cax=cax, orientation='vertical')

# ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
# # Obtaining clusters using DBSCAN
# # dbs = DBSCAN(eps=.6)
# dbs = AgglomerativeClustering(distance_threshold=100, n_clusters=None, linkage='ward')
# X = cloud_mask.values.astype(int)
# dbs.fit(X)
# labels        = dbs.labels_
# unique_labels = set(labels)
# print(f"Unique labels: {unique_labels}")
# colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
# core_saples_mask = np.zeros_like(labels, dtype=bool)

# cluster = ax2.plot(cloud_mask['time'], labels, 'o', markersize=2)
# ax2.set_xlabel('Time')
# ax2.set_ylabel('Labels')
# ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# ax2.grid()
# plt.show()

mask = cloud_mask.values.astype(int)
struct = ndimage.generate_binary_structure(2, 2)
erode  = ndimage.binary_erosion(mask, struct)
edges = mask ^ erode

# the indices of the non-zero locations and their corresponding values
nonzero_idx = np.vstack(np.where(mask)).T
nonzero_vals = mask[mask]

# build a k-D tree
tree = cKDTree(nonzero_idx)

# use it to find the indices of all non-zero values that are at most 1 pixel
# away from each edge pixel
edge_idx = np.vstack(np.where(edges)).T

# Plot the edges
fig, ax = plt.subplots(figsize=(15, 6))
# pc2 = ax.pcolormesh(data['cloud_classification']['time'], data['cloud_classification']['height'], data['cloud_classification'].T, cmap='tab20')
pc = ax.pcolormesh(cloud_mask['time'], cloud_mask['height'], edges.T, cmap='Blues')
# scat = ax.scatter(data['cloud_classification']['time'][edge_idx[:, 1]], edge_idx[:, 0], c='r', s=1)
ax.set_xlabel('Time')
ax.set_ylabel('Height (m)')
# Convert time values to HH:MM format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.grid()
plt.colorbar(pc, ax=ax)
plt.show()
