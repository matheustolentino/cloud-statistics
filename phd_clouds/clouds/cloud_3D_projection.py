from rpgpy import rpg2nc
from gfatpy.radar.rpg_nc import rpg
from pathlib import Path
import os
import xarray as xr
from phd_clouds.utils import spherical_to_cartesian, convert_azimuth_y_to_x, cartesian_to_spherical,\
                             download_cloudnet_products, get_instrument_name
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import LinearNDInterpolator
from matplotlib.colors import Normalize
from mayavi import mlab
import time
import pandas as pd

plt.close('all')

mpl.use('qtagg')

RHI_LV1 = Path(r"/home/matheustolen/shared/NAS_raw_data/UGR/nebula_w/2024/06/26")
# RHI_LV1 = Path(r"/home/matheustolen/shared/NAS_raw_data/UGR/nebula_w/2024/04/20")
# RHI_LV1 = Path(r"/home/matheustolen/shared/NAS_raw_data/UGR/nebula_w/2024/06/08")
PATH_NC = Path(r"../../tests/data/radar")
PATH_CAT = Path(r"../../tests/data/classification")
FIGURE_DIR = Path(r"../figures")

# -------------------------------------------------------------------------------------------
# Instrument and site information
# -------------------------------------------------------------------------------------------
site_meta             = {'name': 'Granada', 'altitude': 680}
instrument_nickname   = 'nebula_ka'
site                  = 'granada'

instrument_name = get_instrument_name(instrument_nickname)
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# Downloading and reading the cloudnet classification products
# -------------------------------------------------------------------------------------------
# 1)  get date from RHI_LV1 Path in the format YYYYMMDD
date_to_read = RHI_LV1.parts[-3] + RHI_LV1.parts[-2] + RHI_LV1.parts[-1]
# 2) convert to datetime64
date_ini = pd.to_datetime(date_to_read, format='%Y%m%d')
# add 1 day to the date_ini
date_end = date_ini + pd.DateOffset(days=1)


# 3) Download the cloudnet products
download_cloudnet_products(date_ini.strftime('%Y-%m-%d'), date_end.strftime('%Y-%m-%d'),
                           path_output=PATH_CAT, product='classification', site=site)

# 4) Read the all classification product and concatenate
ds_classification = xr.open_mfdataset(PATH_CAT.glob("*classification.nc"), combine='by_coords')
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# Read the RPG LV1 data and converting to netCDF
# -------------------------------------------------------------------------------------------

# 1) List all RPG LV1 files
filenames_raw = list(RHI_LV1.glob("240626_234*RHI.LV1"))
# filenames_raw = list(RHI_LV1.glob("240420_15272*RHI.LV1"))
# filenames_raw = list(RHI_LV1.glob("240608_042*RHI.LV1"))
filenames_nc  = []
filenames_nc  = []
for filename in filenames_raw:
    output_file = os.path.join(PATH_NC, filename.name + ".nc")
    filenames_nc.append(Path(output_file))
    rpg2nc(filename, output_file= output_file)

# # Fist view of the data
# radar = rpg(filenames_nc[0])
# radar.quicklook(
#         variable=[
#             "dBZe",
#             "v",
#             "width",
#             "specific_differential_phase",
#         ],
#         dpi=400,
#         savefig=True,
#         output_dir=FIGURE_DIR,
#         **{'cmap': 'jet',
#            'shading':'nearest',
#            }
#     )

# 2) Read the RPG LV1 data plot the RHI measueremts
norm = Normalize(vmin=-40, vmax=10)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
rhi_datasets = []
for i, filename in enumerate(filenames_nc):
    radar = rpg(filename)
    xrradar = radar.data
    rhi_datasets.append(xrradar)
    r, theta, phi, x, y, z = spherical_to_cartesian(xrradar['range'], xrradar['azimuth'], xrradar['elevation'])
    values = xrradar['dBZe'].values.T
    # cartesian_coords = np.array([x.ravel(), y.ravel(), z.ravel()]).T
    # transposed_values = xrradar['dBZe'].values.T.ravel()
    # mask = np.isnan(values)
    # x = x[~mask]
    # y = y[~mask]
    # z = z[~mask]
    # values = values[~mask]

    # rho = np.sqrt(x**2 + y**2)
    colors = plt.get_cmap('jet')(norm(values))
    pcm = ax.plot_surface(x, y, z, facecolors=colors,
                          rstride=1,
                          cstride=1,
                          alpha=0.3,
                          cmap='jet',
                          vmin=-40, vmax=10, shade=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_title('RHI Scanning Measurement')
# plt.colorbar(pcm, ax=ax, label='dBZ', norm=norm)
fig.savefig(FIGURE_DIR / "RHI_scanning_measurement.png")
plt.show()


# 3) Concatenating the reflectivity and radial velocity

print("Concatenating radar data...")
for i, filename in enumerate(filenames_nc):

    radar = rpg(filename)
    print("Reading file: %s with azimuth: %f" % (filename.name, radar.data['azimuth'][0]))

    if i == 0:
        concat_data = radar.data
    else:
        concat_data = xr.concat([concat_data, radar.data], dim='time').sortby('time')
# 4) Get the classification data for closest time in concat_data

sliced_classification = ds_classification.sel(time=slice(concat_data['time'].min()-np.timedelta64(30, 'm'),
                                                         concat_data['time'].max()+np.timedelta64(30, 'm')))

# -------------------------------------------------------------------------------------------
# Converting data to cartesian coordinates for all RHI data
# -------------------------------------------------------------------------------------------
rm, thetam, phim, xm, ym, zm = spherical_to_cartesian(concat_data['range'], concat_data['azimuth'], concat_data['elevation'])
# Usee mayavi to plot the concat_data['dBZe'] in 3D
# Plot one plane for each azimuth

mlab.figure(size=(1000, 1000), bgcolor=(0,0,0))
for i, phi in enumerate(np.unique(concat_data['azimuth'])):
    mask = concat_data['azimuth'] == phi
    radar_slice = concat_data.sel(time=mask)
    _,_,_,xi, yi, zi = spherical_to_cartesian(radar_slice['range'], radar_slice['azimuth'], radar_slice['elevation'])

    mlab.mesh(xi/1e3, yi/1e3, zi/1e3, scalars=radar_slice['dBZe'].values.T, colormap='jet')
    # mlab.volume_slice(xi/1e3, yi/1e3, zi/1e3, radar_slice['dBZe'].values.T, plane_orientation='z_axes', colormap='jet')
    mlab.colorbar(title='dBz', orientation='vertical', nb_labels=8)


    # mlab.savefig(FIGURE_DIR / "RHI_%f.png" % phi)
# mlab.view(azimuth=180, elevation=90)
# mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
# save figure

mlab.savefig(filename="../figures/RHI_cloud_measurement.png")
mlab.show()

mlab.figure(size=(1000, 1000), bgcolor=(0,0,0))
_,_,_,xi, yi, zi = spherical_to_cartesian(concat_data['range'], concat_data['azimuth'], concat_data['elevation'])
mlab.mesh(xi/1e3, yi/1e3, zi/1e3, scalars=concat_data['dBZe'].values.T, 
                   )
mlab.colorbar(title='dBz', orientation='vertical', nb_labels=8)
# mlab.savefig(filename="../figures/RHI_cloud_measurement_volume.png")
mlab.show()

cartesian_coords = np.array([xm.ravel(), ym.ravel(), zm.ravel()]).T
transposed_values = concat_data['dBZe'].values.T.ravel()

# Remove NaN values for the Delaunay triangulation
mask             = np.isnan(transposed_values)
cartesian_coords = cartesian_coords[~mask]
values           = transposed_values[~mask]
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# Delaunay triangulation
# -------------------------------------------------------------------------------------------
tri = Delaunay(cartesian_coords)

# fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(7, 7))
# ax.plot_trisurf(cartesian_coords[:, 0], cartesian_coords[:, 1], cartesian_coords[:, 2],
#                 triangles=tri.simplices,
#                 cmap='viridis',
#                 edgecolor='black',
#                 alpha=0.3)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()

mlab.figure(bgcolor=(1, 1, 1))
for simplex in tri.simplices:
    x = cartesian_coords[simplex, 0]
    y = cartesian_coords[simplex, 1]
    z = cartesian_coords[simplex, 2]
    mlab.triangular_mesh(x, y, z, [[0, 1, 2]], color=(1, 0.6, 0.6), opacity=0.5)

mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
mlab.show()
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# Barycentric Interpolation
# -------------------------------------------------------------------------------------------
dgrid = 200
grid_x, grid_y, grid_z = np.meshgrid(np.arange(xm.min(), xm.max(), dgrid),
                                     np.arange(ym.min(), ym.max(), dgrid),
                                     np.arange(zm.min(), zm.max(), dgrid))

grid_cartesian_coords = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

interp      = LinearNDInterpolator(tri, values)
grid_values = interp(grid_cartesian_coords)

# -------------------------------------------------------------------------------------------
# Cloud grid in caartesian and spherical coordinates
# -------------------------------------------------------------------------------------------
mask                 = np.isnan(grid_values)
cartesian_grid_cloud = grid_cartesian_coords[~mask]
values_cloud         = grid_values[~mask]

spherical_grid_cloud = cartesian_to_spherical(cartesian_grid_cloud[:, 0],
                                              cartesian_grid_cloud[:, 1],
                                              cartesian_grid_cloud[:, 2])
# -------------------------------------------------------------------------------------------
# Plot Cloud using matplotlib
# -------------------------------------------------------------------------------------------
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(7, 7))
# scat = ax.scatter(cartesian_grid_cloud[:, 0],
#                   cartesian_grid_cloud[:, 1],
#                   cartesian_grid_cloud[:, 2],
#                   c = values_cloud,
#                   cmap='jet')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# fig.colorbar(scat, ax=ax, label='dBZ')
# fig.savefig(FIGURE_DIR / "RHI_cloud.png")

# plt.show()
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# Plot Cloud using Mayavi
# -------------------------------------------------------------------------------------------
# mlab.clf()
mlab.figure(size=(1000, 1000), bgcolor=(0,0,0))
points = mlab.points3d(cartesian_grid_cloud[:, 0]/1e3,
                       cartesian_grid_cloud[:, 1]/1e3,
                       cartesian_grid_cloud[:, 2]/1e3,
                       values_cloud, colormap="jet")

colorbar = mlab.colorbar(points, title='dBz', orientation='vertical', nb_labels=8)
# mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
# colorbar.scalar_bar.unconstrained_font_size = True  # Allow font size to be adjusted freely
# colorbar.scalar_bar_representation.position = [0.85, 0.15]
# colorbar.scalar_bar_representation.position2 = [0.2, 0.7]
mlab.savefig("../figures/RHI_cloud_reconstruction.png")
mlab.show()
# -------------------------------------------------------------------------------------------

# Compare interpolation with original data
unique_azimuth       = np.unique(concat_data['azimuth'])
unique_azi_spherical = convert_azimuth_y_to_x(unique_azimuth, 'clockwise')

# We should be able to obtain the azimuth angle from the cartesian coordinates
print("Azimuth domain radar (spheric coords):")
print("Min azimuth (spheric coords): %f, max azimuth (spheric coords): %f" % (phi.min(), phi.max()))

azimuth_sup     = np.rad2deg( np.arctan2(cartesian_grid_cloud[:, 1], cartesian_grid_cloud[:, 0]) )
azimuth_sup[azimuth_sup < 0] += 360

print("Azimuth domain interpolation (spheric coords):")
print("Min azimuth (spheric coords): %f, max azimuth (spheric coords): %f" % (azimuth_sup.min(), azimuth_sup.max()))

for i, phi in enumerate(unique_azi_spherical):
    mask = np.isclose(spherical_grid_cloud[2], phi, atol=20)
    cloud_slice = cartesian_grid_cloud[mask]
    rho = np.sqrt(cloud_slice[:, 0]**2 + cloud_slice[:, 1]**2)

    phis = convert_azimuth_y_to_x(spherical_grid_cloud[2][mask], 'clockwise')
    # Make a scatter plot 2D
    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(2, 1, hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(rho, cloud_slice[:, 2], c=values_cloud[mask], cmap='jet')
    ax1.set_xlabel('Distance from radar (m)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Azimuth: %f' % np.mean(phis))
    # fig.savefig(FIGURE_DIR / "RHI_cloud_%f.png" % phi)

    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    # Make a scatter plot for measurement
    mask = concat_data['azimuth'] == unique_azimuth[i]
    radar_slice = concat_data.sel(time=mask)
    _,_,_,xi, yi, zi = spherical_to_cartesian(radar_slice['range'], radar_slice['azimuth'], radar_slice['elevation'])
    rho = np.sqrt(xi**2 + yi**2)
    ax2.pcolormesh(rho, zi, radar_slice['dBZe'].values.T, cmap='jet')
    ax2.set_xlabel('Distance from radar (m)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Azimuth: %f' % unique_azimuth[i])

    plt.show()
