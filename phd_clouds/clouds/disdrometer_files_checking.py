# disdrometer files checking 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

path_disdro = '~/Documentos/disdro_files/'

# read the files
dsd_cloudnet = xr.open_dataset(path_disdro + '20240309_granada_parsivel_8f31e165.nc')
dsd_ccres_monitoring = xr.open_dataset(path_disdro + 'granada_rpg94-b19-parsivel_dd-proc_20240309_000000.nc')

# check the if time are equal with a tolerance of 1e-6
np.allclose(dsd_cloudnet.time.values.astype('datetime64[ns]').astype('int64'), dsd_ccres_monitoring.time.values.astype('datetime64[ns]').astype('int64'), atol=1e-6)

fig, ax = plt.subplots()
dsd_cloudnet.radar_reflectivity.plot(ax=ax, label='cloudnet')
dsd_ccres_monitoring.Zdd.plot(ax=ax, label='ccres_monitoring')
ax.grid()
ax.legend()
plt.show()

# make a scatter plot o radar_reflectivity variable from both files
fig, ax = plt.subplots()
ax.scatter(dsd_cloudnet.radar_reflectivity.values, dsd_ccres_monitoring.Zdd.values, s=10)
ax.set_xlabel('radar_reflectivity (cloudnet)')
ax.set_ylabel('Zdd (ccres_monitoring)')
ax.grid()
plt.show()