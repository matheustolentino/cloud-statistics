import cloudnetpy as cnp
from cloudnetpy.instruments import rpg2nc
import os

site_meta   = {'name': 'Granada', 'altitude': 680}
PATH_CLOUD_RADAR = "/home/matheustolen/shared/NAS_raw_data/UGR/nephele/2024/02/09/"
PATH_TO_NETCDF   = "/home/matheustolen/Documentos/matheus_doctorado/cloudnet_quicklooks/"

# # Getting the list of files
# files = os.listdir(PATH_CLOUD_RADAR)
# zenith_filenames = [f for f in files if f.endswith('.nc')]


uuid = rpg2nc(PATH_CLOUD_RADAR, output_file=PATH_TO_NETCDF, site_meta=site_meta)
print(f"File {uuid} created")
