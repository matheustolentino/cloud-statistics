import cloudnetpy as cnp
from cloudnetpy.instruments import rpg2nc
import os
import shutil

site_meta   = {'name': 'Granada', 'altitude': 680}
PATH_CLOUD_RADAR = "/home/matheustolen/shared/NAS_raw_data/UGR/nephele/2023/11/30/"

PATH_DATA_HOST = "/home/matheustolen/Documentos/matheus_doctorado/cloudnet_quicklooks/downloaded_data/"
if not os.path.exists(PATH_DATA_HOST):
    shutil.copytree(PATH_CLOUD_RADAR, PATH_DATA_HOST)
else:
    print("Destination directory already exists.")


# # Getting the list of files
# files = os.listdir(PATH_CLOUD_RADAR)
# zenith_filenames = [f for f in files if f.endswith('.nc')]

uuid = rpg2nc(PATH_DATA_HOST, output_file='./', site_meta=site_meta)
print(f"File {uuid} created")
