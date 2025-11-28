import datetime
from phd_clouds.clouds import CloudProcessing
import matplotlib.pyplot as plt
from pdb import set_trace
import matplotlib as mpl
import pandas as pd
import calendar

# Set the font to Times New Roman using LaTeX
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set the fontsize for all elements in the plot
plt.rcParams['font.size'] = 14

plt.close('all')
mpl.use('qtagg')

list_cloud_colors = ["#FFFFFF", "#007CFF", "blue", "cyan", "grey", "yellow", "orange", "magenta"]
cloud_cmap = plt.cm.colors.ListedColormap(list_cloud_colors)

PATH_FIG_TEST     = '../../tests/figures/'

# ---------------------------------------------------------------------------------------------
# Testing the class
# ---------------------------------------------------------------------------------------------

path_save = "/media/matheustolen/Seagate Basic/cloudnet/products/cloud_classification" # Path to save the cloud classification files
path_microphys = "/home/matheustolen/Documentos/matheus_doctorado/output_retrievals" # Path with files already downloaded
path_categorize = "/media/matheustolen/Seagate Basic/cloudnet/categorize" # Path with files already downloaded
path_radar = "/media/matheustolen/Seagate Basic/cloudnet/radar" # Path with files already downloaded
path_classification = "/media/matheustolen/Seagate Basic/cloudnet/classification" # Path with files already downloaded
path_mwr = "/media/matheustolen/Seagate Basic/cloudnet/mwr" # Path with files already downloaded

# --------------------------------------------------------------------------------------------
# JABA experiment:
# ---------------------------------------------------------------------------------------------
# path_save = "/media/matheustolen/Seagate Basic/data_to_jaba/cloudnet/cloud_classification"
# path_microphys = "/media/matheustolen/Seagate Basic/data_to_jaba/cloudnet/microphysics" 
# path_categorize = "/media/matheustolen/Seagate Basic/data_to_jaba/cloudnet/categorize" 
# path_radar = "/media/matheustolen/Seagate Basic/data_to_jaba/cloudnet/radar" 
# path_classification = "/media/matheustolen/Seagate Basic/data_to_jaba/cloudnet/classification" 
# path_mwr = "/media/matheustolen/Seagate Basic/data_to_jaba/cloudnet/mwr"
# path_disdrometer = "/media/matheustolen/Seagate Basic/data_to_jaba/cloudnet/disdrometer" # Path with files already downloaded

# products_to_download = ["classification", 
#                         "categorize", 
#                         "radar", 
#                         "mwr", 
#                         "microphysics",
#                         "disdrometer"]

# path_to_download = {
#     products_to_download[0]: path_classification,
#     products_to_download[1]: path_categorize,
#     products_to_download[2]: path_radar,
#     products_to_download[3]: path_mwr,
#     products_to_download[4]: path_microphys,
#     products_to_download[5]: path_disdrometer
# }

# products_to_download = ["disdrometer"]

# path_to_download = {
#     products_to_download[0]: path_disdrometer,
# }

# ---------------------------------------------------------------------------------------------------
products_to_save = {
    "cloud_occurrence": False,
    "cloud_props": False,
    "cloud_type": False,
    "fit_params": False,
    "count_verification": False,
}

site = 'granada'
# Code to get the intersection between files:
# date_ini = datetime.datetime(2018, 4, 20)
# date_end = datetime.datetime(2023, 12, 31)
date_ini = datetime.datetime(2023, 5, 6) # 2018-09-14
date_end = datetime.datetime(2023, 5, 6)

cloud_processing = CloudProcessing(path_classification=path_classification,
                                path_microphys=None,
                                path_categorize=path_categorize,
                                path_radar=path_radar,
                                path_mwr=path_mwr,
                                site=site)

# cloud_processing.get_filenames(dic_patterns)
# make datestring list from date ini to date end with 1 day step, in the format YYYYMMDD :
date_list = [datetime.datetime.strftime(date, "%Y%m%d") for date in pd.date_range(date_ini.strftime("%Y-%m-%d"), date_end.strftime("%Y-%m-%d"), freq='1D')]
# --------------------------------------------------------------------------------------------
# JABA experiment:
# ---------------------------------------------------------------------------------------------
# print(f"Classification product not found. Downloading...")
# years = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
# months = ["02", "03", "04"]
# # For each year, download data for the month of February, March and April
# for year in years:
#     for month in months:
#         print(f"Downloading data for {year}-{month}...")
#         date_ini = f"{year}-{month}-01"
#         date_end = f"{year}-{month}-31"

#         cloud_processing.download_products(date_ini=date_ini, date_end=date_end, path_output=path_to_download, products=products_to_download)
# 
# date_list = []
# for year in range(2018, 2026):
#     for month in range(2, 5):
#         days_in_month = calendar.monthrange(year, month)[1]
#         for day in range(1, days_in_month + 1):
#             date_list.append(f"{year}{month:02d}{day:02d}")
# 
# set_trace()
# --------------------------------------------------------------------------------------------

for datastr in date_list:
    # Get filenames
    cloud_processing.get_filenames_from_datastr(datastr)

    if not cloud_processing.filenames['classification']:
        print(f"Classification product not found for {datastr}. You should download it first.\nSkipping...")
        continue

    # Load the downloaded with xarray:
    cloud_processing.load_classification()
    cloud_processing.load_categorize()
    cloud_processing.load_radar()
    cloud_processing.load_mwr()

    # Initialize time and datasets     
    cloud_processing.initialize_time()
    cloud_processing.initialize_datasets()

    # ---------------------------------------------------------------------------------------------
    # LWP evaluation
    # ---------------------------------------------------------------------------------------------
    # Add radar LWP
    cloud_processing.add_radar_lwp()

    # Add categorize or MWR LWP
    cloud_processing.add_mwr_lwp()
    # print(cloud_processing.cloud_props.lwp)

    # Calculate fit parameter from radar lwp and mwr lwp
    cloud_processing.calculate_fit_parameters(small_than=0.2, larger_than=0.8)

    cloud_processing.create_count_dataset_for_verification()

    # Printing fit parameters
    # print(cloud_processing.fit_params)
    # print(cloud_processing.count_verification)

    # Plot fit parameters
    cloud_processing.plot_linear_fit_lwp(make_plot=False)
    
    # Generate cloud mask
    cloud_processing.generate_cloud_mask()

    # Classify clusters
    cloud_processing.classify_clusters()
    
    # Analyze clouds: cloud occurrence, ice filter, layer classification, cloud properties i.e. cloud base, cloud top, cloud thickness
    cloud_processing.analyze_clouds(filter_abl_ice_clouds=True, thick_threshold=700, base_threshold=4000)
    
    # Create cluster classification product
    cloud_processing.create_cluster_classification_product()

    # Print variables
    # print(cloud_processing.cluster_classification)

    # Mask attenuation
    cloud_processing.create_attenuation_mask(cloud_cmap, time_roll="10T", lwp_treshold=0.8, corr_treshold=-.5, lwp2_treshold=1, make_plot=True)

    # Add attenuation to clouds
    
    cloud_processing.add_attenuation_to_clouds()

    # # Print variables
    # print(cloud_processing.cloud_occurrence)
    # print(cloud_processing.cloud_props.lwp)
    # print(cloud_processing.cloud_type)
     
    # Save processed data
    cloud_processing.save_processed_data(path_save, products_to_save)
