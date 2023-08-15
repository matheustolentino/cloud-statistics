import os
import datetime
import requests

CLOUDNET_URL = "https://cloudnet.fmi.fi/api/raw-files"

def get_all_filenames_in_directory(directory_path):
    filenames = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            filenames.append(os.path.basename(file_path))
    return filenames

def check_if_data_submitted(file, query_param, url):
    query_cloudnet     = requests.get(url, query_param).json()
    uploaded_filenames = [cq['filename'] for cq in query_cloudnet\
                           if (cq['status'] == 'uploaded' or cq['status'] == 'processed')]
    
    if file in uploaded_filenames:
        print(f"Data for filename '{filename}' was already submitted.")
        return True
    print(f"Data for filename '{filename}' was not found in the cloudnet.")
    return False

if __name__ == "__main__":
    # Actual path to your database folder
    path_database = '../../data/NAS/2023/05/02'

    # Call the function to get all filenames inside the database folder
    all_filenames = get_all_filenames_in_directory(path_database)
    missed_data   = []

    # Print the filenames
    for filename in all_filenames:
        api_param     = {"site": "granada",
                         "date": datetime.datetime.strptime(filename[:6], "%y%m%d").strftime("%Y-%m-%d"), 
                         "instrument": "rpg-fmcw-94"}
        
        if not check_if_data_submitted(filename, api_param, CLOUDNET_URL):
            missed_data.append(filename)


