import os
import requests

# Define the folder path where you want to save the downloaded files
output_folder = '/home/matheustolen/Documentos/matheus_doctorado/output_retrievals'

url = 'https://cloudnet.fmi.fi/api/files'
payload = {
    'product': 'ier',
    'site': 'granada',
    'dateFrom': '2018-04-22',
    'dateTo': '2023-12-31'
}
metadata = requests.get(url, params=payload).json()

# Ensure the output folder exists; create it if not.
os.makedirs(output_folder, exist_ok=True)

for row in metadata:
    res = requests.get(row['downloadUrl'])
    
    # Create the full file path by joining the output folder and the filename
    file_path = os.path.join(output_folder, row['filename'])
    
    with open(file_path, 'wb') as f:
        f.write(res.content)
