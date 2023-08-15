import requests

url = 'https://cloudnet.fmi.fi/api/files'
payload = {
    'product': 'categorize',
    'site': 'granada',
    'dateFrom': '2021-04-01',
    'dateTo': '2021-04-30'
}
metadata = requests.get(url, payload).json()
for row in metadata:
    res = requests.get(row['downloadUrl'])
    with open(row['filename'], 'wb') as f:
        f.write(res.content)
