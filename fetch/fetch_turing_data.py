import numpy as np
import pandas as pd
import requests
import json

from requests.auth import HTTPBasicAuth

# Choose camera id
camera_id = "00001.03652"

# Choose starting and ending times
start_time = "2020-09-01T04:00:00.464Z"
end_time = "2020-09-03T22:00:00.464Z"

# Convert to start and end dates
start_date = start_time.split('T')[0].replace('-','')
end_date = end_time.split('T')[0].replace('-','')

# Choose vehicle class
vehicle_class = ""

# Generate api_request_url and output data filename
api_request_url = "https://urbanair.turing.ac.uk/odysseus/api/v1/jamcams/hourly?"
data_filename = "../data/raw/"
#camera_id={camera_id}&detection_class={vehicle_class}&starttime={start_time}&endtime={end_time}

if camera_id != "":
    api_request_url += f"camera_id={camera_id}"
    data_filename += f"camera_{camera_id}"
if vehicle_class != "":
    api_request_url += f"&detection_class={vehicle_class}"
    data_filename += f"_class_{vehicle_class}"
if start_time != "":
    api_request_url += f"&starttime={start_time}"
    data_filename += f"_start_{start_date}"
if end_time != "":
    api_request_url += f"&endtime={end_time}"
    data_filename += f"_end_{end_date}"
# Add csv extension at the end
data_filename += ".csv"

# Get response
response = requests.get(api_request_url,
                        auth=HTTPBasicAuth('admin', 'x7WBcuRtrgK8255rPZcB'))

if response.status_code == 200:
    print("Success!")
else:
    print('Failure...')

json_data = response.json()
pd_data = pd.DataFrame.from_dict(json_data)

# print(pd_data.head())

pd_data.to_csv(data_filename)

print(f'Data exported as {data_filename}')
