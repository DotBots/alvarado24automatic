import pandas as pd
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from skspatial.objects import Plane
import numpy as np

#############################################################################
###                                Options                                ###
#############################################################################

scene_number = 1

########################################################################
###                            Functions                             ###
########################################################################

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2 """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

#########################################################################
###                              Main                                 ###
#########################################################################

# get calibration data

# Sample TSV data (you would replace this with your actual TSV file content)
tsv_data_file = f"scene_{scene_number}/raw_data/scene_{scene_number}_6D.tsv"
# Read the TSV data into a DataFrame
df_data = pd.read_csv(tsv_data_file, sep='\t', skiprows=13)
data_start_time = pd.read_csv(tsv_data_file, sep='\t', skiprows = lambda x: x not in [7]).columns[1]

# convert timestamp to UTC
data_start_time = datetime.strptime(data_start_time, '%Y-%m-%d, %H:%M:%S.%f')
data_start_time = data_start_time.replace(tzinfo=ZoneInfo("Europe/Paris"))
data_start_time = data_start_time.astimezone(ZoneInfo("UTC"))

# Select only the first 5 columns
df_data = df_data.iloc[:, [0, 1, 19, 20, 21]]
#CHange the names of the columns and reorder the columns
df_data.rename(columns={'DotBot 4 X': 'x', 'Y.1':'y', 'Z.1':'z', 'Time':'timestamp','Frame':'frame'}, inplace=True)
df_data = df_data[['timestamp','frame','x','y','z']]
# Drop rows with 0.0 readings
df_data = df_data[(df_data['x'] != 0.0) | (df_data['y'] != 0.0) | (df_data['z'] != 0.0)]
df_data.reset_index(drop=True, inplace=True)

# Convert timetamp column to a datetime object
df_data['timestamp'] = df_data['timestamp'].apply(lambda x: data_start_time + timedelta(seconds=x))

# Convert timetamp column from a datetime object to a properly formated string
df_data['timestamp'] = df_data['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

# Save the selected columns to a CSV file
csv_data_file_path = f"scene_{scene_number}/mocap_data.csv"  # Specify your desired file path here
df_data.to_csv(csv_data_file_path, index=True)

print(f"File saved to {csv_data_file_path}")