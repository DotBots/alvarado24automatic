import pandas as pd
from datetime import datetime, timedelta
import json
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#############################################################################
###                                Options                                ###
#############################################################################

scene_number = 2

#############################################################################
###                                Code                                   ###
#############################################################################

filename = f"scene_{scene_number}/raw_data/pydotbot.log"

## Read the struct log with the information
# Define a regular expression pattern to extract timestamp and source from log lines
log_pattern = re.compile(r"timestamp=(?P<timestamp>.*?) .*? poly=(?P<poly>\d+) lfsr_index=(?P<lfsr_index>\d+) db_time=(?P<db_time>\d+) sweep=(?P<sweep>\d+)")

# Create an empty list to store the extracted data
data = []

# Open the log file and iterate over each line
with open(filename, "r") as log_file:
    for line in log_file:
        # Extract timestamp and source from the line
        match = log_pattern.search(line)
        if match and "LH2_PROCESSED_DATA" in line:
            # Append the extracted data to the list
            data.append({
                "timestamp": datetime.strptime(match.group("timestamp"), "%Y-%m-%dT%H:%M:%S.%fZ"),
                "poly": int(match.group("poly")),
                "lfsr_index":  int(match.group("lfsr_index")),
                "db_time": int(match.group("db_time")),
                "sweep": int(match.group("sweep")),
            })

# Create a pandas DataFrame from the extracted data

org_data = []
last_state = {"timestamp":-1, "db_time":-1, "lha_poly_0": -1, "lha_count_0": -1, "lha_time_0": -1,  
                                            "lha_poly_1": -1, "lha_count_1": -1, "lha_time_1": -1, 
                                            "lhb_poly_0": -1, "lhb_count_0": -1, "lhb_time_0": -1, 
                                            "lhb_poly_1": -1, "lhb_count_1": -1, "lhb_time_1": -1,}

base_timestamp = {"timestamp":-1, "db_time":-1}

# organize the data into sweep 1 and sweep 2
for idx, sweep in enumerate(data):
    
    # If there was a big time jump, reset the first line assignment
    if last_state["timestamp"] != -1:
        if (sweep["db_time"] - last_state["db_time"]) > 1e6 or (sweep["db_time"] - last_state["db_time"]) < -1e6: # gap larger than 1 sec or a reset occured
            for k in last_state.keys():
                last_state[k] = -1
                # last_state = {"timestamp":-1, "db_time":-1, "lha_poly_0": -1, "lha_count_0": -1, "lha_poly_1": -1, "lha_count_1": -1, "lhb_poly_0": -1, "lhb_count_0": -1, "lhb_poly_1": -1, "lhb_count_1": -1,}


    # lfsr counts larger than 100k, are most likely an outlier
    if sweep['lfsr_index'] > 100000:
        print(f"found outlier: time:{sweep['timestamp']}, poly:{sweep['poly']} count:{sweep['lfsr_index']}")
        continue
    
    # Compute the base timestamp from which db_time will be interpolated
    if last_state["db_time"] == -1:
        # base timestamp has not been saved
        base_timestamp["timestamp"] = sweep["timestamp"]
        base_timestamp["db_time"] = sweep["db_time"]
        last_state["timestamp"] = sweep["timestamp"]
        last_state["db_time"] = sweep["db_time"]
    else:
        last_state["timestamp"] = base_timestamp["timestamp"] + timedelta(microseconds=sweep["db_time"] - base_timestamp["db_time"])
        last_state["db_time"] = sweep["db_time"]

    # Copy the lighthouse basestation indicator
    if sweep["poly"] in [0,1]: lh = "a"
    else: lh = "b"

    # Save to the correct Sweep and LH
    last_state[f"lh{lh}_poly_{sweep['sweep']}"] = sweep['poly']
    last_state[f"lh{lh}_count_{sweep['sweep']}"] = sweep['lfsr_index']


    # Make sure available data is not too old ( newer than 25ms )
    last_state[f"lh{lh}_time_{sweep['sweep']}"] = sweep["db_time"]
    for lh_i, sweep_i in [("a","0"),("a","1"),("b","0"),("b","1")]:
        # check if there is any data too old
        if (sweep["db_time"] - last_state[f"lh{lh_i}_time_{sweep_i}"]) > 25000  and last_state[f"lh{lh_i}_time_{sweep_i}"] != 1:
            # erase that particular data point from the buffer
            last_state[f"lh{lh_i}_poly_{sweep_i}"] = -1
            last_state[f"lh{lh_i}_count_{sweep_i}"] = -1
            last_state[f"lh{lh_i}_time_{sweep_i}"] = -1


    # you found at least 1 datapoint of each LH/sweep.
    # Stop this first line assigment
    if (last_state[f"lha_poly_0"] != -1) and (last_state[f"lha_poly_1"] != -1) and (last_state[f"lhb_poly_0"] != -1) and (last_state[f"lhb_poly_1"] != -1):   
        # save results
        org_data.append(copy.deepcopy(last_state))


df = pd.DataFrame(org_data)

# sort by db time
df.sort_values(by='timestamp', ascending=True, inplace=True)

df = df[["timestamp", "db_time", "lha_poly_0", "lha_count_0", "lha_poly_1", "lha_count_1", "lhb_poly_0", "lhb_count_0", "lhb_poly_1", "lhb_count_1"]]


# ## sort the lhc_counts so that the lowest value is always on lhX_count_0
# # Swapping the lha_counts if lha_count_1 < lha_count_0
# df[['lha_count_0', 'lha_count_1']] = df.apply(
#     lambda row: [row['lha_count_1'], row['lha_count_0']] 
#     if row['lha_count_1'] < row['lha_count_0'] else [row['lha_count_0'], row['lha_count_1']],
#     axis=1, result_type='expand')

# # Swapping the lhb_counts in the same way
# df[['lhb_count_0', 'lhb_count_1']] = df.apply(
#     lambda row: [row['lhb_count_1'], row['lhb_count_0']] 
#     if row['lhb_count_1'] < row['lhb_count_0'] else [row['lhb_count_0'], row['lhb_count_1']],
#     axis=1, result_type='expand')



#############################################################################
###                           Clear Outliers                         ###
#############################################################################
# This goes grid point by grid point and removes datapoints who are too far away from mean.

def clear_outliers(df, speed_threshold=5e3, jump_threshold=7e3):
    """
    takes a dataframe with the following coulmns 
    "timestamp", 'LHA_count_1', 'LHA_count_2', 'LHB_count_1', 'LHB_count_2'
    and removes any rows in which a change of more than 10k units per second occur.
    """


    # Function to calculate the rate of change
    def rate_of_change(row, prev_row):
        time_diff = (row['timestamp'] - prev_row['timestamp']).total_seconds()
        if time_diff > 0:
            for col in ['lha_count_0', 'lha_count_1', 'lhb_count_0', 'lhb_count_1']:
                count_diff = abs(row[col] - prev_row[col])
                rate = count_diff  / time_diff
                if rate > speed_threshold and count_diff > 1e3:
                    return True
        return False
    
    def check_jump(row, prev_row, next_row):
        for col in ['lha_count_0', 'lha_count_1', 'lhb_count_0', 'lhb_count_1']:
            if abs(row[col] - prev_row[col]) > jump_threshold and abs(next_row[col] - row[col]) > jump_threshold:
                return True
        return False

    should_restart = True
    restart_index = 0
    while should_restart:
        should_restart = False
        index_list = df.index.tolist()
        for i in range(restart_index, len(index_list)):

            if i == 0:
                continue
            # for i, row in df.iterrows():

            # Check for any lhb_coun_0 with is not 41342 (this is a glitch value)
            if df.loc[index_list[i]]['lhb_count_0'] in [41342, 54273]:
                df.drop(index_list[i], axis=0, inplace=True)
                should_restart = True
                print(f"[!] - lhb_count_0 == 41342: {index_list[i]}")
                restart_index = max(0, i-2)
                break

            # Check for any lhb_coun_1 with is not 86292 (this is a glitch value)
            if df.loc[index_list[i]]['lhb_count_1'] in [86292, 78591, 60594, 48280]:

                df.drop(index_list[i], axis=0, inplace=True)
                should_restart = True
                print(f"[!] - lhb_count_1 == 86292 or 78591 or 60594: {index_list[i]}")
                restart_index = max(0, i-2)
                break

            # Check for any lha_coun_1 with is not 42439 (this is a glitch value)
            if df.loc[index_list[i]]['lha_count_0'] in [42439, 33746, 54634, 37796]:
                df.drop(index_list[i], axis=0, inplace=True)
                should_restart = True
                print(f"[!] - lha_count_0 == 42439, 33746, 54634: {index_list[i]}")
                restart_index = max(0, i-2)
                break

            # Check for any lha_coun_1 with is not 42439 (this is a glitch value)
            if df.loc[index_list[i]]['lha_count_1'] in [78647]:
                df.drop(index_list[i], axis=0, inplace=True)
                should_restart = True
                print(f"[!] - lha_count_0 == 42439, 33746, 54634: {index_list[i]}")
                restart_index = max(0, i-2)
                break

            # Check for quikly changing outputs
            if rate_of_change(df.loc[index_list[i]], df.loc[index_list[i-1]]):
                df.drop(index_list[i], axis=0, inplace=True)
                should_restart = True
                print(f"[!] - found rate of change fault: {index_list[i]}")
                restart_index = max(0, i-2)
                break

            # Check for individual peaks, 1-row deltas (don't run on the last index)
            if i != len(index_list)-1:
                if check_jump(df.loc[index_list[i]], df.loc[index_list[i-1]], df.loc[index_list[i+1]]):
                    df.drop(index_list[i], axis=0, inplace=True)
                    should_restart = True
                    print(f"[!] - found individual jump fault: {index_list[i]}")
                    restart_index = max(0, i-2)
                    break

            # Check for any row with a 0
            if df.loc[index_list[i]][['lha_count_0', 'lha_count_1', 'lhb_count_0', 'lhb_count_1']].eq(0).any():
                df.drop(index_list[i], axis=0, inplace=True)
                should_restart = True
                print(f"[!] - measurement == 0: {index_list[i]}")
                restart_index = max(0, i-2)
                break


    return df

# df.to_csv("./outlier_scene_2.csv",index=True)
# Get the cleaned values back on the variables needed for the next part of the code.
df = clear_outliers(df, 40e3)

# Change the format of the timestamp column
df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))

df.to_csv(f"scene_{scene_number}/lh_data.csv",index=True)