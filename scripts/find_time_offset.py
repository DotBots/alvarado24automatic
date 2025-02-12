# Synchronize the timestamp between the Lighthouse and Mocap Dataset
# This file needs to be moved into the root of the reposotory before executing

# import the necessary packages
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TKAgg')

from functions.data_processing import   import_data, \
                                        LH2_count_to_pixels, \
                                        camera_to_world_homography

####################################################################################
###                               Options                                        ###
####################################################################################
# Define which of the 6 experimetns you want to plot
experiment_number = 2
LH='LHB'
# Define the start and end time for the plotted trajectory. Useful for plotting smaller sections of large experiments
# Also controls what data is used for calculating the error.
start_idx = 1626 # df.loc index
end_idx = 4200 # df.loc index
# Time offset between the LH and the Mocap
# start_time = -500000e-6
# stop_time = 500000e-6
# steps = 100
start_time = 90e-3
stop_time = 115e-3
steps = 100

####################################################################################
###                            Read Dataset                                      ###
####################################################################################

# file with the data to analyze
LH_data_file = f'./dataset/scene_{experiment_number}/lh_data.csv'
mocap_data_file = f'./dataset/scene_{experiment_number}/mocap_data.csv'
calib_file = './dataset/calibration.json'

# save dataframe
result = {'time offest [ms]':[],
             'MAE':[],
             'RMS':[],
             'STD':[]}

for time_offest in np.linspace(start_time, stop_time, steps):

    # Import data
    df, calib_data = import_data(LH_data_file, mocap_data_file, calib_file, time_offest)
    start_time = df.loc[start_idx]['time_s'] - df.iloc[0]['time_s']
    end_time   = df.loc[end_idx]['time_s']   - df.iloc[0]['time_s']

    ####################################################################################
    ###                            Process Data                                      ###
    ####################################################################################

    # Project sweep angles on to the z=1 image plane
    pts_lighthouse_A = LH2_count_to_pixels(df['LHA_count_1'].values, df['LHA_count_2'].values, 0)
    pts_lighthouse_B = LH2_count_to_pixels(df['LHB_count_1'].values, df['LHB_count_2'].values, 1)

    # Add the LH2 projected matrix into the dataframe that holds the info about what point is where in real life.
    df['LHA_proj_x'] = pts_lighthouse_A[:,0]
    df['LHA_proj_y'] = pts_lighthouse_A[:,1]
    df['LHB_proj_x'] = pts_lighthouse_B[:,0]
    df['LHB_proj_y'] = pts_lighthouse_B[:,1]

    # Extract the calibration points needed to calibrate the homography.
    pts_src = np.array([calib_data['corners_lh2_proj'][LH][key][0:2] for key in ['tl', 'tr', 'br', 'bl']])
    pts_dst = np.array([calib_data['corners_mm'][key][0:2] for key in ['tl', 'tr', 'br', 'bl']])

    # Convert the 4k camera pixel data and the LH2 pixel data to the world coordinate frame of reference.
    pts_cm_lh2 = camera_to_world_homography(df, calib_data)

    # Calculate the L2 distance error
    error = np.linalg.norm(
        df[[LH+'_hom_x',LH+'_hom_y']].values - 
        df[['real_x_mm','real_y_mm']].values, 
        axis=1)
    
    # take only the relevant points for testing
    error = error[start_idx:end_idx]
    # Calculate statistical moments
    error_mean = error.mean()
    error_rms  = np.sqrt((error**2).mean())
    error_std  = error.std()

    # Print results
    print(f"{time_offest*1000:0.3f}ms\t{error_mean:0.3f}\t{error_rms:0.3f}\t{error_std:0.3f}")

    result['time offest [ms]'].append(time_offest)
    result['MAE'].append(error_mean)
    result['RMS'].append(error_rms)
    result['STD'].append(error_std)

# Export results
df_results = pd.DataFrame(result)
