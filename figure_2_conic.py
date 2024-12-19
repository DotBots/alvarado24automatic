# import the necessary packages
import json
import numpy as np
import pandas as pd
from datetime import datetime

# import matplotlib
# matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from functions.data_processing import   import_data, \
                                        LH2_count_to_pixels, \
                                        camera_to_world_homography, \
                                        get_circles, \
                                        intersect_ellipses, \
                                        compute_correcting_homography


from functions.plotting import plot_trajectory_and_error, plot_error_histogram, plot_projected_LH_views

####################################################################################
###                               Options                                        ###
####################################################################################
# Define which of the 6 experimetns you want to plot
experiment_number = 1

####################################################################################
###                            Read Dataset                                      ###
####################################################################################

# file with the data to analyze
LH_data_file = f'./dataset/scene_{experiment_number}/lh_data.csv'
mocap_data_file = f'./dataset/scene_{experiment_number}/mocap_data.csv'
calib_file = './dataset/calibration.json'

# Import data
df, calib_data, (start_idx, end_idx) = import_data(LH_data_file, mocap_data_file, calib_file)

start_time = df.loc[start_idx]['time_s'] - df.iloc[0]['time_s']
end_time   = df.loc[end_idx]['time_s']   - df.iloc[0]['time_s']

####################################################################################
###                            Process Data                                      ###
####################################################################################
errors=[]
for LH in ['LHA', 'LHB']:
    # Project sweep angles on to the z=1 image plane
    pts_lighthouse_A = LH2_count_to_pixels(df['LHA_count_1'].values, df['LHA_count_2'].values, 0)
    pts_lighthouse_B = LH2_count_to_pixels(df['LHB_count_1'].values, df['LHB_count_2'].values, 1)

    # Plot the LH view for debbuging
    # plot_projected_LH_views(pts_lighthouse_A, pts_lighthouse_B)

    # Add the LH2 projected matrix into the dataframe that holds the info about what point is where in real life.
    df['LHA_proj_x'] = pts_lighthouse_A[:,0]
    df['LHA_proj_y'] = pts_lighthouse_A[:,1]
    df['LHB_proj_x'] = pts_lighthouse_B[:,0]
    df['LHB_proj_y'] = pts_lighthouse_B[:,1]

    ####################################################################################
    ###                             Conic algorithm                                 ###
    ####################################################################################

    # Get the conic equations for all of the circles.
    circles = get_circles(df, calib_data, LH)

    # Grab two circles to test
    C1 = circles[0]
    C2 = circles[1] 

    sol = intersect_ellipses(C1, C2)

    linf, Cinf, H_affinity, H_projective = compute_correcting_homography(sol, [C1, C2])

    #TODO: Finish error computing code
    # And make the code that matches both frames of reference, for comparison

    # Calculate the L2 distance error
    error = np.linalg.norm(
        df[[LH+'_hom_x',LH+'_hom_y']].values - 
        df[['real_x_mm','real_y_mm']].values, 
        axis=1)
    
    errors.append(error[start_idx:end_idx])
    ####################################################################################
    ###                                 Plot Results                                 ###
    ####################################################################################

    lh2_data = {'x':    df[LH+'_hom_x'].values,
                'y':    df[LH+'_hom_y'].values,
                'time': df['time_s'].values}

    camera_data = { 'x':    df['real_x_mm'].values,
                    'y':    df['real_y_mm'].values,
                    'time': df['time_s'].values}

    plot_trajectory_and_error(lh2_data, camera_data, error, start_time, end_time)

errors = np.hstack(errors)
# Print the RMSE, MAE and STD
print(f"Error Mean = {errors.mean()}mm")
print(f"Root Mean Square Error = {np.sqrt((errors**2).mean())} mm")
print(f"Error std = {errors.std()}mm ")
print(f"Number of data point = {errors.shape}")

plot_error_histogram(errors)