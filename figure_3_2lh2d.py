# import the necessary packages
import json
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from functions.data_processing import   import_data, \
                                        LH2_count_to_pixels, \
                                        read_calibration_file, \
                                        camera_to_world_homography, \
                                        reorganize_data, \
                                        interpolate_camera_to_lh2, \
                                        find_closest_point, \
                                        twoLH_solve_2d_scene_get_Rtn, \
                                        twoLH_process_calibration, \
                                        twoLH_solve_point_plane, \
                                        twoLH_scale_LH2_to_real_size, \
                                        twoLH_interpolate_cam_data, \
                                        twoLH_correct_perspective

from functions.plotting import plot_trajectory_and_error, plot_error_histogram, plot_projected_LH_views, twoLH_plot_reconstructed_3D_scene

####################################################################################
###                               Options                                        ###
####################################################################################
# Define which of the 6 experimetns you want to plot
errors=[]
for experiment_number in [1,2]:

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
    ###                             2LH 2D algorithm                                 ###
    ####################################################################################


    # Solve the scene to find the transformation R,t from LHA to LHB
    solution_1, solution_2, zeta = twoLH_solve_2d_scene_get_Rtn(pts_lighthouse_A, pts_lighthouse_B)
    # solution_1, solution_2, zeta = fil_solve_2d(pts_lighthouse_A, pts_lighthouse_B)
    if experiment_number == 1:
        t_star, R_star, n_star = solution_1
    if experiment_number == 2:
        t_star, R_star, n_star = solution_2

    # Convert the for 4 calibration points from a LH projection to a 3D point
    calib_data = twoLH_process_calibration(n_star, zeta, calib_data)

    # Transform LH projected points into 3D
    point3D = twoLH_solve_point_plane(n_star, zeta, pts_lighthouse_A)

    # Scale up the LH2 points
    # TODO: After this line, point3D becomes NaN
    lh2_scale, calib_data, point3D = twoLH_scale_LH2_to_real_size(calib_data, point3D)

    df['LH_x'] = point3D[:,0]
    df['LH_y'] = point3D[:,1]   # We need to invert 2 of the axis because the LH2 frame Z == depth and Y == Height
    df['LH_z'] = point3D[:,2]   # But the dataset assumes X = Horizontal, Y = Depth, Z = Height

    # Find the transform that superimposes one dataset over the other.
    df,_,_ = twoLH_correct_perspective(df, calib_data)

    # Calculate the L2 distance error
    error = np.linalg.norm(
    df[['LH_Rt_x', 'LH_Rt_y', 'LH_Rt_z']].values - 
    df[['real_x_mm', 'real_y_mm', 'real_z_mm']].values, 
    axis=1)

    errors.append(error[start_idx:end_idx])

errors = np.hstack(errors)
####################################################################################
###                                 Plot Results                                 ###
####################################################################################


# Plot Error Histogram
plot_error_histogram(errors)

# Plot 3D reconstructed scene
twoLH_plot_reconstructed_3D_scene( df)
