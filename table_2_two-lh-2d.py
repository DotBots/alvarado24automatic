# Results for Table 2, Two-LH 2D algorithm accuracy.
# Results get printed to the terminal

# import the necessary packages
import numpy as np
import matplotlib
matplotlib.use('TKAgg')

from functions.data_processing import   import_data, \
                                        LH2_count_to_pixels, \
                                        twoLH_solve_2d_scene_get_Rtn, \
                                        twoLH_process_calibration, \
                                        twoLH_solve_point_plane, \
                                        twoLH_scale_LH2_to_real_size, \
                                        twoLH_correct_perspective

from functions.plotting import plot_error_histogram, plot_projected_LH_views, twoLH_plot_reconstructed_3D_scene

####################################################################################
###                               Options                                        ###
####################################################################################

errors=[]
for experiment_number in [1,2]:

    ####################################################################################
    ###                            Read Dataset                                      ###
    ####################################################################################

    # file with the data to analyze
    LH_data_file = f'./dataset/scene_{experiment_number}/lh_data.csv'
    mocap_data_file = f'./dataset/scene_{experiment_number}/mocap_data.csv'
    calib_file = './dataset/calibration.json'

    # Import datataset
    df, calib_data, (start_idx, end_idx) = import_data(LH_data_file, mocap_data_file, calib_file)

    ####################################################################################
    ###                            Process Data                                      ###
    ####################################################################################

    # Project sweep angles on to the z=1 image plane
    pts_lighthouse_A = LH2_count_to_pixels(df['LHA_count_1'].values, df['LHA_count_2'].values, 0)
    pts_lighthouse_B = LH2_count_to_pixels(df['LHB_count_1'].values, df['LHB_count_2'].values, 1)

    # Plot the LH view for debbuging
    # plot_projected_LH_views(pts_lighthouse_A, pts_lighthouse_B)

    # Add the LH2 projected matrix into the dataframe that holds the info about which point is where in real life.
    df['LHA_proj_x'] = pts_lighthouse_A[:,0]
    df['LHA_proj_y'] = pts_lighthouse_A[:,1]
    df['LHB_proj_x'] = pts_lighthouse_B[:,0]
    df['LHB_proj_y'] = pts_lighthouse_B[:,1]

    ####################################################################################
    ###                             2LH 2D algorithm                                 ###
    ####################################################################################

    # Solve the scene to find the transformation R,t from LHA to LHB
    solution_1, solution_2, zeta = twoLH_solve_2d_scene_get_Rtn(pts_lighthouse_A, pts_lighthouse_B)
    # Of the two solutions, we need to choose which one is the correct one, we do this manually.
    if experiment_number == 1:
        t_star, R_star, n_star = solution_1
    if experiment_number == 2:
        t_star, R_star, n_star = solution_2

    # Convert the for 4 calibration points from a LH projection to a 3D point
    calib_data = twoLH_process_calibration(n_star, zeta, calib_data)

    # Transform LH projected points into 3D
    point3D = twoLH_solve_point_plane(n_star, zeta, pts_lighthouse_A)

    # Scale up the LH2 points
    lh2_scale, calib_data, point3D = twoLH_scale_LH2_to_real_size(calib_data, point3D)

    df['LH_x'] = point3D[:,0]
    df['LH_y'] = point3D[:,1]
    df['LH_z'] = point3D[:,2]

    # Find the transform that superimposes one dataset over the other.
    df,_,_ = twoLH_correct_perspective(df, calib_data)

    # Calculate the L2 distance error
    error = np.linalg.norm(
    df[['LH_Rt_x', 'LH_Rt_y', 'LH_Rt_z']].values - 
    df[['real_x_mm', 'real_y_mm', 'real_z_mm']].values, 
    axis=1)

    # Add the errors of this experiments to the general error list
    errors.append(error[start_idx:end_idx])

errors = np.hstack(errors)
####################################################################################
###                                 Plot Results                                 ###
####################################################################################

# Plot Error Histogram
plot_error_histogram(errors)

# Plot 3D reconstructed scene
twoLH_plot_reconstructed_3D_scene( df)
