# Figure 5, Conic Rectification Trajectory example.
# Several examples of trajectories of different trajectories, experiments and LH perspective are plotted in quick sucession.
# In this order:
# Experiment 1 - LHA perspective
# Experiment 1 - LHB perspective
# Experiment 2 - LHA perspective
# Experiment 2 - LHB perspective


# import the necessary packages
import numpy as np
import matplotlib
matplotlib.use('TKAgg')

from functions.data_processing import   import_data, \
                                        LH2_count_to_pixels, \
                                        get_circles, \
                                        apply_corrective_homography, \
                                        correct_similarity_distrotion, \
                                        compute_best_correcting_homography

from functions.plotting import plot_trajectory_and_error, plot_error_histogram, plot_projected_LH_views

####################################################################################
###                                Option                                        ###
####################################################################################
# Which crcles to use for the calibration
circle_indices = list(range(0,10,2))  # 5 equally interspersed circles throughout the dataset
# circle_indices = list(range(0,10))  # all 10 circles

# Which Lighthouse perspectives to use for computing the accuracy
LH_indices = ["LHA", "LHB"]
# Which of the two available experiments to use for computing the accuracy.
experiment_indices = [1,2]

####################################################################################
###                                Main                                          ###
####################################################################################

errors=[]
for experiment_number in experiment_indices:
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
    for LH in LH_indices:
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

        # Get the fitted conic equations for all of the circles.
        all_circles = get_circles(df, calib_data, LH)

        # Select some of the circles and compute the best homography
        circles = [all_circles[i] for i in circle_indices]
        H_projective, eccentricity = compute_best_correcting_homography(circles)

        # Undistort the data with the computed homography
        df = apply_corrective_homography(df, H_projective)

        # And make the code that matches both frames of reference, for comparison
        df,_,_ = correct_similarity_distrotion(df, calib_data)

        # Calculate the L2 distance error
        error = np.linalg.norm(
            df[[LH+'_Rt_x', LH+'_Rt_y']].values - 
            df[['real_x_mm', 'real_y_mm']].values, 
            axis=1)
        
        # Add the errors of this experiments to the general error list
        errors.append(error[start_idx:end_idx])
        ####################################################################################
        ###                                 Plot Results                                 ###
        ####################################################################################

        lh2_data = {'x':    df[LH+'_Rt_x'].values,
                    'y':    df[LH+'_Rt_y'].values,
                    'time': df['time_s'].values}

        camera_data = { 'x':    df['real_x_mm'].values,
                        'y':    df['real_y_mm'].values,
                        'time': df['time_s'].values}

        # Plot an example trajectory
        plot_trajectory_and_error(lh2_data, camera_data, error, start_time, end_time-50, decimate=True) # -50:  magic number to make this plot fill the entire available space. Otherwise it looks ugly
