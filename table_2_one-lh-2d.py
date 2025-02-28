# Results for Table 2, One-LH 2D algorithm accuracy.
# Results get printed to the terminal

# import the necessary packages
import numpy as np
import matplotlib
matplotlib.use('TKAgg')

from functions.data_processing import   import_data, \
                                        LH2_count_to_pixels, \
                                        camera_to_world_homography, \
                                        correct_similarity_distrotion

from functions.plotting import plot_trajectory_and_error, plot_error_histogram, plot_projected_LH_views

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

    # Import data
    df, calib_data, (start_idx, end_idx) = import_data(LH_data_file, mocap_data_file, calib_file)

    ####################################################################################
    ###                            Process Data                                      ###
    ####################################################################################
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
        ###                             1LH 2D algorithm                                 ###
        ####################################################################################

        # Convert the LH2 pixel data to the world coordinate frame of reference.
        pts_dst = np.array([[0,400],[400,400],[400,0],[0,0]])  # calibration 40x40 cm square
        df = camera_to_world_homography(df, calib_data, pts_dst)

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

        # Uncomment to plot examples of the reconstructed trajectories
        # start_time = df.loc[start_idx]['time_s'] - df.iloc[0]['time_s']
        # end_time   = df.loc[end_idx]['time_s']   - df.iloc[0]['time_s']
        # plot_trajectory_and_error(lh2_data, camera_data, error, start_time, end_time)

errors = np.hstack(errors)
# Print the RMSE, MAE and STD and,
# Plot the histogram error.
plot_error_histogram(errors)