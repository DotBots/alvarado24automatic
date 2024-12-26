# test every circle pair combination for computing the dual conic
# to find the best homography.
#


# import the necessary packages
import json
import numpy as np
import pandas as pd
import itertools
from datetime import datetime

# import matplotlib
# matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from functions.data_processing import   import_data, \
                                        LH2_count_to_pixels, \
                                        get_circles, \
                                        intersect_ellipses, \
                                        compute_correcting_homography, \
                                        apply_corrective_homography, \
                                        correct_similarity_distrotion, \
                                        distance_between_conics, \
                                        conic_eccentricity, \
                                        apply_conic_homography, \
                                        compute_best_correcting_homography


from functions.plotting import plot_trajectory_and_error, plot_error_histogram, plot_projected_LH_views

####################################################################################
###                               Options                                        ###
####################################################################################
# Define which of the 6 experimetns you want to plot
experiment_number = 1

####################################################################################
###                            Read Dataset                                      ###
####################################################################################

# Create the dataframe that will store the results
out_df = pd.DataFrame(columns=['num_circles','experiment_number','LH','mae_all','rms_all','std_all','mae_experiment','rms_experiment','std_experiment','eccentricity'])

for experiment_number in [1,2]:

    # file with the data to analyze
    LH_data_file = f'./dataset/scene_{experiment_number}/lh_data.csv'
    mocap_data_file = f'./dataset/scene_{experiment_number}/mocap_data.csv'
    calib_file = './dataset/calibration.json'

    # Import data
    df, calib_data, (start_idx, end_idx) = import_data(LH_data_file, mocap_data_file, calib_file)
    start_time = df.loc[start_idx]['time_s'] - df.iloc[0]['time_s']
    end_time   = df.loc[end_idx]['time_s']   - df.iloc[0]['time_s']

    # Compute all possible combination of circles
    num_circles=10
    circle_combination = []
    for n in range(2,num_circles+1):
        circle_combination += list(itertools.combinations(range(num_circles),n))


    ####################################################################################
    ###                            Process Data                                      ###
    ####################################################################################
    for LH in ['LHA', 'LHB']:
        # Project sweep angles on to the z=1 image plane
        pts_lighthouse_A = LH2_count_to_pixels(df['LHA_count_1'].values, df['LHA_count_2'].values, 0)
        pts_lighthouse_B = LH2_count_to_pixels(df['LHB_count_1'].values, df['LHB_count_2'].values, 1)

        # Add the LH2 projected matrix into the dataframe that holds the info about what point is where in real life.
        df['LHA_proj_x'] = pts_lighthouse_A[:,0]
        df['LHA_proj_y'] = pts_lighthouse_A[:,1]
        df['LHB_proj_x'] = pts_lighthouse_B[:,0]
        df['LHB_proj_y'] = pts_lighthouse_B[:,1]

        ####################################################################################
        ###                             Conic algorithm                                 ###
        ####################################################################################

        # Get the conic equations for all of the circles.
        all_circles = get_circles(df, calib_data, LH)

        # Go over all possible combination of 2 to 10 circles to calculate the error per number of circle
        for circle_indices in circle_combination:

            circles = [all_circles[i] for i in circle_indices]

            H_projective, eccentricity = compute_best_correcting_homography(circles)

            df = apply_corrective_homography(df, H_projective)

            # And make the code that matches both frames of reference, for comparison
            df,_,_ = correct_similarity_distrotion(df, calib_data)

            # Calculate the L2 distance error
            all_error = np.linalg.norm(
                df[[LH+'_Rt_x', LH+'_Rt_y']].values - 
                df[['real_x_mm', 'real_y_mm']].values, 
                axis=1)
        
            # limit the error to only the part of the experiment where you were dragging the DotBot around. 
            trajectory_error = all_error[start_idx:end_idx]

            ## Compute all the values you need to save.
            #  num_circles , LH , mae_all , rms_all , std_all , mae_experiment , rms_experiment , std_experiment , eccentricity 
            out_df.loc[len(out_df)] = [len(circles),
                                        experiment_number,
                                        LH,
                                        all_error.mean(),
                                        np.sqrt((all_error**2).mean()),
                                        all_error.std(),
                                        trajectory_error.mean(),
                                        np.sqrt((trajectory_error**2).mean()),
                                        trajectory_error.std(),
                                        eccentricity]
            print(out_df.tail(1))
            
            out_df.to_csv("conic_error_vs_num_circles_csv",index=True)
        
####################################################################################
###                                 Plot Results                                 ###
####################################################################################

