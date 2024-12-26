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
                                        get_circles_sk, \
                                        intersect_ellipses, \
                                        compute_correcting_homography, \
                                        apply_corrective_homography, \
                                        correct_similarity_distrotion, \
                                        distance_between_conics, \
                                        conic_eccentricity


from functions.plotting import plot_trajectory_and_error, plot_error_histogram, plot_projected_LH_views

print()
print("sk,LH,c1,c2,mae,rms,std,dist,angle_1,angle_2,max_angle,ecc1,ecc2,ecc_dif")

####################################################################################
###                               Options                                        ###
####################################################################################
# Define which of the 6 experimetns you want to plot
experiment_number = 2

####################################################################################
###                            Read Dataset                                      ###
####################################################################################

# file with the data to analyze
LH_data_file = f'./dataset/scene_{experiment_number}/lh_data.csv'
mocap_data_file = f'./dataset/scene_{experiment_number}/mocap_data.csv'
calib_file = './dataset/calibration.json'

# Extract the number of circles
df, calib_data, (start_idx, end_idx) = import_data(LH_data_file, mocap_data_file, calib_file)
num_circles = calib_data['circles']['quantity']

for n1,n2 in itertools.combinations(range(num_circles),2):

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
        circles = get_circles_sk(df, calib_data, LH)

        # Grab two circles to test
        C1 = circles[n1]
        C2 = circles[n2] 

        sol = intersect_ellipses(C1, C2)

        linf, Cinf, H_affinity, H_projective = compute_correcting_homography(sol, [C1, C2])

        df = apply_corrective_homography(df, H_projective)

        #TODO: Finish error computing code
        # And make the code that matches both frames of reference, for comparison
        df,_,_ = correct_similarity_distrotion(df, calib_data)

        # Calculate the L2 distance error
        error = np.linalg.norm(
            df[[LH+'_Rt_x', LH+'_Rt_y']].values - 
            df[['real_x_mm', 'real_y_mm']].values, 
            axis=1)
        
        # errors.append(error[start_idx:end_idx])
        ####################################################################################
        ###                                 Plot Results                                 ###
        ####################################################################################

        # errors = np.hstack(errors)
        errors = error
        # Compute the distance between the circle centers
        start = calib_data['circles'][str(n1+1)][0]
        end   = calib_data['circles'][str(n1+1)][1]
        circle_data = df.loc[ (df['timestamp'] > start) & (df['timestamp'] < end)]
        center_1 = circle_data[['real_x_mm','real_y_mm']].mean(axis=0).values
        angle_1 = np.linalg.norm(circle_data[[LH+'_proj_x',LH+'_proj_y']].mean(axis=0).values)

        start = calib_data['circles'][str(n2+1)][0]
        end   = calib_data['circles'][str(n2+1)][1]
        circle_data = df.loc[ (df['timestamp'] > start) & (df['timestamp'] < end)]
        center_2 = circle_data[['real_x_mm','real_y_mm']].mean(axis=0).values
        angle_2 = np.linalg.norm(circle_data[[LH+'_proj_x',LH+'_proj_y']].mean(axis=0).values)

        distance = np.linalg.norm(center_1 - center_2)
        max_angle = max(angle_1,angle_2)

        # Compute the eccentricity
        ecc1 = conic_eccentricity(C1)
        ecc2 = conic_eccentricity(C2)
        ecc_diff = np.abs(ecc1 - ecc2)

        # print(f"({n1},{n2}): MAE: {errors.mean():0.2f}\tRMS: {np.sqrt((errors**2).mean()):0.2f}\tSTD: {errors.std():0.2f} - Dist: {distance:0.2f}")
        print(f"total,{LH},{n1},{n2},{errors.mean():0.2f},{np.sqrt((errors**2).mean()):0.2f},{errors.std():0.2f},{distance:0.2f},{angle_1:0.3f},{angle_2:0.3f},{max_angle:0.3f},{ecc1:0.3f},{ecc2:0.3f},{ecc_diff:0.3f}")

