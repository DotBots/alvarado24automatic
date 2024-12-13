import json
import numpy as np
import pandas as pd
import cv2
from dateutil import parser
import re

####################################################################################
###                                 Private                                      ###
####################################################################################

def process_calibration(calib_data):
    
    # Create the nested dictionary structure needed
    calib_data['corners_lh2_proj'] = {}
    calib_data['corners_lh2_proj']['LHA'] = {}
    calib_data['corners_lh2_proj']['LHB'] = {}

    # Project calibration points 
    for corner in ['tl','tr','bl','br']:
        # Project the points
        c1a = np.array([calib_data['corners_lh2_count'][corner]['LHA_count_1']])
        c2a = np.array([calib_data['corners_lh2_count'][corner]['LHA_count_2']])
        c1b = np.array([calib_data['corners_lh2_count'][corner]['LHB_count_1']])
        c2b = np.array([calib_data['corners_lh2_count'][corner]['LHB_count_2']])
        pts_A = LH2_count_to_pixels(c1a, c2a, 0)
        pts_B = LH2_count_to_pixels(c1b, c2b, 1)

        # Add it back to the calib dictionary
        calib_data['corners_lh2_proj']['LHA'][corner] = pts_A[0]
        calib_data['corners_lh2_proj']['LHB'][corner] = pts_B[0]

    return calib_data

####################################################################################
###                                  Public                                      ###
####################################################################################


def import_data(data_file, mocap_file, calib_file):

    # Read the files.
    lh_data = pd.read_csv(data_file, index_col=0, parse_dates=['timestamp'])
    mocap_data = pd.read_csv(mocap_file, parse_dates=['timestamp'])
    with open(calib_file, 'r') as json_file:
        calib_data = json.load(json_file)
    # Get which scene number we are processing
    scene_id = re.search(r"scene_\d+", data_file).group(0)
    lh2_calib_time = calib_data[scene_id]["timestamps_lh2"]

    # Add a Z=0 axis to the Camera (X,Y) coordinates.
    # exp_data['z'] = 0.0

    # Convert the strings to datetime objects
    for key in lh2_calib_time:
        lh2_calib_time[key] = [parser.parse(ts) for ts in lh2_calib_time[key]]

    # Get a timestamp column of the datetime.
    for df in [lh_data, mocap_data]:
        df['time_s'] = df['timestamp'].apply(lambda x: x.timestamp() )

    # Convert mocap corners to numpy arrays
    for corner in ['tl', 'tr', 'bl', 'br']:
        calib_data[scene_id]['corners_mm'][corner] = np.array(calib_data[scene_id]['corners_mm'][corner])

    # Slice the calibration data and add it to the  data dataframe.
    tl = lh_data.loc[ (lh_data['timestamp'] > lh2_calib_time["tl"][0]) & (lh_data['timestamp'] < lh2_calib_time["tl"][1])].mean(axis=0, numeric_only=True)
    tr = lh_data.loc[ (lh_data['timestamp'] > lh2_calib_time["tr"][0]) & (lh_data['timestamp'] < lh2_calib_time["tr"][1])].mean(axis=0, numeric_only=True)
    bl = lh_data.loc[ (lh_data['timestamp'] > lh2_calib_time["bl"][0]) & (lh_data['timestamp'] < lh2_calib_time["bl"][1])].mean(axis=0, numeric_only=True)
    br = lh_data.loc[ (lh_data['timestamp'] > lh2_calib_time["br"][0]) & (lh_data['timestamp'] < lh2_calib_time["br"][1])].mean(axis=0, numeric_only=True)
        # Save the calibration data.
    calib_data[scene_id]['corners_lh2_count'] = {'tl':tl,
                                 'tr':tr,
                                 'bl':bl,
                                 'br':br,
                                 }
    # Calculate the projected LH points from the polynomial counts of the corners
    calib_data[scene_id] = process_calibration(calib_data[scene_id])

    # slice the datasets to be in the same timeframe.
    # Slice LH to Mocap
    start = mocap_data['timestamp'].iloc[0]  # Use the a point about 250ms later than the start of the dataset, to address the time delay correction that we will do later when we interpolate the data.
    end   = mocap_data['timestamp'].iloc[-1]
    lh_data = lh_data.loc[ (lh_data['timestamp'] > start) & (lh_data['timestamp'] < end)]
    # Slice Mocap to LH
    start = lh_data['timestamp'].iloc[0]  
    end   = lh_data['timestamp'].iloc[-1]
    mocap_data = mocap_data.loc[ (mocap_data['timestamp'] > start) & (mocap_data['timestamp'] < end)]

   
    ## Interpolate the Mocap data to match the LH data.
    mocap_np = {'time': mocap_data['time_s'].to_numpy(),
                'x':    mocap_data['x'].to_numpy(),
                'y':    mocap_data['y'].to_numpy(),
                'z':    mocap_data['z'].to_numpy()}
    
    lh_time = lh_data['time_s'].to_numpy()

    # Offset the camera timestamp to get rid of the communication delay.
    mocap_np['time'] += 265000e-6 # seconds
    mocap_np['x_interp_lh'] = np.interp(lh_time, mocap_np['time'],  mocap_np['x'])
    mocap_np['y_interp_lh'] = np.interp(lh_time, mocap_np['time'],  mocap_np['y'])
    mocap_np['z_interp_lh'] = np.interp(lh_time, mocap_np['time'],  mocap_np['z'])
    

    merged_data = pd.DataFrame({
                          'timestamp' : lh_data['timestamp'],
                          'time_s' : lh_data['time_s'],
                          'LHA_count_1' : lh_data['LHA_count_1'],
                          'LHA_count_2' : lh_data['LHA_count_2'],
                          'LHB_count_1' : lh_data['LHB_count_1'],
                          'LHB_count_2' : lh_data['LHB_count_2'],
                          'real_x_mm': mocap_np['x_interp_lh'],
                          'real_y_mm': mocap_np['y_interp_lh'],
                          'real_z_mm': mocap_np['z_interp_lh']}
                          )

    return merged_data, calib_data[scene_id]

def read_calibration_file(calib_filename):
    """
    Reads the coordinates for the 4 calibration points in:
    - Overhead 4K camera frame
    - lighthouse camera frame
    - Actual world coordinates in centimeters.
    """

    # Load the file with the corner locations
    with open(calib_filename) as f:
        data = json.load(f)

    # Read the corner locations
    c_px  = data['GX010145.MP4']['corners_px']
    c_cm  = data['GX010145.MP4']['corners_cm']
    c_lh2 = data['GX010145.MP4']['corners_lh2']

    # Define Source and Destination points for the Homography calculations
    pts_src_px  = np.array([c_px['tl'], c_px['tr'], c_px['br'], c_px['bl']])     # pixels
    pts_src_lh2 = np.array([c_lh2['tl'], c_lh2['tr'], c_lh2['br'], c_lh2['bl']]) # pixels
    pts_dst     = np.array([c_cm['tl'], c_cm['tr'], c_cm['br'], c_cm['bl']])     # centimeters

    return pts_src_px, pts_src_lh2, pts_dst

def LH2_count_to_pixels(count_1, count_2, mode):
    """
    Convert the sweep count from a single lighthouse into pixel projected onto the LH2 image plane
    ---
    count_1 - int - polinomial count of the first sweep of the lighthouse
    count_2 - int - polinomial count of the second sweep of the lighthouse
    mode - int [0,1] - mode of the LH2, let's you know which polynomials are used for the LSFR. and at which speed the LH2 is rotating.
    """
    periods = [959000, 957000]

    # Translate points into position from each camera
    a1 = (count_1*8/periods[mode])*2*np.pi  # Convert counts to angles traveled in the weird 40deg planes, in radians
    a2 = (count_2*8/periods[mode])*2*np.pi   

    # Transfor sweep angles to azimuth and elevation coordinates
    azimuth   = (a1+a2)/2 
    elevation = np.pi/2 - np.arctan2(np.sin(a2/2-a1/2-60*np.pi/180),np.tan(np.pi/6)) 

    # Project the angles into the z=1 image plane
    pts_lighthouse = np.zeros((len(count_1),2))
    for i in range(len(count_1)):
        pts_lighthouse[i,0] = -np.tan(azimuth[i])
        pts_lighthouse[i,1] = -np.sin(a2[i]/2-a1[i]/2-60*np.pi/180)/np.tan(np.pi/6) * 1/np.cos(azimuth[i])

    # Return the projected points
    return pts_lighthouse

def camera_to_world_homography(df, calib_data,):
    """
    Calculate the homography transformation between src_corners and dst_corners.
    And apply that transformation to df
    
    Parameters
    ----------
    df : dataframe with {'x', 'y'} columns
        points to transform.
    src_corners : array_like, shape(4,2)
        4 corresponding points in df' frame of reference
    dst_corners : array_like, shape(4,2)
        4 corresponding points in the target frame of reference

    Returns
    -------
    output_points : array_like, shape (N,2)
        points transformed to dst_corners frame of reference
    """

    for LH in ['LHA', 'LHB']:
        # Extract the calibration points needed to calibrate the homography.
        pts_src = np.array([calib_data['corners_lh2_proj'][LH][key][0:2] for key in ['tl', 'tr', 'br', 'bl']])  # shape(4,2)
        pts_dst = np.array([calib_data['corners_mm'][key][0:2] for key in ['tl', 'tr', 'br', 'bl']])            # shape(4,2)

        # Calculate the Homography Matrix
        H, status = cv2.findHomography(pts_src, pts_dst)

        # Prepare pixel points to convert
        input_points = df[[LH+'_proj_x', LH+'_proj_y']].to_numpy().reshape((1,-1,2))
        # pts_example = np.array([[[200, 400], [1000, 1500], [3000, 2000]]], dtype=float)  # Shape of the input array must be (1, n_points, 2), note the double square brackets before and after the points.

        # Run the transformation
        output_points = cv2.perspectiveTransform(input_points, H)
        output_points = output_points.reshape((-1, 2))                  # We can reshape the output so that the points look like [[3,4], [1,4], [5,1]]
                                                                    # They are easier to work with like this, without all that double square bracket non-sense
        # save results to the main dataframe
        df[LH+'_hom_x'] = output_points[:,0]                              
        df[LH+'_hom_y'] = output_points[:,1]                              

    return df

def reorganize_data(xy_data, timestamp):
    """
    Create a dictionary of arrays to easily manipulate the data later on.
    With the following keys ('x', 'y', 't'), where 't' is in unix epoch

    Parameters
    ----------
    xy_data : array_like, float (N,2)
        X-Y position data
    timestamp : dataframe
        datarame coulmn with the data timestamp as strings of UTC datetimes

    Returns
    -------
    data : dictionary
        values:
            'x': array, float (N,)
                Original data X axis
            'y': array, float (N,)
                Original data Y axis
            'time': array, int (N,)
                Original data timestamps in unix epoch (microseconds)

    """
    # Convert the dataframe timestamp to an array of unix epochs
    time_s = pd.to_datetime(timestamp)
    time_s = time_s.apply(lambda x: x.timestamp())
    time_s = time_s.to_numpy()

    # Reorganize the data to make it easier to manipulate in numpy (it's easier to do linear interpolation in numpy, instead of pandas.)
    data = {'time':     time_s,
                'x':    xy_data[:,0],
                'y':    xy_data[:,1],}
    
    return data

def interpolate_camera_to_lh2(camera_data, lh2_data):
    """
    Interpolate the camera data to the lh2 timebase,
    so that a 1-to-1 accuracy comparison is possible. 

    Parameters
    ----------
    camera_data : array, shape (N,2)
        camera X-Y points
    camera_timebase : array, (N,)
        timestamps of the camera data in unisx epoch (microseconds)
    lh2_timebase : array, (M,)
        timestamps of the lh2 data in unisx epoch (microseconds)

    Returns
    -------
    interp_data: Dict
        Dictionary of numpy arrays with the interpolated LH2 data.
        values:
            'x': array, float (N,)
                Interpolated data X axis
            'y': array, float (N,)
                Interpolated data Y axis
            'time': array, int (N,)
                interpolated data timestamps in unix epoch (microseconds)

    """

    # Offset the camera timestamp to calibrate for the communication delay.
    camera_data['time'] += 318109e-6 # seconds

    # Interpolate the camera data against the lh2
    interpolated_x = np.interp(lh2_data['time'], camera_data['time'],  camera_data['x'])
    interpolated_y = np.interp(lh2_data['time'], camera_data['time'],  camera_data['y'])

    # Put the interpolated data in a dictionary matching the structure of the input data.
    interp_data = {'time':  lh2_data['time'],
                    'x':    interpolated_x,
                    'y':    interpolated_y,}
    
    return interp_data


def find_closest_point(data, t):
    """
    returns the [x,y] pair closest to a particular t.
    Useful to compare if two datasets are time-aligned.

    Parameters
    ----------
    data : Dict
        Dictionary of numpy arrays with the interpolated LH2 data.
        values:
            'x': array, float (N,)
                Interpolated data X axis
            'y': array, float (N,)
                Interpolated data Y axis
            'time': array, int (N,)
                interpolated data t in unix epoch (microseconds)
    t :
        time of the point to find


    Returns
    -------
    point : array, float (2,)
        point in  data closest in time to t

    """

    idx = np.abs(data['time'] - t).argmin()
    point = [data['x'][idx], data['y'][idx]]
    return point


def get_start_end_index(data, t_start, t_stop):
    """
    Return the index closest to a particular timestamps to use in ploting

    Parameters
    ----------
    data : Dict
        Dictionary of numpy arrays with the interpolated LH2 data.
        values:
            'x': array, float (N,)
                Interpolated data X axis
            'y': array, float (N,)
                Interpolated data Y axis
            'time': array, int (N,)
                interpolated data t in unix epoch (microseconds)
    t_start : float
        Start timestamp for the plot
    t_stop : float
        Start timestamp for the plot


    Returns
    -------
    point : array, float (2,)
        point in  data closest in time to t

    """

    idx = np.abs(data['time'] - t).argmin()
    point = [data['x'][idx], data['y'][idx]]
    return point