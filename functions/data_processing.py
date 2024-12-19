import json
import numpy as np
import pandas as pd
import cv2
from dateutil import parser
import re
from skspatial.objects import Plane
import sympy as sp

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
        c1a = np.array([calib_data['corners_lh2_count'][corner]['lha_count_0']])
        c2a = np.array([calib_data['corners_lh2_count'][corner]['lha_count_1']])
        c1b = np.array([calib_data['corners_lh2_count'][corner]['lhb_count_0']])
        c2b = np.array([calib_data['corners_lh2_count'][corner]['lhb_count_1']])
        pts_A = LH2_count_to_pixels(c1a, c2a, 0)
        pts_B = LH2_count_to_pixels(c1b, c2b, 1)

        # Add it back to the calib dictionary
        calib_data['corners_lh2_proj']['LHA'][corner] = pts_A[0]
        calib_data['corners_lh2_proj']['LHB'][corner] = pts_B[0]

    return calib_data

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2 """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

####################################################################################
###                                  Public                                      ###
####################################################################################


def import_data(data_file, mocap_file, calib_file, time_offset = None, experiment_indices = None):

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

    # Get time offset from the calib file if it's not specified
    if time_offset == None:
        time_offset = calib_data[scene_id]["dataset_time_offset"]

    # Get the start and end index of the experiment dataset from the calib file if it's not specified
    if experiment_indices == None:
        start_idx = calib_data[scene_id]["experiment_timeframe"]["start_idx"]
        end_idx = calib_data[scene_id]["experiment_timeframe"]["end_idx"]
        experiment_indices = (start_idx, end_idx)


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
    mocap_np['time'] += time_offset # seconds
    mocap_np['x_interp_lh'] = np.interp(lh_time, mocap_np['time'],  mocap_np['x'])
    mocap_np['y_interp_lh'] = np.interp(lh_time, mocap_np['time'],  mocap_np['y'])
    mocap_np['z_interp_lh'] = np.interp(lh_time, mocap_np['time'],  mocap_np['z'])
    

    merged_data = pd.DataFrame({
                          'timestamp' : lh_data['timestamp'],
                          'time_s' : lh_data['time_s'],
                          'LHA_count_1' : lh_data['lha_count_0'],
                          'LHA_count_2' : lh_data['lha_count_1'],
                          'LHB_count_1' : lh_data['lhb_count_0'],
                          'LHB_count_2' : lh_data['lhb_count_1'],
                          'real_x_mm': mocap_np['x_interp_lh'],
                          'real_y_mm': mocap_np['y_interp_lh'],
                          'real_z_mm': mocap_np['z_interp_lh']}
                          )


    ## The mocap data plane is not perfectly horizontal, make it so.
    # Get a best plane fit
    best_plane = Plane.best_fit(merged_data[['real_x_mm', 'real_y_mm', 'real_z_mm']].values)
    normal = np.array(best_plane.normal)
    z1 = np.array([0,0,1])
    R = rotation_matrix_from_vectors(normal, z1)
    # Straighten out the plane data.
    merged_data[['real_x_mm','real_y_mm','real_z_mm']] =  (R @ merged_data[['real_x_mm','real_y_mm','real_z_mm']].values.T).T

    # Apply the straightening to the calibration corners
    for corner in ['tl', 'tr', 'bl', 'br']:
        calib_data[scene_id]['corners_mm'][corner] = (R @ calib_data[scene_id]['corners_mm'][corner].T).T

    return merged_data, calib_data[scene_id], experiment_indices

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
    for i in range(pts_lighthouse.shape[0]):
        pts_lighthouse[i,0] = -np.tan(azimuth[i])
        pts_lighthouse[i,1] = -np.sin(abs(a2[i]/2-a1[i]/2)-60*np.pi/180)/np.tan(np.pi/6) * 1/np.cos(azimuth[i])

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


####################################################################################
###                            Conic Algorithm                                   ###
####################################################################################

### Private Functions
# 3. Calculate the line at infinity from the calibration grid
def compute_line_at_infinity(proj_calib_points):
    
    # make the points homogeneous
    ones = np.ones((proj_calib_points["start"].shape[0],1))
    proj_calib_points["start"] = np.hstack((proj_calib_points["start"], ones))
    proj_calib_points["end"] = np.hstack((proj_calib_points["end"], ones))

    # Cross product of the start and end point of the line to get the line homogeneous equation.
    l1 = np.cross(proj_calib_points["start"][0], proj_calib_points["end"][0])
    l2 = np.cross(proj_calib_points["start"][1], proj_calib_points["end"][1])
    l3 = np.cross(proj_calib_points["start"][2], proj_calib_points["end"][2])
    l4 = np.cross(proj_calib_points["start"][3], proj_calib_points["end"][3])

    # Cross product the lines to get the homogeneous points at the image of the line at infinity
    p1_inf = np.cross(l1, l2)
    p2_inf = np.cross(l3, l4)

    # Cross product of the infinite points to get the image of the line at infinite.
    linf = np.cross(p1_inf, p2_inf)

    return linf

def apply_point_homography(points, H):

    # If the point only has a single dimension, add another
    if len(points.shape) == 1:
        points = points.reshape((1,-1))

    # If the points are, 2D, add a third dimension
    if points.shape[1] == 2:
        ones = np.ones((points.shape[0],1))
        points = np.hstack((points, ones))

    # Transform the points trough the Homography
    h_points = (H @ points.T).T
    # Normalize the homogeneous points
    h_points = h_points/h_points[:,2,np.newaxis]

    return h_points

def apply_conic_homography(conic, H):
    # Re arm the conic equation into its matrix form
    if len(conic) == 6:
        A, B, C, D, E, F = conic
    if conic.shape == (3,3):
        A = conic[0,0]
        B = conic[1,0]*2
        C = conic[1,1]
        D = conic[0,2]*2
        E = conic[1,2]*2
        F = conic[2,2]

    C_orig = np.array([[A,   B/2, D/2],
                  [B/2, C,   E/2],
                  [D/2, E/2, F]])
    
    # invert the Homography matrix
    Hinv = np.linalg.inv(H)

    # Perform Homography transformation
    C_homo = Hinv.T @ C_orig @ Hinv

    # Extract the parameters of the new conic
    A_h = C_homo[0,0]
    B_h = C_homo[1,0] * 2
    C_h = C_homo[1,1]
    D_h = C_homo[0,2] * 2
    E_h = C_homo[1,2] * 2
    F_h = C_homo[2,2]

    # Return the new conic parameters
    return A_h, B_h, C_h, D_h, E_h, F_h

# 1. Ellipses fitting
def fit_ellipse(points):

    x = points[:,0]
    y = points[:,1]

    # Construct the design matrix for the equation Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T

    _, _, V = np.linalg.svd(D)  # Singular Value Decomposition for more stability
    params = V[-1, :]           # Solution is in the last row of V

    a,b,c,d,e,f = params

    residual = a * x**2 + b * x*y + c * y**2 + d * x + e * y + f

    return params, residual  # Returns the coefficients [A, B, C, D, E, F]


### Public Functions

def get_circles(df:pd.DataFrame, calib_data:dict, LH: str):
    """
    Extracts the data related to calibration circles, as indicated by the calibration data.
    Fits them to a conic equation

    Params:
        df: dataset, with the following columns: ['timestamp', 'time_s', 'LHA_count_1', 'LHA_count_2', 'LHB_count_1',
                                                    'LHB_count_2', 'real_x_mm', 'real_y_mm', 'real_z_mm', 'LHA_proj_x',
                                                    'LHA_proj_y', 'LHB_proj_x', 'LHB_proj_y']

        calib_data: calibration data with the start and end timestamp of every circle

        LH: 'LHA' or 'LHB', depending of which LH we are working with,
    """
    bit = 5
    circles = []
    for id in list(calib_data['circles'].keys()):

        # Compatibility for a previous version of the code
        if id == 'quantity': continue

        # Get start and end timestamp for the circle data
        start = calib_data['circles'][id][0]
        end   = calib_data['circles'][id][1]

        # Extract the selected LH projected data
        circle_data = df.loc[ (df['timestamp'] > start) & (df['timestamp'] < end)]
        points = circle_data[[LH+'_proj_x',LH+'_proj_y']].values

        # Try to fit the data to an ellipse conic
        circle, residual = fit_ellipse(points)  

        # Add it to the list      
        circles.append(circle)

        # Print the equation for debbuging purposes
        print(f"param: {circle}, residual: {abs(residual).mean()} ")
    
    return circles


# 2. Ellipses intersection
def intersect_ellipses(C1, C2):
    """
    This function returns all imaginary intersection points of the Conic sections C1 and C2. In their standard form:
    Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    And in the homogenous form at infinite, where W=0
    Ax^2 + Bxy + Cy^2 + Dxw + Eyw + Fw^2 = 0    ; thus
    Ax^2 + Bxy + Cy^2 = 0
    """

    ## starting seeds
    seed_values = [1+1j, 1-1j, -1+1j, -1-1j]

    solutions = []
    solutions_w = []
    # Iterate over all the possible startng value
    for x0 in seed_values:
        for y0 in seed_values:
            x,y = sp.symbols('x y',complex=True)
            # Standard form
            eq1 = C1[0]*x**2 + C1[1]*x*y + C1[2]*y**2 + C1[3]*x + C1[4]*y + C1[5]
            eq2 = C2[0]*x**2 + C2[1]*x*y + C2[2]*y**2 + C2[3]*x + C2[4]*y + C2[5]

            # Homogeneous w=0 infinite equations
            eq1_w = C1[0]*x**2 + C1[1]*x*y + C1[2]*y**2
            eq2_w = C2[0]*x**2 + C2[1]*x*y + C2[2]*y**2 

            found_flag = False
            found_flag_w = False
            try:
                local_solutions = sp.nsolve([eq1, eq2], (x,y), (x0,y0))
                found_flag = True
            except:
                pass    

            try:
                local_solutions_w = sp.nsolve([eq1_w, eq2_w], (x,y), (x0,y0))
                found_flag_w = True
            except:
                pass

            if found_flag:
                # Convert solution to numpy
                numeric_solution = np.array(local_solutions, dtype=np.complex128)
                # Go one by one and get rid of floating point errors (real_if_close, close_to_zero)
                for i in range(numeric_solution.shape[0]):
                    for j in range(numeric_solution.shape[1]):

                        # Check if number is real
                        numeric_solution[i][j] = np.real_if_close(numeric_solution[i][j])

                        # Check if real part is zero
                        if np.isclose(np.real(numeric_solution[i][j]),0): numeric_solution[i][j] = 1j * np.imag(numeric_solution[i][j]) 

                        # Check if number is zero
                        if np.isclose(np.real(numeric_solution[i][j]),0) and np.isclose(np.imag(numeric_solution[i][j]),0): numeric_solution[i][j] = 0
        
                # Add to the list of solutions if it's not there. (also add the complex conjugate, because it will appear there sooner or later)
                # Standard
                if not np.any(numeric_solution == solutions):
                    solutions.append(numeric_solution)
                # Conjugate
                if not np.any(np.conjugate(numeric_solution) == solutions):
                    solutions.append(np.conjugate(numeric_solution))

            if found_flag_w:
                # Convert solution to numpy
                numeric_solution_w = np.array(local_solutions_w, dtype=np.complex128)
                # Same as above, but for the homogeneous w=0 case
                for i in range(numeric_solution_w.shape[0]):
                    for j in range(numeric_solution_w.shape[1]):

                        # Check if number is real
                        numeric_solution_w[i][j] = np.real_if_close(numeric_solution_w[i][j])

                        # Check if real part is zero
                        if np.isclose(np.real(numeric_solution_w[i][j]),0): numeric_solution_w[i][j] = 1j * np.imag(numeric_solution_w[i][j]) 

                        # Check if number is zero
                        if np.isclose(np.real(numeric_solution_w[i][j]),0) and np.isclose(np.imag(numeric_solution_w[i][j]),0): numeric_solution_w[i][j] = 0

                # Add to the list of solutions if it's not there. (also add the complex conjugate, because it will appear there sooner or later)
                # Standard
                if not np.any(numeric_solution_w == solutions_w):
                    solutions_w.append(numeric_solution_w)
                # Conjugate
                if not np.any(np.conjugate(numeric_solution_w) == solutions_w):
                    solutions_w.append(np.conjugate(numeric_solution_w))

    # Split the results into complex conjugates pairs
    sorted_results = []
    # Normal results (w=1 equation)
    if len(solutions) > 0:
        for sol in solutions:
            if not np.any(sol == sorted_results):
                sorted_results.append([sol, np.conjugate(sol)])
    # Inifite results (w=0, for when H is an affinity)
    if len(solutions_w) > 0:
        for sol in solutions_w:
            if not np.any(sol == sorted_results):
                sorted_results.append([sol, np.conjugate(sol)])    

    return sorted_results

# 8. Compute Correcting Homography 
def compute_correcting_homography(intersections, conics):

    # candidate solutions, we will store the possible solutions and return the best one.
    candidate_solutions = []
    candidate_error = []

    for sol in intersections:
        # Extract the image of the circular points
        II = np.array([sol[0][0][0],sol[0][1][0],1]).reshape((-1,1))
        JJ = np.array([sol[1][0][0],sol[1][1][0],1]).reshape((-1,1))
        # Calculate the Line at infinity
        linf = np.cross(II.reshape((-1,)), JJ.reshape((-1,))).reshape((-1,1))
        linf = linf/linf[2] # normalize by the independent element
        linf = np.real_if_close(linf)
        # Calculate the Dual Conic
        Cinf = II @ JJ.T + JJ @ II.T
        U,S,Vh = np.linalg.svd(Cinf)

        # Compute rectification up to affinity
        Hp_prime_inv = np.linalg.inv(np.array([[1, 0, 0],
                        [0, 1, 0],
                        [-linf[0][0]/linf[2][0], -linf[1][0]/linf[2][0], 1/linf[2][0]]]))
        
        # Compute rectification up to similarity
        H_sim = np.real_if_close(U)
        H_sim_inv = np.linalg.inv(H_sim)

        # Test rectification to see which one returns a proper circle (A is equal to C, and B =0)
        error = 0
        for conic in conics:
            circle = apply_conic_homography(conic, H_sim_inv)
            error += abs(circle[1]) + abs(circle[0] - circle[2]) # B + (A-C)

        # Store this candidate solution
        candidate_solutions.append([linf, Cinf, Hp_prime_inv, H_sim_inv])    
        candidate_error.append(error)    

    # Choose and return the best solution, the one with the least error
    idx = np.array(candidate_error).argmin()
    return candidate_solutions[idx]
        