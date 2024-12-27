import numpy as np
import seaborn as sns

# Fix to avoid Type 3 fonts on the figures
# http://phyletica.org/matplotlib-fonts/
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_trajectory_and_error(lh2_data, camera_data, error, start_time, end_time):
    """
    Plots a superposition of the ground truth trajectory and estimated LH2 trajectory.
    As well as a separate subplot with the error.

    Parameters
    ----------
    lh2_data : Dict
        Dictionary of numpy arrays with the LH2 data.
        values:
            'x': array, float (N,)
                X axis data
            'y': array, float (N,)
                Y axis data
            'time': array, int (N,)
                t in unix epoch (microseconds)
    camera_data : Dict
        Same as lh2_data, but with the camera data
    error : array, float, shape (N,2)
        Euclidean error between lh2_data and camera_data
    start_time : float
        start time for the plot (in seconds)
    end_time : float
        end time for the plot (in seconds)

    """

    # Find the indexes of the start and end times for the plots
    t_i = camera_data['time'][0] + start_time # seconds
    t_o = t_i + end_time
    # Find the closest point to this time
    c_sta = np.abs(camera_data['time'] - t_i).argmin()
    l_sta = np.abs(lh2_data['time'] - t_i).argmin()
    # e_sta = np.abs(ekf_np['time'] - t_i).argmin()

    # Find the closest point to this time
    c_sto = np.abs(camera_data['time'] - t_o).argmin()
    l_sto = np.abs(lh2_data['time'] - t_o).argmin()


    # Plot the results
    fig = plt.figure(layout="constrained", figsize=(5,4))
    gs = GridSpec(6, 3, figure = fig)
    # Define individual subplots
    xy_ax    = fig.add_subplot(gs[0:4, 0:3])
    error_ax = fig.add_subplot(gs[4:6, :])
    axs = (xy_ax, error_ax)


    # X vs. Y plots
    xy_ax.plot(lh2_data['x'][l_sta:l_sto], lh2_data['y'][l_sta:l_sto], '--',color='b', lw=1, label="lighthouse")
    xy_ax.plot(camera_data['x'][c_sta:c_sto], camera_data['y'][c_sta:c_sto], '-',color='k', lw=1, label="ground truth")
    xy_ax.scatter(lh2_data['x'][l_sta:l_sto], lh2_data['y'][l_sta:l_sto],color='b', alpha=0.3)
    xy_ax.scatter(camera_data['x'][c_sta:c_sto], camera_data['y'][c_sta:c_sto], alpha=0.3,color='k', lw=1)
    # xy_ax.scatter([80,120,120,80], [80,80,120,120], edgecolor='r', facecolor='red', lw=1, label="markers")
    # Plot one synchronized point to check for a delay.
    idx = error.argmax()
    xy_ax.scatter(camera_data['x'][idx], camera_data['y'][idx], edgecolor='k', facecolor='xkcd:red', lw=1)
    xy_ax.scatter(lh2_data['x'][idx], lh2_data['y'][idx], edgecolor='k', facecolor='xkcd:pink', lw=1)

    error_ax.plot(lh2_data['time'][l_sta:l_sto] - t_i, error[l_sta:l_sto], '-',color='b', lw=1, label="LH error")

    # Add labels and grids
    for ax in axs:
        ax.grid()
        ax.legend()
    xy_ax.axis('equal')
    # 
    xy_ax.set_xlabel('X [mm]')
    xy_ax.set_ylabel('Y [mm]')
    #
    # xy_ax.set_xlim([60, 160])

    error_ax.set_xlabel('Time [s]')
    error_ax.set_ylabel('Error [mm]')
    #
    error_ax.set_xlim([0, lh2_data['time'][l_sto] - lh2_data['time'][l_sta]])


    plt.savefig('Result-A-1lh_2d-example.pdf')

    plt.show()


def plot_projected_LH_views(pts_a, pts_b, extra_pts=None):
    """
    Plot the projected views from each of the lighthouse
    """

    fig = plt.figure(layout="constrained")
    gs = GridSpec(6, 3, figure = fig)
    lh1_ax    = fig.add_subplot(gs[0:3, 0:3])
    lh2_ax = fig.add_subplot(gs[3:6, 0:3])
    axs = (lh1_ax, lh2_ax)

    t = np.arange(pts_a.shape[0])

    # 2D plots - LH2 perspective
    lh1_ax.scatter(pts_a[:,0], pts_a[:,1], c=t,cmap='inferno', alpha=0.5, lw=1, label="LH1")
    lh2_ax.scatter(pts_b[:,0], pts_b[:,1], c=t,cmap='inferno', alpha=0.5, lw=1, label="LH2")
    lh1_ax.scatter(pts_a[0,0], pts_a[0,1], color='xkcd:blue', alpha=1, lw=1, label="top_point")
    lh2_ax.scatter(pts_b[0,0], pts_b[0,1], color='xkcd:blue', alpha=1, lw=1, label="top_point")

    # Plot the groundtruth of the top point to check if the conversion is working well.
    if extra_pts != None:
        LHA, LHC = extra_pts
        lh1_ax.scatter(LHA[0], LHA[1], color='xkcd:pink', alpha=1, lw=1, label="top point real")
        lh2_ax.scatter(LHC[0], LHC[1], color='xkcd:pink', alpha=1, lw=1, label="top point real")

    # Add labels and grids
    for ax in axs:
        ax.grid()
        ax.legend()
    lh1_ax.axis('equal')
    lh2_ax.axis('equal')
    # 
    lh1_ax.set_xlabel('U [px]')
    lh1_ax.set_ylabel('V [px]')
    #
    lh2_ax.set_xlabel('U [px]')
    lh2_ax.set_ylabel('V [px]')
    #
    # lh1_ax.invert_yaxis()
    # lh2_ax.invert_yaxis()

    plt.show()


def plot_error_histogram(errors):
    """

    Plot a histogram of the provided distance errors

    Parameters
    ----------
    errors : array_like, float, shape (N,2)
        Array of euclidean error

    """

    # print the mean and standard deviation
    print(f"Mean Absolute Error = {errors.mean()} mm")
    print(f"Root Mean Square Error = {np.sqrt((errors**2).mean())} mm")
    print(f"Error Standard Deviation = {errors.std()} mm")

    # Plot the results
    fig = plt.figure(layout="constrained", figsize=(5,4))
    gs = GridSpec(3, 3, figure = fig)
    # Define individual subplots
    hist_ax    = fig.add_subplot(gs[0:3, 0:3])
    axs = [hist_ax,]

    # Sea-born KDE histogram plot
    sns.histplot(data=errors,  bins=50, ax=hist_ax, linewidth=0, color="xkcd:baby blue")
    hist_ax.set_xlim((0, 22))
    ax2 = hist_ax.twinx()
    sns.kdeplot(data=errors, ax=ax2, label="density", color="xkcd:black", linewidth=1, linestyle='--')

    # Plot the mean line
    hist_ax.axvline(x=errors.mean(), color='xkcd:red', label="Mean")
    # Trick to get the legend unified between the TwinX plots
    hist_ax.plot([], [], color="xkcd:black", linestyle='--', label = 'density')

    # Add labels and grids
    for ax in axs:
        ax.legend()
    
    # Configure the plot options
    xticks_locs = np.linspace(0, 20, 5)  # 5 x-ticks from 0 to 10
    hist_ax.set_xticks(xticks_locs)
    hist_ax.set_xlabel('Distance Error [mm]')
    hist_ax.set_ylabel(f'Measurements (n = {errors.shape[0]})')

    # Save and show figure
    plt.savefig('Result-B-1lh_2d-histogram.pdf')
    plt.show()

def twoLH_plot_reconstructed_3D_scene(df):
    """
    Plot a 3D scene with the traingulated points previously calculated
    ---
    input:
    point3D - array [3,N] - triangulated points of the positions of the LH2 receveier
    t_star  - array [3,1] - Translation vector between the first and the second lighthouse basestation
    R_star  - array [3,3] - Rotation matrix between the first and the second lighthouse basestation
    point3D - array [3,N] - second set of pointstriangulated points of the positions of the LH2 receveier
    """
    ## Plot the two coordinate systems
    #  x is blue, y is red, z is green
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    # First lighthouse:

    lh_points = df[['LH_Rt_x','LH_Rt_y','LH_Rt_z']].values
    mocap_points = df[['real_x_mm', 'real_y_mm', 'real_z_mm']].values
    ax.scatter(mocap_points[:,0], mocap_points[:,1], mocap_points[:,2], alpha=0.1, color="xkcd:red", label="Mocap")
    ax.scatter(lh_points[:,0],lh_points[:,1],lh_points[:,2], alpha=0.1, label="LH")

   
    ax.axis('equal')
    ax.legend()
    ax.set_title('2D solved scene - 3D triangulated Points')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')   

    plt.show()

def plot_projected_fitted_ellipses(pts, circles, circles_2=None):
    """
    Plot the projected views from each of the lighthouse
    """

    fig = plt.figure(layout="constrained")
    gs = GridSpec(6, 3, figure = fig)
    lh_ax    = fig.add_subplot(gs[0:6, 0:6])
    axs = (lh_ax,)

    t = np.arange(pts.shape[0])

    # 2D plots - LH2 perspective
    lh_ax.scatter(pts[:,0], pts[:,1], c=t,cmap='inferno', alpha=0.5, lw=1, label="LH1")

    for C in circles:
        plot_conic(lh_ax, C, 'xkcd:blue')

    if circles_2 is not None:
        for C in circles_2:
            plot_conic(lh_ax, C, 'xkcd:green')

    # Plot the groundtruth of the top point to check if the conversion is working well.
    # if extra_pts != None:
    #     LHA, LHC = extra_pts
    #     lh1_ax.scatter(LHA[0], LHA[1], color='xkcd:pink', alpha=1, lw=1, label="top point real")

    # Add labels and grids
    for ax in axs:
        ax.grid()
        ax.legend()
    lh_ax.axis('equal')
    lh_ax.set_xlabel('U [px]')
    lh_ax.set_ylabel('V [px]')


    plt.show()

def plot_conic(ax, circle, color='blue', label=None):
    """
    Plots a conic section defined by Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 on a given axis.
    
    Parameters:
        ax (matplotlib.axes.Axes): The subplot axis to draw the conic section.
        A, B, C, D, E, F (float): Coefficients of the conic equation.
        color (str): Color of the conic plot. Default is 'blue'.
        label (str): Label for the conic plot. Default is None.
    """

    A, B, C, D, E, F = circle

    # Define the function for the conic equation
    def conic_eq(x, y):
        return A * x**2 + B * x * y + C * y**2 + D * x + E * y + F

    # Set up a grid for plotting
    x = np.linspace(-1, 1, 1000)  # Adjust the range if needed
    y = np.linspace(-1, 1, 1000)
    X, Y = np.meshgrid(x, y)

    # Evaluate the conic equation on the grid
    Z = conic_eq(X, Y)

    # Plot the contour where the conic equation equals zero
    ax.contour(X, Y, Z, levels=[0], colors=color, label=label)
