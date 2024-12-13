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


def plot_error_histogram(errors):
    """

    Plot a histogram of the provided distance errors

    Parameters
    ----------
    errors : array_like, float, shape (N,2)
        Array of euclidean error

    """

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