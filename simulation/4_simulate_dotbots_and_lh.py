import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

####################### OPTIONS ############################
# position of the circles
dotbot_1 = np.array([1,0.3,0])  
dotbot_2 = np.array([1,-0.3,0])

radius = 0.05 # 10cm, diameter
samples = 100 # how many samples to use per circle

# Pose of the LH
lh_t = np.array([0,0,1]) # Origin, z = 1m 
lh_R, _ = cv2.Rodrigues(np.array([0, np.pi/4, 0 ])) # pointing towards X-axis, elevation angle 45
# lh_R, _ = cv2.Rodrigues(np.array([0, 0., 0 ])) # pointing towards X-axis, elevation angle 45

######################## FUNCTION ###########################

# 1. Define a 3D Circle in space. Defaults to 10cm diameter circle on the X-Y plane, centered at (0,0,0),  with 100 samples
def generate_circle_3d(radius=0.05, center=(0,0,0), num_points=100):


    # Parametric angle for the circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    
    # Circle in 3D centered at (0, 0, 0)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)  # Circle lies in the XY plane
    
    # Stack to create a set of 3D points
    circle_points = np.vstack((x, y, z))

    # Add center to each point to translate the circle
    circle_points[0, :] += center[0]  # X-coordinate offset
    circle_points[1, :] += center[1]  # Y-coordinate offset
    circle_points[2, :] += center[2]  # Z-coordinate offset
    
    return circle_points.T

# 2. Add Gaussian noise to the 3D points
def add_noise(points, noise_std):
    noisy_points = points + np.random.normal(0, noise_std, points.shape)
    return noisy_points

# 4. Project 3D points onto the 2D image plane using the pinhole camera model
def project_points_pinhole(points, camera_t, camera_R):
    # Translate and Rotation points to camera coordinates
    rot_pts = camera_R.T @ (points - camera_t).T
    
    elevation = np.arctan2( rot_pts[2], np.sqrt(rot_pts[0]**2 + rot_pts[1]**2))
    azimuth = np.arctan2(rot_pts[1], rot_pts[0])

    proj_pts = np.array([np.tan(azimuth),       # horizontal pixel  
                             np.tan(elevation) * 1/np.cos(azimuth)]).T  # vertical   pixel 
    

    # Return projected 2D points
    return proj_pts

########################## MAIN ###########################

# 5. Main function to generate the projection
def main():

    
    # Generate the 3D circle points
    circle_1 = generate_circle_3d(radius, dotbot_1, samples)
    circle_2 = generate_circle_3d(radius, dotbot_2, samples)
    
    # Add Gaussian noise to the circle points
    noise_std = 0.1  # Standard deviation of the noise
    # noisy_circle_3d = add_noise(circle_3d, noise_std)
    
    # Project the noisy circle points through the pinhole camera
    proj_points_1 = project_points_pinhole(circle_1, lh_t, lh_R)
    proj_points_2 = project_points_pinhole(circle_2, lh_t, lh_R)
    
    # Turn the data into (N,2) shape if it is in (2,) shape
    if len(proj_points_1.shape) == 1: proj_points_1 = proj_points_1[np.newaxis,:]
    if len(proj_points_2.shape) == 1: proj_points_2 = proj_points_2[np.newaxis,:]

########################## PLOT ###########################

    # Plot the resulting 2D projection (should look like an ellipse)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, aspect='equal', adjustable='box')
    # ax.set_aspect('equal')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.scatter(proj_points_1[:,0], proj_points_1[:,1], c='r', label='points 1')
    ax.scatter(proj_points_2[:,0], proj_points_2[:,1], c='r', label='points 2')
    ax.set_title("Circle viewed through a Pinhole Camera with Orientation (Projected Ellipse)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.set_ylim(ax.get_xlim())
    ax.grid(True)

    # Plot 3D scene
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_aspect('equal')
    ax2.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    ax2.set_proj_type('ortho')

    # Plot the lighthouse orientation
    arrow = np.array([1,0,0]).reshape((-1,1))
    ax2.quiver(lh_t[0],lh_t[1],lh_t[2], (lh_R @ arrow)[0], (lh_R @ arrow)[1], (lh_R @ arrow)[2], length=0.4, color='xkcd:red')
    ax2.scatter(dotbot_1[0],dotbot_1[1],dotbot_1[2], color='xkcd:blue', label='dotbot 1', s=50)
    ax2.scatter(circle_1[:,0],circle_1[:,1],circle_1[:,2], color='xkcd:blue', label='dotbot 1', s=50)
    ax2.scatter(dotbot_2[0],dotbot_2[1],dotbot_2[2], color='xkcd:green', label='dotbot 2', s=50)
    ax2.scatter(circle_2[:,0],circle_2[:,1],circle_2[:,2], color='xkcd:green', label='dotbot d', s=50)
    ax2.set_xlim(0, 2)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(0, 1)

    plt.show()

# Run the main function
main()

