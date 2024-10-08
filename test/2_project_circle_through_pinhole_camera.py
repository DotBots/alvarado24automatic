import numpy as np
import matplotlib.pyplot as plt

# 1. Define a 3D Circle in space
def generate_circle_3d(radius, num_points):
    # Parametric angle for the circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    
    # Circle in 3D centered at (0, 0, 0)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)  # Circle lies in the XY plane
    
    # Stack to create a set of 3D points
    circle_points = np.vstack((x, y, z))
    
    return circle_points

# 2. Add Gaussian noise to the 3D points
def add_noise(points, noise_std):
    noisy_points = points + np.random.normal(0, noise_std, points.shape)
    return noisy_points

# 3. Define the camera orientation using Euler angles (pitch, yaw, roll)
def get_rotation_matrix(pitch, yaw, roll):
    # Rotation matrices for pitch, yaw, and roll
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    
    # Full rotation matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R

# 4. Project 3D points onto the 2D image plane using the pinhole camera model
def project_points_pinhole(points, focal_length, camera_position, rotation_matrix):
    # Translate points to camera coordinates
    translated_points = points - np.reshape(camera_position, (3, 1))
    
    # Apply camera orientation (rotation)
    rotated_points = rotation_matrix @ translated_points
    
    # Project points onto the image plane (simple perspective projection)
    x_proj = (focal_length * rotated_points[0, :]) / rotated_points[2, :]
    y_proj = (focal_length * rotated_points[1, :]) / rotated_points[2, :]
    
    # Return projected 2D points
    return x_proj, y_proj

# 5. Main function to generate the projection
def main():
    # Circle parameters
    radius = 5
    num_points = 100  # Number of sample points on the circle
    
    # Generate the 3D circle points
    circle_3d = generate_circle_3d(radius, num_points)
    
    # Add Gaussian noise to the circle points
    noise_std = 0.1  # Standard deviation of the noise
    noisy_circle_3d = add_noise(circle_3d, noise_std)
    
    # Camera parameters
    focal_length = 10  # Distance between camera and image plane
    camera_position = np.array([0, 0, -20])  # Camera position behind the circle
    
    # Define camera orientation using Euler angles (pitch, yaw, roll)
    pitch = np.radians(10)  # Rotation around the X-axis (in radians)
    yaw = np.radians(20)    # Rotation around the Y-axis (in radians)
    roll = np.radians(0)    # Rotation around the Z-axis (in radians)
    
    # Get the rotation matrix for the camera orientation
    rotation_matrix = get_rotation_matrix(pitch, yaw, roll)
    
    # Project the noisy circle points through the pinhole camera
    x_proj, y_proj = project_points_pinhole(noisy_circle_3d, focal_length, camera_position, rotation_matrix)
    
    # Plot the resulting 2D projection (should look like an ellipse)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_proj, y_proj, c='r', label='Projected points')
    plt.title("Circle viewed through a Pinhole Camera with Orientation (Projected Ellipse)")
    plt.xlabel("X'")
    plt.ylabel("Y'")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the main function
main()

