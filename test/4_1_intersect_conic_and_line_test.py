# Intersect and plot an arbitrary conic (circle or Ellipses), with an arbitrary line
# Plots results that are not imaginary
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

####################### OPTIONS ############################
## Options of conics, uncomment one.
# Circle of radius 2 centered at (2,2)
# x^2 + y^2 - 4x - 4y + 4
A = np.array([  [ 1,   0,  -2],
                [ 0,   1,  -2],
                [-2,  -2,   4]])

# Ellipses of width 5 and height 2 (mayor axis == x-axis) centered at (6,2)
# 0.16x^2 + 1y^2 - 1.92x - 4y + 8.76
# A = np.array([  [ 0.16,   0,  -0.96],
#                 [ 0,   1,  -2],
#                 [-0.96,  -2,  8.76]])


## Line options in homogeneous coordinates, uncomment one.
l = np.array([1, -1, 1])  # Line, slope 45deg, passing through (0,1)
# l = np.array([1, -1, 0])  # Line, slope 45deg, passing through the origin
# l = np.array([1, -1, -3])   # Line, slope 45deg, passing through (0,-3)
# l = np.array([1, 0, -6])    # Line, vertical, passing through (x=6)


######################## FUNCTION ###########################

def extract_ellipse_params(A, B, C, D, E, F):
    # Matrix of the quadratic form
    conic_matrix = np.array([[A, B / 2], [B / 2, C]])
    
    # Translation vector (for completing the square)
    translation = np.array([D, E]) / (-2)
    
    # Find the center of the ellipse
    center = np.linalg.solve(conic_matrix, translation)
    
    # Substitute h and k into the equation to find the constant E'
    # F' = A * h^2 + C * k^2 - F
    F_prime = A * center[0]**2 + C * center[1]**2 - F

    # Semi-major and semi-minor axes
    if F_prime <= 0:
        raise ValueError("Invalid coefficients, the result is not an ellipse.")

    a = np.sqrt(abs(F_prime / A))  # Length of semi-major/minor axes squared
    b = np.sqrt(abs(F_prime / C))

    # # Eigenvalue decomposition to find the axis lengths and rotation angle
    eigenvalues, eigenvectors = np.linalg.eig(conic_matrix)
    
    # Rotation angle is the angle of the eigenvector associated with the largest eigenvalue
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Return the center, axes lengths, and angle
    return center, a, b, angle

def plot_conic_matrix_ellipse(conic_matrix, ax, color):
    # Extract the elements from the matrix
    A = conic_matrix[0, 0]
    B = 2 * conic_matrix[0, 1]  # Note: we double since the matrix stores B/2
    C = conic_matrix[1, 1]
    D = 2 * conic_matrix[0, 2]
    E = 2 * conic_matrix[1, 2]
    F = conic_matrix[2, 2]

    # Form the general quadratic equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    # Now solve for the center, axes, and angle of the ellipse.

    center, a, b, angle = extract_ellipse_params(A, B, C, D, E, F)

    # Generate ellipse points for plotting
    ellipse = plt.matplotlib.patches.Ellipse(xy =  (center[0], center[1]), 
                                             width = 2 * a,
                                             height = 2 * b,
                                             angle = angle, 
                                             edgecolor = color, 
                                             facecolor = 'none')
    
    ax.add_patch(ellipse)

def plot_homogeneous_line(line, ax, x_range=(-10, 10)):
    """
    Plots a line given in homogeneous coordinates (a, b, c) where the line is ax + by + c = 0.
    
    Parameters:
    - line: A list or tuple of 3 elements [a, b, c], representing the line in homogeneous coordinates.
    - x_range: A tuple defining the range of x-values to plot the line over.
    """
    a = line[0]
    b = line[1]
    c = line[2]
    
    # Ensure the line is not vertical (b != 0), if it is vertical, handle it separately
    if b != 0:
        # Solve for y = (-a/b)x - (c/b)
        def line_eq(x):
            return (-a / b) * x - (c / b)
        
        # Choose two x-values at the bounds of the x_range
        x_vals = np.array([x_range[0], x_range[1]])
        y_vals = line_eq(x_vals)
        
        # Plot the line using only two points
        ax.plot(x_vals, y_vals, color='xkcd:sky blue', label=f'line: {a}x + {b}y + {c} = 0')
    
    else:
        # If b = 0, the line is vertical (x = -c/a), just plot a vertical line
        x_vert = -c / a
        ax.axvline(x=x_vert, color='xkcd:sky blue', linestyle='--', label=f'Vertical line x = {x_vert}')
    
def intersect_line_with_conic(l, A):
    """
    Intersects a line (l) with a conic (A) and returns a list of points with the real and complex intersection.
    might return None if there are no intersection
    """
    # Make sure tau (last element of the line) is not zero.

    # Express l as a cross product matrix
    Ml = np.array([[0,     l[2], -l[1]],
                   [-l[2], 0,     l[0]],
                   [l[1], -l[0],  0]])
    
    # Calculate alpha
    B = Ml.T @ A @ Ml
    if (abs(l[2]) > 1e-5):
        tau = l[2]
        alpha = 1/tau * np.emath.sqrt( -1* np.linalg.det(B[0:2, 0:2]))
    else:
        lmbd = l[0]
        alpha = 1/lmbd * np.emath.sqrt( -1* np.linalg.det(B[1:3, 1:3]))

    # If more than one solution is given by the square root, choose the first one
    if type(alpha) == np.ndarray: alpha = alpha[0]

    # Intersect the line and the conic
    C = B + alpha * Ml

    # Find a non-zero element of C and get the two intersecting points.
    found_flag = False
    # for i in reversed(range(C.shape[0])):
    for i in range(C.shape[0]):
        if found_flag: break
        for j in range(C.shape[1]):
            if abs(C[i][j]) > 1e-5:

                p = C[i,:]  # get the full row
                q = C[:,j]  # get the full column 
                found_flag = True
                break

    # de-homogenize the points
    p = p[0:2]/p[2]
    q = q[0:2]/q[2]

    # return intersecting points
    return p,q

########################## MAIN #############################

# prepare the plot
fig = plt.figure(layout="constrained", figsize=(5,4))
gs = GridSpec(3, 3, figure = fig)
ax = fig.add_subplot(gs[0:3, 0:3])
ax.set_aspect('equal', 'box')

# Plot original conics
plot_conic_matrix_ellipse(A, ax, 'xkcd:blue')
ax.autoscale()

# Create middle degenerate conic
p,q = intersect_line_with_conic(l, A)
print(f"intersections:\np={p}\np={q}")

# plot homogeneous lines
plot_homogeneous_line(l, ax, x_range=(-10, 10))

# plot intersection points
if np.all(np.isreal(p)) and np.all(np.isreal(q)):
    ax.scatter(p[0], p[1], color="xkcd:orange", label="interection points")
    ax.scatter(q[0], q[1], color="xkcd:orange")

ax.set_title('Ellipse from Conic Matrix')
ax.grid(True)
ax.set_xlim([-1,5])
ax.set_ylim([-1,5])
ax.legend()
    
plt.show()