import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math

import sympy as sp

####################### OPTIONS ############################
# Circle of radius 2 centered at (2,2)
# x^2 + y^2 - 4x - 4y + 4
A = np.array([  [ 1,   0,  -2],
                [ 0,   1,  -2],
                [-2,  -2,   4]])


# Ellipses of radius 2 and 1 (mayor axis == x-axis) centered at (4,2)
# x^2 + 4y^2 - 12x - 16y + 48
B = np.array([  [ 1/4,   0,  -1],
                [ 0,   1,  -2],
                [-1,  -2,  7]])

# Ellipses of width 5 and height 2 (mayor axis == x-axis) centered at (6,2)
# 0.16x^2 + 1y^2 - 1.92x - 4y + 8.76
# B = np.array([  [ 0.16,   0,  -0.96],
#                 [ 0,   1,  -2],
#                 [-0.96,  -2,  8.76]])

# A = 0.16   # Coefficient of x^2
# B = 0   # Coefficient of xy (no xy term)
# C = 1   # Coefficient of y^2
# D = -1.92 # Coefficient of x
# E = -4 # Coefficient of y
# F = 8.76  # Constant term

# Ellipses of width 6 and height 2 (mayor axis == x-axis) centered at (2,2) roughly.
# B = np.array([  [ 1/9,   0,  -2/9],
#                 [ 0,     1,  -2],
#                 [-2/9,  -2,   7]])


# B = np.array([  [ 0.105263,            0,      -0.210527],
#                 [ 0,             1.66667,      -3.333335],
#                 [-0.210527,    -3.333335,       5.39772]])

# 0.105263 x^2 - 0.421053 x + 1.66667 y^2 - 6.66667 y + 5.39772 = 0
# ((x-2)^2)/9.5 + ((y-2)^2)/0.6 = 1.3^2

######################## FUNCTION ###########################

def cuberoot( z ):
    z = complex(z)
    x = z.real
    y = z.imag
    mag = abs(z)
    arg = math.atan2(y,x)
    return [ mag**(1./3) * np.exp( 1j*(arg+2*n*math.pi)/3 ) for n in range(1,4) ]

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
    # if F_prime <= 0:
    #     raise ValueError("Invalid coefficients, the result is not an ellipse.")

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

def mix_conics_into_degenerate(A, B):

    # Mix Conics A and B into degenerate Conic C

    # Get the cubic equation constant
    # C = lambda * A + mu * B
    # alpha * labda^3 + beta * labda^2 * mu + sigma * lambda * mu^2 + delta * mu^3
    alpha = np.linalg.det(np.vstack((A[0], A[1], A[2])))

    beta =  np.linalg.det(np.vstack((A[0], A[1], B[2]))) + \
            np.linalg.det(np.vstack((A[0], B[1], A[2]))) + \
            np.linalg.det(np.vstack((B[0], A[1], A[2]))) 
    
    sigma = np.linalg.det(np.vstack((A[0], B[1], B[2]))) + \
            np.linalg.det(np.vstack((B[0], A[1], B[2]))) + \
            np.linalg.det(np.vstack((B[0], B[1], A[2]))) 

    delta = np.linalg.det(np.vstack((B[0], B[1], B[2])))

    # Solve the cubic equation
    W = -2*(beta**3) + 9*alpha*beta*sigma - 27*(alpha**2)*delta
    D = -(beta**2)*(sigma**2) + 4*alpha*(sigma**3) + 4*(beta**3)*delta - 18*alpha*beta*sigma*delta + 27*(alpha**2)*(delta**3)
    Q = W - alpha*np.emath.sqrt(27*D)
    # Check if there is more than one result of the square root
    # if type(Q) == np.ndarray: Q = Q[0]
    R = cuberoot(4*Q)[0]
    L = np.array([2*(beta**2) - 6*alpha*sigma, -beta, R]).reshape((3,1))
    M = np.array([3*alpha*R, 3*alpha, 2*3*alpha]).reshape((3,1))

    # Get the solution points
    w = -1/2 + 1j*np.sqrt(3/4)
    ww = np.array([[w,    1,  w**2],
                   [1,    1,  1],
                   [w**2, 1,  w]])
    
    lmbd_v = ww @ L
    mu_v = ww @ M

    # Check if any result is real, and choose it
    found_result = False
    for i in range(lmbd_v.shape[0]):
        if np.isreal(lmbd_v[i]) and np.isreal(mu_v[i]) and ((lmbd_v[i] > 1e-5) or (mu_v[i] > 1e-5)):
            lmbd = lmbd_v[i]
            mu = mu_v[i]
            found_result = True
            break

    # If no result is real, just choose a non-zero result
    if not found_result:
        for i in range(lmbd_v.shape[0]):
            if ((lmbd_v[i] > 1e-5) or (mu_v[i] > 1e-5)):
                lmbd = lmbd_v[i]
                mu = mu_v[i]
                found_result = True
                break

    # If no valid results is result
    if not found_result:
        raise ValueError("All mu and lambda == 0")
    
    # Return degenerate conic
    C = lmbd * A + mu * B
    return C

def split_degenerate_conic(A):

    """
    Split the degenerate conic A into two homogeneous cordinates lines g and h
    """

    # Get upper triangular portion of degenerate conic A
    B = np.triu(A)

    # Find a non-zero diagonal element of B and calculate the intersection point p
    for i in range(B.shape[0]):
    # for i in reversed(range(B.shape[0])):

        if abs(B[i][i]) > 1e-5:
            beta = np.sqrt(B[i][i])

            p = B[:,i] / beta

            break

    # Express p as a cross product matrix
    Mp = np.array([[0,     p[2], -p[1]],
                   [-p[2], 0,     p[0]],
                   [p[1], -p[0],  0]])
    
    # get the matrix from which we will extract the points.
    C = A + Mp
    
    # Find a non-zero element of C and get the two intersecting lines.
    found_flag = False
    # for i in reversed(range(C.shape[0])):
    for i in range(C.shape[0]):
        if found_flag: break
        for j in range(C.shape[1]):
            if abs(C[i][j]) > 1e-5:

                g = C[i,:]  # get the full row
                h = C[:,j]  # get the full column 
                found_flag = True
                break


    # Normalize lines before returning them
    g = g / np.linalg.norm(g[0:2])
    h = h / np.linalg.norm(h[0:2])


    # return the two intersecting lines of the degenerate conic
    return g,h

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
        ax.plot(x_vals, y_vals, label=f'{a}x + {b}y + {c} = 0')
    
    else:
        # If b = 0, the line is vertical (x = -c/a), just plot a vertical line
        x_vert = -c / a
        ax.axvline(x=x_vert, color='r', linestyle='--', label=f'Vertical line x = {x_vert}')

########################## MAIN ###########################


# prepare the plot
fig = plt.figure(layout="constrained", figsize=(5,4))
gs = GridSpec(3, 3, figure = fig)
ax = fig.add_subplot(gs[0:3, 0:3])
ax.set_aspect('equal', 'box')

# Plot original conics
plot_conic_matrix_ellipse(A, ax, 'xkcd:blue')
plot_conic_matrix_ellipse(B, ax, 'xkcd:red')
ax.autoscale()

# Create middle degenerate conic
C = mix_conics_into_degenerate(A, B)
g,h = split_degenerate_conic(C)


#Symbolic solver
# Define the variables
x, y = sp.symbols('x y')
# Define the two quadratic equations
eq1 = x**2 + y**2 - 4 * x - 4 * y + 4
# eq2 = 0.105263 * x**2 + 1.66667 * y**2 - 0.421053 * x - 6.66667 * y + 5.39772  # Last B
eq2 = 1 * x**2 + 4 * y**2 - 12 * x - 16 * y + 48  
# x^2 + 4y^2 - 12x - 16y + 48
# Solve the system of equations
solutions = sp.solve([eq1, eq2], (x, y))

# x^2 + y^2 - 4x - 4y + 4
# 0.105263 x^2 - 0.421053 x + 1.66667 y^2 - 6.66667 y + 5.39772 = 0

sol = np.real_if_close(np.array(solutions).astype(complex))
ax.scatter(sol[:,0], sol[:,1], color="xkcd:orange")


ax.set_title('Ellipse from Conic Matrix')
ax.grid(True)
# ax.set_xlim([-1,None])
# ax.set_ylim([-1,None])
    
plt.show()