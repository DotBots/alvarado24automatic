import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
import sympy as sp

####################### OPTIONS ############################
x, y = sp.symbols('x y', complex=True)
# Circle of radius 1 centered at (1,1)
# x^2 + y^2 - 2x - 2y + 1
A = np.array([  [ 1,   0,  -1],
                [ 0,   1,  -1],
                [-1,  -1,   1]])
eqA = x**2 + y**2 - 2 * x - 2 * y + 1
eqA_inf = x**2 + y**2

# Circle of radius 1 centered at (2,2)
# x^2 + y^2 - 4x - 4y + 7
B = np.array([  [ 1,   0,  -2],
                [ 0,   1,  -2],
                [-2,  -2,   7]])
eqB = x**2 + y**2 - 4 * x - 4 * y + 7
eqB_inf = x**2 + y**2
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

def mix_conics_into_degenerate(A, B):

    # Mix Conics A and B into degenerate Conic C

    # form the cubic equation det|A + lambda*B| = 0
    a = np.linalg.det(B)
    b = (np.linalg.det(A + B) + np.linalg.det(A - B)) / 2 - np.linalg.det(A)
    c = (np.linalg.det(A + B) - np.linalg.det(A - B)) / 2 - np.linalg.det(B)
    d = np.linalg.det(A)

    # Solve cubic equation
    delta0 = b**2 - 3*a*c
    delta1 = 2*b**3 - 9*a*b*c + 27*d*a**2
    for i in range(3):
        omega_min = cuberoot((delta1 - np.emath.sqrt(delta1**2 - 4*delta0**3))/2)[i]
        omega_plus = cuberoot((delta1 + np.emath.sqrt(delta1**2 - 4*delta0**3))/2)[i]

        # Get the solution points
        k = 0
        sol = {}
        for k in range(3):
            sol[k] = - (b + np.e**(1j*2*np.pi*k/3)*omega_plus + np.e**(1j*-2*np.pi*k/3)*omega_min ) / 3*a
            sol[k] = np.real_if_close(sol[k])

        # Get lambda and mu
        lmbd = -3*a
        mu = b + omega_min + omega_plus

        if np.isreal(sol[0]) or np.isreal(sol[1]) or np.isreal(sol[2]): 
            break

    if not(np.isreal(sol[0]) or np.isreal(sol[1]) or np.isreal(sol[2])):
        raise ValueError("no real roots")
    
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

# plot homogeneous lines
plot_homogeneous_line(g, ax, x_range=(-10, 10))
plot_homogeneous_line(h, ax, x_range=(-10, 10))

# Intersect lines with conic
p,q = intersect_line_with_conic(g,A)

# plot intersection points
if np.all(np.isreal(p)) and np.all(np.isreal(q)):
    ax.scatter(p[0], p[1], color="xkcd:orange", label="interection points")
    ax.scatter(q[0], q[1], color="xkcd:orange")

ax.set_title('Ellipse from Conic Matrix')
ax.grid(True)
# ax.set_xlim([-1,None])
# ax.set_ylim([-1,None])
    
plt.show()