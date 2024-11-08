import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

####################### OPTIONS ############################

# Circle, p=[2,2], r=2
# x^2 + y^2 - 4x - 4y + 4 = 0
# A = np.array([[1, 0, -2],
#               [0, 1, -2],
#               [-2, -2, 4]])

# Ellipses, p[3,2], a=10, b=1
# 1/10 x^2 + 1y^2 - 3/5x - 4y + 39/10 = 0  
# B = np.array([[1/10,   0, -3/10],
#               [0,      1,    -2],
#               [-3/10, -2, 39/10]])


# Intersection points:
# (x=0.0296444, y=1.65693)
# (x=0.0296444, y=2.34307)
# (x=3.74813,   y=1.02839)
# (x=3.74813,   y=2.97161)

# Intersection lines:
#  0.353553 x  + 1.64645 = y
# -0.353553 x  + 2.35355 = y



# # Ellipse centered at (2,2), h-radius = 1.4, v-radius = 0.6
# # 0.5x^2 + 3.33333y^2 - 2x - 13.33333y + 14.3333333
A = np.array([  [ 0.5,   0,  -1],
                [ 0,   3.33333,  -6.6666],
                [-1,  -6.6666,   14.333333]])


# # Circle of radius 1 centered at (2,2)
# # x^2 + y^2 - 4x - 4y + 7
B = np.array([  [ 1,   0,  -2],
                [ 0,   1,  -2],
                [-2,  -2,   7]])




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

def split_degenerate_conic(A):
    """
    Split the degenerate conic A into two homogeneous cordinates lines g and h
    """

    # Check if the conic is an X (rank 2), or double lines (rank 1)
    rank = np.linalg.matrix_rank(A)

    if rank == 3:
        raise ValueError("Conic is not degenerate, Rank = 3")
    if rank == 1:
        C = A
    if rank == 2:
        # Extract the important matrix coefficient
        a = A[0, 0]
        b = A[0, 1] 
        c = A[1, 1]
        d = A[0, 2]
        e = A[1, 2]
        f = A[2, 2]

        # TODO Figure out how to handle horizontal and vertical lines
        # Handle the case of a completely vertical or completely horizontal line
        # Vertical
        if np.isclose(b,0) and np.isclose(c,0) and np.isclose(e, 0):
            roots = np.roots([a, 2*d, f])
            g = np.array([1, 0, -roots[0]])
            h = np.array([1, 0, -roots[1]])
            return g,h
        
        # Horizontal
        if np.isclose(a,0) and np.isclose(b,0) and np.isclose(d, 0):
            roots = np.roots([c, 2*e, f])
            g = np.array([0, 1, -roots[0]])
            h = np.array([0, 1, -roots[1]])
            return g,h


        # If it is a more complex type of line, continue with the full algorithm
        # Compute the adjoint matrix
        B = np.array([[  c*f-e*e, -(b*f-d*e),  b*e-d*c ],
                    [-(b*f-d*e),  a*f-d*d, -(a*e-b*d) ],
                    [  b*e-d*c, -(a*e-b*d), (a*c-b*b) ]])
        
        # Find the smallest (non-zero) diagonal element of the adjoint 
        diag_B = np.diag(B)
        for i in range(3):
            if not np.isclose(diag_B[np.argsort(diag_B)[i]],0): 
                idx = np.argsort(diag_B)[i]
                break

        # Get the intersection point of the lines
        beta = np.emath.sqrt(B[idx,idx])
        p = B[:,idx]/beta
        # Normalize the point p
        if not np.isclose(p[2], 0):
            p = p/p[2]
        p = np.real_if_close(p)

        # Get the corss product matrix
        Mp = np.array([[   0,   p[2], -p[1] ],
                    [-p[2],    0,   p[0] ],
                    [ p[1], -p[0],    0]])
        
        # Get the degenerate  Rank-1 matrix
        C = A + Mp

    # Find a non-zero element and get the corresponding row and column
    found = False
    for i in range(3):
        for j in range(3):

            if not np.isclose( C[i][j] , 0):
                g = C[i,:]
                h = C[:,j]

                found = True
            if found: break
        if found: break
    
    return g,h

def mix_conics_into_degenerate(A, B):

    # Mix Conics A and B into degenerate Conic C

    # form the cubic equation det|A + lambda*B| = 0
    a = np.linalg.det(B)
    b = (np.linalg.det(A + B) + np.linalg.det(A - B)) / 2 - np.linalg.det(A)
    c = (np.linalg.det(A + B) - np.linalg.det(A - B)) / 2 - np.linalg.det(B)
    d = np.linalg.det(A)

    # Normalize the cubic equation
    magnitude = np.linalg.norm([a,b,c,d])
    a /= magnitude
    b /= magnitude
    c /= magnitude
    d /= magnitude

    # Solve cubic equation
    delta0 = b*b - 3*a*c
    delta1 = 2*b*b*b - 9*a*b*c + 27*a*a*d
    delta01 = delta1 * delta1 - 4 * delta0 * delta0 * delta0

    # Accomodate sign
    s0 = np.sign(delta0) + 1*( np.sign(delta0) == 0)
    s1 = np.sign(delta1) + 1*( np.sign(delta1) == 0)

    omega_p = s1 * np.emath.power((np.abs(delta1) + np.emath.sqrt(delta01))/2, 1/3)
    omega_m = s0*s1 * np.emath.power(s0*(np.abs(delta1) - np.emath.sqrt(delta01))/2, 1/3)

    # Get the real root of the cubic equation
    lmbd = -3*a
    mu = b + np.real(omega_m + omega_p)
    C = lmbd * A + mu * B

    # Get the other 2 roots of the cubic equation
    mu1 = b + np.real_if_close(np.exp(-1j*np.pi*2/3) * omega_m + np.exp(1j*np.pi*2/3) * omega_p)
    mu2 = b + np.real_if_close(np.exp(-1j*np.pi*2/3*2)*omega_m + np.exp(1j*np.pi*2/3*2)*omega_p)

    C1 = lmbd * A + mu1 * B
    C2 = lmbd * A + mu2 * B

    # Make sure close to zero components are actually zero
    for i in range(3):
        for j in range(3):

            if np.isclose( C[i][j] , 0):
                C[i,j] = 0
            if np.isclose( C1[i][j] , 0):
                C1[i,j] = 0
            if np.isclose( C2[i][j] , 0):
                C2[i,j] = 0
    
    return C, C1, C2
    # return C

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
    if not np.isclose( l[2] , 0):
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
    found = False
    for i in range(3):
        for j in range(3):

            if not np.isclose( C[i][j] , 0):
                p = C[i,:]
                q = C[:,j]

                found = True
            if found: break
        if found: break

    # de-homogenize the points
    p = p[0:2]/p[2]
    q = q[0:2]/q[2]

    # return intersecting points
    return p,q

def plot_conic_matrix_ellipse(conic_matrix, ax, color, label=""):
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
                                             facecolor = 'none',
                                             label = label)
    
    ax.add_patch(ellipse)

def plot_homogeneous_line(line, ax, x_range=(-10, 10), color="xkcd:green",label=""):
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
        ax.plot(x_vals, y_vals, color="xkcd:green",label=label)
    
    else:
        # If b = 0, the line is vertical (x = -c/a), just plot a vertical line
        x_vert = -c / a
        ax.axvline(x=x_vert, color='r', linestyle='--', label=label)


########################## MAIN ###########################




########################## PLOT #############################

# prepare the plot
fig = plt.figure(layout="constrained", figsize=(5,4))
gs = GridSpec(3, 3, figure = fig)
ax = fig.add_subplot(gs[0:3, 0:3])
ax.set_aspect('equal', 'box')

# Plot original conics
plot_conic_matrix_ellipse(A, ax, 'xkcd:blue', label = "conic 1")
plot_conic_matrix_ellipse(B, ax, 'xkcd:red', label = "conic 2")
ax.autoscale()

# Create middle degenerate conic
# C = mix_conics_into_degenerate(A, B)
C, C1, C2 = mix_conics_into_degenerate(A, B)

# TODO: find a way to auto select the best possible  degenerate mix conic. Rank 2, and 2 parallel lines
# Create middle degenerate conic
g,h = split_degenerate_conic(C1)
print(f"lines:\np={g}\np={h}")

# Create middle degenerate conic
p,q = intersect_line_with_conic(g, A)
p1,q1 = intersect_line_with_conic(h, A)
print(f"intersections:\np={p}\np={q}")

# plot homogeneous lines
plot_homogeneous_line(g, ax, x_range=(-10, 10), color = "xkcd:green", label="degenerate mix")
plot_homogeneous_line(h, ax, x_range=(-10, 10), color = "xkcd:green")

# plot intersection points
if np.all(np.isreal(p)) and np.all(np.isreal(q)):
    ax.scatter(p[0], p[1], color="xkcd:orange", label="interection points")
    ax.scatter(q[0], q[1], color="xkcd:orange")
    ax.scatter(p1[0], p1[1], color="xkcd:orange")
    ax.scatter(q1[0], q1[1], color="xkcd:orange")

ax.set_title('Ellipse from Conic Matrix')
ax.grid(True)
# ax.set_xlim([-1,None])
# ax.set_ylim([-1,None])
    
plt.show()