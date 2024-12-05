import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sympy as sp
import cv2

####################### OPTIONS ############################
# position of the circles
dotbot_1 = np.array([0.9,0.3,0])  
dotbot_2 = np.array([1.25,-0.3,0])

radius = 0.05 # 10cm, diameter
samples = 100 # how many samples to use per circle

# Pose of the LH
lh_t = np.array([0,0,1]) # Origin, z = 1m 
lh_R, _ = cv2.Rodrigues(np.array([-np.pi/16, np.pi/4, np.pi/16 ])) # pointing towards X-axis, elevation angle 45
# lh_R, _ = cv2.Rodrigues(np.array([0, 0., 0 ])) # pointing towards X-axis, elevation angle 45

# Debug grid
grid_p1 = np.array([# Horizontal Line
                    [0, -1.0],
                    [0, -0.75],
                    [0, -0.5],
                    [0, -0.25],
                    [0, 0],
                    [0, 0.25],
                    [0, 0.5],
                    [0, 0.75],
                    [0, 1.0],
                    # Vertical Lines
                    [ 0, -3],
                    [ 0.25, -3],
                    [ 0.5, -3],
                    [ 0.75, -3],
                    [ 1.0, -3],      
                    [ 1.25, -3],
                    [ 1.5, -3],
                    [ 1.75, -3],              
                    [ 2.0, -3],      
                    [ 2.25, -3],
                    [ 2.5, -3],
                    [ 2.75, -3],
                    [ 3.0, -3],      
                    [ 3.25, -3],
                    [ 3.5, -3],
                    [ 3.75, -3],
                    [ 10, -3],
                    # Diagonal lines
                    [ 0, 1],
                    [ 0, 0.5],
                    [ 0, 0],
                    [ 0, -0.5],
                    [ 0, -1],              
                    [ 0.5, -1],              
                    [ 1, -1],              
                    [ 1.5, -1],              
                    [ 2, -1],              
                    ])

grid_p2 = np.array([# Horizontal Line
                    [+6, -1.0],
                    [+6, -0.75],
                    [+6, -0.5],
                    [+6, -0.25],
                    [+6, 0],
                    [+6, 0.25],
                    [+6, 0.5],
                    [+6, 0.75],
                    [+6, 1.0],
                    # Vertical Lines
                    [ 0, +3],
                    [ 0.25, +3],
                    [ 0.5, +3],
                    [ 0.75, +3],
                    [ 1.0, +3],      
                    [ 1.25, +3],
                    [ 1.5, +3],
                    [ 1.75, +3],              
                    [ 2.0, +3],      
                    [ 2.25, +3],
                    [ 2.5, +3],
                    [ 2.75, +3],  
                    [ 3.0, +3],      
                    [ 3.25, +3],
                    [ 3.5, +3],
                    [ 3.75, +3],
                    [ 10, +3],
                    # Diagonal lines
                    [ 3, 4],
                    [ 3, 3.5],
                    [ 3, 3],
                    [ 3, 2.5],
                    [ 3, 2],              
                    [ 3.5, 2],              
                    [ 4, 2],              
                    [ 4.5, 2],              
                    [ 5, 2],        
                    ])

calib_points = {"start": np.array([
                    [0, -1.0],          # Line 1  - parallel 1 - start
                    [0, 1.0],           # Line 2  - parallel 1 - start
                    [ 0, 1],            # Line 3  - parallel 2 - start
                    [ 2, -1], ]),       # Line 4  - parallel 2 - start
                "end": np.array([
                    [+6, -1.0],         # Line 1  - parallel 1 - end
                    [+6, 1.0],          # Line 2  - parallel 1 - end
                    [ 3, 4],            # Line 3  - parallel 2 - end
                    [ 5, 2],  ]),       # Line 4  - parallel 2 - end
                    }

proj_calib_points = {}

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

    # If the point only has a single dimension, add another
    if len(points.shape) == 1:
        points = points.reshape((1,-1))

    # If the points are, 2D, add a third dimension
    if points.shape[1] == 2:
        z = np.zeros((points.shape[0],1))
        points = np.hstack((points, z))

    assert (points.shape[1] == 3), f" points should be (N,3), not {points.shape}"

    # Translate and Rotation points to camera coordinates
    rot_pts = camera_R.T @ (points - camera_t).T
    
    elevation = np.arctan2( rot_pts[2], np.sqrt(rot_pts[0]**2 + rot_pts[1]**2))
    azimuth = np.arctan2(rot_pts[1], rot_pts[0])

    proj_pts = np.array([-np.tan(azimuth),       # horizontal pixel  
                             np.tan(elevation) * 1/np.cos(azimuth)]).T  # vertical   pixel 
    

    # Return projected 2D points
    return proj_pts

# 5. Ellipses fitting
def fit_ellipse(points):

    x = points[:,0]
    y = points[:,1]

    # Construct the design matrix for the equation Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T

    _, _, V = np.linalg.svd(D)  # Singular Value Decomposition for more stability
    params = V[-1, :]           # Solution is in the last row of V

    a,b,c,d,e,f = params

    residuals = a * x**2 + b * x*y + c * y**2 + d * x + e * y + f

    return params  # Returns the coefficients [A, B, C, D, E, F]

# 6. Ellipses intersection, using the algebraic methods
def intersect_ellipses(C1, C2):
    
    # Make the input conics matrix
    A = np.array([[C1[0],   C1[1]/2, C1[3]/2],
                  [C1[1]/2, C1[2],   C1[4]/2],
                  [C1[3]/2, C1[4]/2, C1[5]]])
    
    B = np.array([[C2[0],   C2[1]/2, C2[3]/2],
                  [C2[1]/2, C2[2],   C2[4]/2],
                  [C2[3]/2, C2[4]/2, C2[5]]])

    Cdeg, Cdeg_1, Cdeg_2 = mix_conics_into_degenerate(A, B)

    g,h = split_degenerate_conic(Cdeg)

    p1,q1 = intersect_line_with_conic(g, A)
    p2,q2 = intersect_line_with_conic(h, A)
    p3,q3 = intersect_line_with_conic(g, B)
    p4,q4 = intersect_line_with_conic(h, B)

    return [p1,q1]

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


def intersect_ellipses_numerical(C1, C2):
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

# 7. Calculate the line at infinity from the calibration grid
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
        
######################## PLOTTING FUNCTION ###########################

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
    eigenvalues = np.real_if_close(eigenvalues)
    eigenvectors = np.real_if_close(eigenvectors)
    
    # Rotation angle is the angle of the eigenvector associated with the largest eigenvalue
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Return the center, axes lengths, and angle
    return center, a, b, angle

def plot_conic_matrix_ellipse(conic_params, ax, color, label=""):
    # Extract the elements from the matrix
    if len(conic_params) == 6:
        A, B, C, D, E, F = conic_params
    if conic_params.shape == (3,3):
        A = conic_params[0,0]
        B = conic_params[1,0]
        C = conic_params[1,1]
        D = conic_params[0,2]
        E = conic_params[1,2]
        F = conic_params[2,2]

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

def plot_grid(p1 ,p2 ,ax, color):
    for i in range(p1.shape[0]):
        ax.plot([p1[i,0], p2[i,0]],[p1[i,1], p2[i,1]], '--', color=color)

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
    proj_circle_1 = project_points_pinhole(circle_1, lh_t, lh_R)
    proj_circle_2 = project_points_pinhole(circle_2, lh_t, lh_R)
    proj_dotbot_1 = project_points_pinhole(dotbot_1, lh_t, lh_R)
    proj_dotbot_2 = project_points_pinhole(dotbot_2, lh_t, lh_R)
    proj_grid_p1 = project_points_pinhole(grid_p1, lh_t, lh_R)
    proj_grid_p2 = project_points_pinhole(grid_p2, lh_t, lh_R)

    # Project the calibration points, and comupte the line at infinity
    proj_calib_points["start"] = project_points_pinhole(calib_points["start"], lh_t, lh_R)
    proj_calib_points["end"] = project_points_pinhole(calib_points["end"], lh_t, lh_R)
    linf_orig = compute_line_at_infinity(proj_calib_points)
    
    # Turn the data into (N,2) shape if it is in (2,) shape
    if len(proj_circle_1.shape) == 1: proj_circle_1 = proj_circle_1[np.newaxis,:]
    if len(proj_circle_2.shape) == 1: proj_circle_2 = proj_circle_2[np.newaxis,:]

    # Get the conic equation
    C1 = fit_ellipse(proj_circle_1)
    C2 = fit_ellipse(proj_circle_2)

    # Get the intersection points
    sol = intersect_ellipses_numerical(C1, C2)
    sol_alg = intersect_ellipses(C1, C2)
    print("test")

    # Compute the homography to correct the perspective
    linf, Cinf, H_affinity, H_projective = compute_correcting_homography(sol, [C1, C2])

    ###################### Projective -> Affine rectification ##############

    affine_grid_p1 = apply_point_homography(proj_grid_p1, H_affinity)
    affine_grid_p2 = apply_point_homography(proj_grid_p2, H_affinity)
    affine_circle_1 = apply_point_homography(proj_circle_1, H_affinity)
    affine_circle_2 = apply_point_homography(proj_circle_2, H_affinity)


    ###################### Affine -> Similarity rectification ##############

    similarity_grid_p1 = apply_point_homography(proj_grid_p1, H_projective)
    similarity_grid_p2 = apply_point_homography(proj_grid_p2, H_projective)
    similarity_circle_1 = apply_point_homography(proj_circle_1, H_projective)
    similarity_circle_2 = apply_point_homography(proj_circle_2, H_projective)

########################## PLOT ###########################

    # Plot the resulting 2D projection (should look like an ellipse)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, aspect='equal', adjustable='box')
    # ax.set_aspect('equal')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    # Plot debug grid
    plot_grid(proj_grid_p1, proj_grid_p2 ,ax, "xkcd:gray")
    ax.scatter(proj_circle_1[:,0], proj_circle_1[:,1], c='r', label='dotbot 1')
    ax.scatter(proj_circle_2[:,0], proj_circle_2[:,1], c='g', label='dotbot 2')
    ax.scatter(proj_dotbot_1[:,0], proj_dotbot_1[:,1], c='r')
    ax.scatter(proj_dotbot_2[:,0], proj_dotbot_2[:,1], c='g')
    # plot_conic_matrix_ellipse(C1, ax, "xkcd:blue", label="")
    # plot_conic_matrix_ellipse(C2, ax, "xkcd:blue", label="")
    ax.set_title("LH view: Original Uncorrected")
    ax.set_xlabel("U [px]")
    ax.set_ylabel("V [px]")
    ax.legend()
    # ax.set_xlim([-0.7,0.7])
    # ax.set_ylim([-0.3,0.3])
    ax.set_ylim(ax.get_xlim())
    # ax.grid(True)

    # Plot 3D scene
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_aspect('equal')
    ax2.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    ax2.set_proj_type('ortho')
    # Plot the lighthouse orientation
    arrow = np.array([1,0,0]).reshape((-1,1))
    ax2.quiver(lh_t[0],lh_t[1],lh_t[2], (lh_R @ arrow)[0], (lh_R @ arrow)[1], (lh_R @ arrow)[2], length=0.4, color='xkcd:red', label="Lighthouse")
    ax2.scatter(dotbot_1[0],dotbot_1[1],dotbot_1[2], color='xkcd:red', label='dotbot 1', s=10)
    ax2.scatter(circle_1[:,0],circle_1[:,1],circle_1[:,2], color='xkcd:red', s=10)
    ax2.scatter(dotbot_2[0],dotbot_2[1],dotbot_2[2], color='xkcd:green', label='dotbot 2', s=10)
    ax2.scatter(circle_2[:,0],circle_2[:,1],circle_2[:,2], color='xkcd:green', s=10)
    ax2.set_xlim(0, 2)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(0, 1)
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.set_zlabel("Z [m]")
    ax2.legend()
    ax2.set_title("3D scene: LH observing two DotBot circling around")


    # Plot reconstructed circle
    fig3 = plt.figure(figsize=(6,6))
    ax3 = fig3.add_subplot(111, aspect='equal', adjustable='box')
    # ax.set_aspect('equal')
    ax3.set_aspect(1.0/ax3.get_data_ratio(), adjustable='box')
    plot_grid(grid_p1, grid_p2 ,ax3, "xkcd:gray")
    ax3.scatter(circle_1[:,0],circle_1[:,1], color='xkcd:red', label='dotbot 1', s=50)
    ax3.scatter(circle_2[:,0],circle_2[:,1], color='xkcd:green', label='dotbot 2', s=50)
    ax3.set_title("Original circle")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.legend()
    ax3.set_ylim((-1.5, 1.5))
    ax3.set_xlim((-1.5, 1.5))
    ax3.grid(True)


    # Plot Projective to Affine correction
    fig4 = plt.figure(figsize=(6,6))
    ax4 = fig4.add_subplot(111, aspect='equal', adjustable='box')
    # ax4.set_aspect('equal')
    ax4.set_aspect(1.0/ax4.get_data_ratio(), adjustable='box')
    # Plot debug grid
    plot_grid(affine_grid_p1, affine_grid_p2 ,ax4, "xkcd:gray")
    ax4.scatter(affine_circle_1[:,0], affine_circle_1[:,1], c='r', label='dotbot 1')
    ax4.scatter(affine_circle_2[:,0], affine_circle_2[:,1], c='g', label='dotbot 2')
    # ax4.scatter(proj_dotbot_1[:,0], proj_dotbot_1[:,1], c='r')
    # ax4.scatter(proj_dotbot_2[:,0], proj_dotbot_2[:,1], c='g')
    # plot_conic_matrix_ellipse(C1, ax4, "xkcd:blue", label="")
    # plot_conic_matrix_ellipse(C2, ax4, "xkcd:blue", label="")
    ax4.set_title("Corrected View #1: Affine-only Distortion")
    ax4.set_xlabel("U [px]")
    ax4.set_ylabel("V [px]")
    ax4.legend()
    # ax4.set_xlim([-0.7,0.7])
    # ax4.set_ylim([-0.3,0.3])
    ax4.set_ylim(ax4.get_xlim())
    # ax4.grid(True)



    # Plot Affine to Similarity correction
    fig5 = plt.figure(figsize=(6,6))
    ax5 = fig5.add_subplot(111, aspect='equal', adjustable='box')
    # ax5.set_aspect('equal')
    ax5.set_aspect(1.0/ax5.get_data_ratio(), adjustable='box')
    # Plot debug grid
    plot_grid(similarity_grid_p1, similarity_grid_p2 ,ax5, "xkcd:gray")
    ax5.scatter(similarity_circle_1[:,0], similarity_circle_1[:,1], c='r', label='dotbot 1')
    ax5.scatter(similarity_circle_2[:,0], similarity_circle_2[:,1], c='g', label='dotbot 2')
    # ax5.scatter(proj_dotbot_1[:,0], proj_dotbot_1[:,1], c='r')
    # ax5.scatter(proj_dotbot_2[:,0], proj_dotbot_2[:,1], c='g')
    # plot_conic_matrix_ellipse(C1, ax5, "xkcd:blue", label="")
    # plot_conic_matrix_ellipse(C2, ax5, "xkcd:blue", label="")
    ax5.set_title("Corrected View #2: Similarity-only Distortion")
    ax5.set_xlabel("U [px]")
    ax5.set_ylabel("V [px]")
    ax5.legend()
    # ax5.set_xlim([-0.7,0.7])
    # ax5.set_ylim([-0.3,0.3])
    ax5.set_ylim(ax5.get_xlim())
    # ax5.grid(True)

    plt.show()

# Run the main function
main()

