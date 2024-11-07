import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

####################### OPTIONS ############################
# Degenerate conic, X centered at  (2,2)
# y = x
# y = 4 - x
# A = np.array([  [ 1,   0,  -2],
#                 [ 0,  -1,   2],
#                 [-2,   2,   0]])

# Degenerate conic, X centered at  (2,3)
# y = x + 1
# y = 5 - x
# A = np.array([  [ 1,   0,  -2],
#                 [ 0,  -1,   3],
#                 [-2,   3,   -5]])





# Degenerate conic, Dual line collapsed. Vertical line at x=2
A = np.array([  [ 1,   0,  -2],
                [ 0,   0,   0],
                [ -2,   0,   4]])

# Should result in one line: [1,0,-2]



######################## FUNCTION ###########################

def split_degenerate_conic(A):

    """
    Split the degenerate conic A into two homogeneous cordinates lines g and h
    """


    # Normalize the matrix
    a = A[0, 0]
    b = 2 * A[0, 1]  # Note: we double since the matrix stores B/2
    c = A[1, 1]
    d = 2 * A[0, 2]
    e = 2 * A[1, 2]
    f = A[2, 2]

    # scale = np.linalg.norm(A)
    # A_scaled = A / scale

    # a = A_scaled[0, 0]
    # b = 2 * A_scaled[0, 1]  # Note: we double since the ma_scaledtrix stores B/2
    # c = A_scaled[1, 1]
    # d = 2 * A_scaled[0, 2]
    # e = 2 * A_scaled[1, 2]
    # f = A_scaled[2, 2]


    # Compute the adjoint matrix
    # adjoint = np.array([      [b*f-e*e/4, (d*e)/4-(c*f)/2, (c*e)/4-(b*d)/2],
    #                 [(d*e)/4-(c*f)/2, a*f-d*d/4      , (c*d)/4-(a*e)/2],
    #                 [(c*e)/4-(b*d)/2,(c*d)/4-(a*e)/2,a*b-c*c/4       ]])
    adjoint = np.array([[  c*f-e*e, -(b*f-d*e),  b*e-d*c ],
                  [-(b*f-d*e),  a*f-d*d, -(a*e-b*d) ],
                  [  b*e-d*c, -(a*e-b*d), (a*c-b*b) ]])
    
    # find the smallest, non-zero diagonal element
    diag = np.emath.sqrt(-np.diag(adjoint))
    i = np.argmax(diag)

    i = 2

    # Compute intersection point between lines:
    p = np.array([ adjoint[i][0] / diag[i] ,
                   adjoint[i][1] / diag[i] ,
                   adjoint[i][2] / diag[i] ,
                  ])
    p = np.real_if_close(p) # convert to real if possible

    # compute N matrix
    N = np.array([  [a,            c/2 - p[2],  d/2 + p[1]],
                    [c/2 + p[2],       b,       e/2 - p[0]],
                    [d/2 - p[1],   e/2 + p[0],      f  ]])


    # Get the highest valued row and column
    col_max = 99
    col_val_max = 0
    row_max = 99
    row_val_max = 0
    for j in range(N.shape[0]):
        # check if column is higher
        if (N[1,j]**2 + N[2,j]**2) > col_val_max:
            col_max = j
            col_val_max = (N[1,j]**2 + N[2,j]**2)

        # check if row is higher
        if (N[j,1]**2 + N[j,2]**2) > row_val_max:
            row_max = j
            row_val_max = (N[j,1]**2 + N[j,2]**2)

    u1, v1, w1 = N[:,col_max]
    u2, v2, w2 = N[row_max,:]

    # Compute the line parameters
    alpha_1 = np.arctan2(v1, u1)
    alpha_2 = np.arctan2(v2, u2)

    g = np.array([np.cos(alpha_1), np.sin(alpha_1), w1 / np.emath.sqrt(u1*u1 + v1*v1)])
    h = np.array([np.cos(alpha_2), np.sin(alpha_2), w2 / np.emath.sqrt(u2*u2 + v2*v2)])

    # return the two intersecting lines of the degenerate conic
    return g,h

def split_degenerate_conic_book(A):
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

        # Compute the adjoint matrix
        B = np.array([[  c*f-e*e, -(b*f-d*e),  b*e-d*c ],
                    [-(b*f-d*e),  a*f-d*d, -(a*e-b*d) ],
                    [  b*e-d*c, -(a*e-b*d), (a*c-b*b) ]])
        
        # Find the smallest (non-zero) diagonal element of the adjoint 
        for i in range(3):
            if np.isclose(np.argsort(np.diag(B))[i],0): 
                idx = i
                break

        # Get the intersection point of the lines
        beta = np.emath.sqrt(B[idx,idx])
        p = B[:,idx]/beta
        # Normalize the point p
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




########################## PLOT #############################

# prepare the plot
fig = plt.figure(layout="constrained", figsize=(5,4))
gs = GridSpec(3, 3, figure = fig)
ax = fig.add_subplot(gs[0:3, 0:3])
ax.set_aspect('equal', 'box')

# Create middle degenerate conic
# g1,h1 = split_degenerate_conic(A)
g, h = split_degenerate_conic_book(A)
print(f"lines:\np={g}\np={h}")

# plot homogeneous lines
plot_homogeneous_line(g, ax, x_range=(-10, 10))
plot_homogeneous_line(h, ax, x_range=(-10, 10))

ax.set_title('Ellipse from Conic Matrix')
ax.grid(True)
# ax.set_xlim([-1,None])
# ax.set_ylim([-1,None])
    
plt.show()