import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

####################### OPTIONS ############################
# Degenerate conic, X centered at  (1,0)
A = np.array([  [ 1,   1,  -1],
                [-1,  -1,   1],
                [ 1,   1,  -1]])

# Should result in these two lines: [1,1,-1], [1,-1,1]


# Degenerate conic, Dual line collapsed. Vertical line at x=2
# A = np.array([  [ 1,   0,  -2],
#                 [ 0,   0,   0],
#                 [ -2,   0,   4]])

# Should result in one line: [1,0,-2]



######################## FUNCTION ###########################

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




########################## PLOT #############################

# prepare the plot
fig = plt.figure(layout="constrained", figsize=(5,4))
gs = GridSpec(3, 3, figure = fig)
ax = fig.add_subplot(gs[0:3, 0:3])
ax.set_aspect('equal', 'box')

# Create middle degenerate conic
g,h = split_degenerate_conic(A)

print(f"lines:\np={g}\np={h}")

# plot homogeneous lines
plot_homogeneous_line(g, ax, x_range=(-10, 10))
plot_homogeneous_line(h, ax, x_range=(-10, 10))

ax.set_title('Ellipse from Conic Matrix')
ax.grid(True)
# ax.set_xlim([-1,None])
# ax.set_ylim([-1,None])
    
plt.show()