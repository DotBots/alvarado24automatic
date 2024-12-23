import numpy as np
import matplotlib.pyplot as plt

def plot_conic(ax, A, B, C, D, E, F, color='blue', label=None):
    """
    Plots a conic section defined by Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 on a given axis.
    
    Parameters:
        ax (matplotlib.axes.Axes): The subplot axis to draw the conic section.
        A, B, C, D, E, F (float): Coefficients of the conic equation.
        color (str): Color of the conic plot. Default is 'blue'.
        label (str): Label for the conic plot. Default is None.
    """
    # Define the function for the conic equation
    def conic_eq(x, y):
        return A * x**2 + B * x * y + C * y**2 + D * x + E * y + F

    # Set up a grid for plotting
    x = np.linspace(-10, 10, 500)  # Adjust the range if needed
    y = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x, y)

    # Evaluate the conic equation on the grid
    Z = conic_eq(X, Y)

    # Plot the contour where the conic equation equals zero
    ax.contour(X, Y, Z, levels=[0], colors=color, label=label)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True)

# Example usage
fig, ax = plt.subplots(figsize=(8, 8))

# Plot several ellipses
plot_conic(ax, 1, 0, 2, 0, 0, -4, color='blue', label='Ellipse 1')
plot_conic(ax, 2, 0, 1, 0, 0, -4, color='red', label='Ellipse 2')
plot_conic(ax, 1, 1, 1, 0, 0, -6, color='green', label='Ellipse 3')

ax.legend()
plt.title('Multiple Conics')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
