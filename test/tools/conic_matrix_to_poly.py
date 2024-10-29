import numpy as np
from sympy import symbols, Matrix

def conic_matrix_to_polynomial(conic_matrix):
    # Define symbolic variables for x and y
    x, y = symbols('x y')
    
    # Convert the numpy matrix to a sympy Matrix
    M = Matrix(conic_matrix)
    
    # Define the coordinate vector (x, y, 1)
    coord_vector = Matrix([x, y, 1])
    
    # Perform the matrix multiplication and expand the result
    polynomial_form = (coord_vector.T * M * coord_vector)[0].expand()
    
    return polynomial_form

# Example usage with a 3x3 NumPy matrix for a conic section
# Example conic matrix for: 2x^2 + 3y^2 + 4xy + 5x + 6y + 7 = 0
# conic_matrix = np.array([[-1.12217296e+00,  0.00000000e+00,  2.24434591e+00],
#        [ 0.00000000e+00,  6.11006579e-10,  2.37624685e-05],
#        [ 2.24434591e+00,  2.37624685e-05, -3.56454447e+00]])
conic_matrix = np.array([  [ 1,   1,  -1],
                           [-1,  -1,   1],
                           [ 1,   1,  -1]])

# Get the polynomial form
polynomial_form = conic_matrix_to_polynomial(conic_matrix)
print("Conic section in polynomial form:")
print(polynomial_form)
