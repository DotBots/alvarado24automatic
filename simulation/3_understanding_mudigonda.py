import sympy as sp
from sympy import pprint, latex

# Define symbolic variables for X_0 and Y_0
x, y = sp.symbols('x y')
# Define a general 3x3 matrix with symbolic entries in sympy
a, b, c, f, g, h = sp.symbols('a b c f g h')
# Define a general 3x3 matrix with symbolic entries in sympy
h1, h2, h3, h4, h5, h6, h7, h8, h9 = sp.symbols('h1 h2 h3 h4 h5 h6 h7 h8 h9')

# Define the 3x3 matrix
CC = sp.Matrix([[a, h, g],
                [h, b, f],
                [g, f, c]])

# Define the 3x3 matrix
X  = sp.Matrix([[x, y, 0]])
Xt = sp.Matrix([[x], 
                [y], 
                [0]])


M = X @ CC @ Xt

print(latex(M))



### solve circle intersection

import sympy as sp

# Define the variables
x, y = sp.symbols('x y')

# Define the two conic equations
C1 = 2*x**2 + 3*y**2 + 4*x*y + 5*x + 6*y + 7
C2 = x**2 + y**2 + x*y + 4*x + 3*y + 9

# Solve the system of equations
solution = sp.solve([C1, C2], (x, y))

# Print the solutions
pprint(solution)


### Numerical solution
import numpy as np
from scipy.optimize import fsolve

# Define the two conic equations as functions

def conic1(p):
    x, y = p
    return 2*x**2 + 3*y**2 + 4*x*y + 5*x + 6*y + 7  # Example conic 1

def conic2(p):
    x, y = p
    return x**2 + y**2 + x*y + 4*x + 3*y + 9         # Example conic 2

# Function that returns the system of two conics
def system(p):
    return [conic1(p), conic2(p)]

# Initial guess (this is important, as fsolve needs a good starting point)
initial_guess = [0, 0]

# Solve the system numerically
solution = fsolve(system, initial_guess)

# Print the solution (intersection point)
print("Intersection point:", solution)
