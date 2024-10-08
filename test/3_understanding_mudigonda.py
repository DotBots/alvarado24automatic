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

