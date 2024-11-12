import sympy as sp
from sympy import pprint, latex

# Define symbolic variables for X_0 and Y_0
X_0, Y_0 = sp.symbols('X_0 Y_0')
# Define a general 3x3 matrix with symbolic entries in sympy
a, b, c, d, e, f, g, h, i = sp.symbols('a b c d e f g h i')
# Define a general 3x3 matrix with symbolic entries in sympy
h1, h2, h3, h4, h5, h6, h7, h8, h9 = sp.symbols('h1 h2 h3 h4 h5 h6 h7 h8 h9')

# Define the 3x3 matrix
CC = sp.Matrix([[a, b, c],
               [d, e, f],
               [g, h, i]])

# Define the 3x3 matrix
H = sp.Matrix([[h1, h2, h3],
               [h4, h5, h6],
               [h7, h8, h9]])

# Initialize the conic matrix 1
C1 = sp.Matrix([[1, 0, -X_0],
               [0, 1, -Y_0],
               [-X_0, -Y_0, X_0**2 + Y_0**2 - 25]])

C2 = sp.Matrix([[1,  0,   0],
                [0,  1,   0],
                [0,  0, -25]])


M = CC @ H - H @ C2.inv() @ C1

print(latex(M))
