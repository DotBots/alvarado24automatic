import sympy as sp
from sympy import pprint, latex

# Define symbolic variables for X_0 and Y_0
X_0, Y_0 = sp.symbols('X_0 Y_0')
# Define a general 3x3 matrix with symbolic entries in sympy
a, b, v1, v2 = sp.symbols('a b v1 v2')
# Define a general 3x3 matrix with symbolic entries in sympy
h1, h2, h3, h4, h5, h6, h7, h8, h9 = sp.symbols('h1 h2 h3 h4 h5 h6 h7 h8 h9')

# Define the 3x3 matrix
Ha = sp.Matrix([[a, b, 0],
               [0, 1/a, 0],
               [0, 0, 1]])

Hp = sp.Matrix([[1, 0, 0],
               [0, 1, 0],
               [v1, v2, 1]])

# Define the 3x3 matrix
H = Ha @ Hp

# H = sp.Matrix([[h1, h2, 0],
#                [0, h5, 0],
#                [h7, h8, 1]])

C = Hp = sp.Matrix([[25, 0, -75],
                    [0, 36, 36],
                    [-75, 36, -639]])

ii = sp.Matrix([[1],
               [sp.I],
               [0]])

Eq = ii.T @ H.inv().T @ C @ H.inv() @ ii

pprint(H.inv())
pprint(Eq)

