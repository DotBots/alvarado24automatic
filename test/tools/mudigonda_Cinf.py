import sympy as sp
from sympy import pprint, latex

# Define a general 3x3 matrix with symbolic entries in sympy
a, b, v1, v2 = sp.symbols('a b v1 v2')

# Define the 3x3 matrix
K  = sp.Matrix([[a,  b],
                [0, 1/a]])

# Define the 3x3 matrix
v = sp.Matrix([[v1], [v2]])

kkt = K @ K.T
ktv = K.T @ v
vtk = v.T @ K

print(f"K @ K.T: {kkt}, shape = {kkt.shape}")
print(f"K.T @ v: {ktv}, shape = {ktv.shape}")
print(f"v.T @ K: {vtk}, shape = {vtk.shape}")

