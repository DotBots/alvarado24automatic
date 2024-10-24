import numpy as np
import math
from pprint import pprint

def cuberoot( z ):
    z = complex(z)
    x = z.real
    y = z.imag
    mag = abs(z)
    arg = math.atan2(y,x)
    return [ mag**(1./3) * np.exp( 1j*(arg+2*n*math.pi)/3 ) for n in range(1,4) ]


### Define a few equtions

# x^3 + 7 x^2 + 12 x - 20
# (x-1)(x + 4 - i2)(x + 4 + i2)
a = 1
b = 7
c = 12
d = -20

# x^3 + 2 x^2 - 40 x + 64
# (x - 2)(x - 4)(x + 8)
# a = 1
# b = 2
# c = -40
# d = 64

# x^3 - 3x^2 + 9x + 13
# (x + 1) (x - 2 + 3i) * (x - 2 - 3i)
# a = 1
# b = -3
# c = 9
# d = 13

# Mix Conics A and B into degenerate Conic C

# Solve the cubic equation
delta0 = b**2 - 3*a*c
delta1 = 2*b**3 - 9*a*b*c + 27*d*a**2
for i in range(3):
    omega_min = cuberoot((delta1 - np.emath.sqrt(delta1**2 - 4*delta0**3))/2)[i]
    omega_plus = cuberoot((delta1 + np.emath.sqrt(delta1**2 - 4*delta0**3))/2)[i]

    # Get the solution points
    k = 0
    sol = {}
    for k in range(3):
        sol[k] = - (b + np.e**(1j*2*np.pi*k/3)*omega_plus + np.e**(1j*-2*np.pi*k/3)*omega_min ) / 3*a
        sol[k] = np.real_if_close(sol[k])

    if np.isreal(sol[0]) or np.isreal(sol[1]) or np.isreal(sol[2]): 
        break

if np.isreal(sol[0]):
    raise ValueError("no real roots")

pprint(sol)