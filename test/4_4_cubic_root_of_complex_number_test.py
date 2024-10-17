import math
import numpy as np

def cuberoot( z ):
    z = complex(z)
    x = z.real
    y = z.imag
    mag = abs(z)
    arg = math.atan2(y,x)
    return [ mag**(1./3) * np.exp( 1j*(arg+2*n*math.pi)/3 ) for n in range(1,4) ]

print(f"cbrt of 1 = {cuberoot(1+0j)}")