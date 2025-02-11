# test every circle pair combination for computing the dual conic
# to find the best homography.
#


# import the necessary packages
import json
import numpy as np
import pandas as pd
import itertools

####################################################################################
###                               Options                                        ###
####################################################################################
# Define which of the 6 experimetns you want to plot


def fit_ellipse(points):

    x = points[:,0]
    y = points[:,1]

    # Construct the design matrix for the equation Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T

    _, _, V = np.linalg.svd(D)  # Singular Value Decomposition for more stability
    params = V[-1, :]           # Solution is in the last row of V

    a,b,c,d,e,f = params

    residual = a * x**2 + b * x*y + c * y**2 + d * x + e * y + f

    # Normalize the parameter to F=1
    params = params / params[5]

    return params, residual  # Returns the coefficients [A, B, C, D, E, F]



def conic_eccentricity(circle):
    """
    Computes the eccentricity of a conic section given the coefficients of its general equation:
    Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0.

    Parameters:
        A, B, C, D, E, F (float): Coefficients of the conic equation.

    Returns:
        float: Eccentricity of the conic section.
    """
    # Calculate the discriminant of the conic
    A, B, C, D, E, F = circle

    conic = np.array([[A, B/2, D/2],
                      [B/2, C, E/2],
                      [D/2, E/2, F]])
    
    n = -1*np.sign(np.linalg.det(conic))

    discriminant = np.sqrt((A-C)**2 + B**2)

    numerator = np.sqrt(2 * discriminant )
    denominator = np.sqrt(n*(A+C) + discriminant)

    eccentricity = numerator / denominator
    
    return eccentricity


####################################################################################
###                            Read Dataset                                      ###
####################################################################################


# Create the dataframe that will store the results
concrete_mocap_file = f'concrete_floor_mocap.csv'
plastic_mocap_file = f'plastic_sheet_mocap.csv'

# Read files
concrete_data = pd.read_csv(concrete_mocap_file, parse_dates=['timestamp'])
plastic_data  = pd.read_csv(plastic_mocap_file, parse_dates=['timestamp'])

# Decimate data
# n1 = len(concrete_data)
# n2 = len(plastic_data)
# concrete_data = concrete_data.iloc[:n1 // 2]
# plastic_data = plastic_data.iloc[:n2 // 2]

# Get point data
concrete_rubber_points = concrete_data[['rubber_x','rubber_y']].values
concrete_40d_points = concrete_data[['40d_x','40d_y']].values
plastic_rubber_points = plastic_data[['rubber_x','rubber_y']].values
plastic_40d_points = plastic_data[['40d_x','40d_y']].values

# Try to fit the data to an ellipse conic
concrete_rubber_circle, residual_1 = fit_ellipse(concrete_rubber_points)  
concrete_40d_circle, residual_2 = fit_ellipse(concrete_40d_points)  
plastic_rubber_circle, residual_3 = fit_ellipse(plastic_rubber_points)  
plastic_40d_circle, residual_4 = fit_ellipse(plastic_40d_points)  

# Compute eccentricity
concrete_rubber_ecc = conic_eccentricity(concrete_rubber_circle)
concrete_40d_ecc = conic_eccentricity(concrete_40d_circle)
plastic_rubber_ecc = conic_eccentricity(plastic_rubber_circle)
plastic_40d_ecc = conic_eccentricity(plastic_40d_circle)


print("residuals: ")
print(f"cr: {abs(residual_1).mean()}")
print(f"c4: {abs(residual_2).mean()}")
print(f"pr: {abs(residual_3).mean()}")
print(f"p4: {abs(residual_4).mean()}")

print("\neccentricity: ")
print(f"cr: {concrete_rubber_ecc}")
print(f"c4: {concrete_40d_ecc}")
print(f"pr: {plastic_rubber_ecc}")
print(f"p4: {plastic_40d_ecc}")



####################################################################################
###                            Read Dataset                                      ###
####################################################################################


