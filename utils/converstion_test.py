import numpy as np
from scipy.optimize import least_squares

# WGS-84 ellipsoid parameters
a = 6378137.0  # Semi-major axis
f = 1 / 298.257223563  # Flattening
e2 = 2 * f - f ** 2  # Square of eccentricity


# Function to convert degrees, minutes, seconds to decimal degrees
def dms_to_dd(d, m, s):
    return d + m / 60 + s / 3600


# Function to convert spherical coordinates to Cartesian coordinates
def spherical_to_cartesian(lat_dms, lon_dms, h):
    lat_dd = dms_to_dd(*lat_dms)
    lon_dd = dms_to_dd(*lon_dms)

    lat_rad = np.radians(lat_dd)
    lon_rad = np.radians(lon_dd)

    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    x = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    z = ((1 - e2) * N + h) * np.sin(lat_rad)

    return x, y, z


# Provided spherical coordinates for reference stations
P0 = [(49, 39, 20), (12, 31, 26), 700]
P1 = [(49, 48, 43), (12, 27, 55), 600]
P2 = [(49, 32, 28), (12, 35, 39), 600]

print(f"P0 Spherical: {P0}\nP0 Cartesian: {spherical_to_cartesian(P0[0], P0[1], P0[2])}\n")
P0[2] = 0
print(f"P0 Spherical: {P0}\nP0 Cartesian: {spherical_to_cartesian(P0[0], P0[1], P0[2])}\n")
P0[2] = 700

# Convert all reference points to Cartesian coordinates
P0_cartesian = spherical_to_cartesian(P0[0], P0[1], P0[2])
P1_cartesian = spherical_to_cartesian(P1[0], P1[1], P1[2])
P2_cartesian = spherical_to_cartesian(P2[0], P2[1], P2[2])

print(f"P0 Cartesian: {P0_cartesian}")
print(f"P1 Cartesian: {P1_cartesian}")
print(f"P2 Cartesian: {P2_cartesian}")

# Cartesian coordinates from the previous conversion
x0, y0, z0 = P0_cartesian
x1, y1, z1 = P1_cartesian
x2, y2, z2 = P2_cartesian

# Define the known TDOA measurements with respect to the reference station
delta_t10 = -6.044e-6
delta_t20 = 6.916e-6

# Speed of the signal (e.g., speed of light)
v = 3e8  # speed of light in m/s

# Known altitude of the node
h = 4864491.90  # given height in meters

# Receivers' Cartesian coordinates
receivers = np.array([
    [x0, y0, z0],  # Reference station
    [x1, y1, z1],
    [x2, y2, z2]
])

# Known TDOA measurements with respect to the reference station
tdoa = np.array([delta_t10, delta_t20])





# Bounds for optimization variables (x, y)
bounds = ([-np.inf, -np.inf], [np.inf, np.inf])  # Adjust as needed

def equations(vars, receivers, tdoa, v, h):
    x, y = vars
    diffs = []
    for i in range(1, len(receivers)):  # Start from 1 to skip the reference station
        d_i = np.sqrt((x - receivers[i, 0])**2 + (y - receivers[i, 1])**2 + (h - receivers[i, 2])**2)
        d_0 = np.sqrt((x - receivers[0, 0])**2 + (y - receivers[0, 1])**2 + (h - receivers[0, 2])**2)
        diffs.append(v * tdoa[i-1] - (d_i - d_0))
    return diffs

# Use an improved initial guess based on TDOA measurements and known height
# Here, you need to implement a method to estimate initial guess based on TDOA measurements and known height
initial_guess = [(x0+x1+x2)/3, (y0+y1+y2)/3]  # Adjust as needed

# Solve the system of equations using Levenberg-Marquardt algorithm
result = least_squares(equations, initial_guess, bounds=bounds, method='lm', args=(receivers, tdoa, v, h))

# Extract the solution
x, y = result.x
print(f"Estimated position: x={x}, y={y}, h={h}")