import numpy as np
from scipy.optimize import least_squares

# Define the known positions of the receivers and the reference station
x0, y0, z0 = 4039139.89, 897222.76, 4838608.59  # Reference station
x1, y1, z1 = 4027031.42, 890211.32, 4849775.99
x2, y2, z2 = 4047433.94, 904276.02, 4830281.62

# True position for verification
x_true, y_true, z_true = 3999117.68, 963774.50, 4864491.90

# Define the known TDOA measurements with respect to the reference station
delta_t10 = -6.044e-6
delta_t20 = 6.916e-6

# Known positions of the receivers and the reference station
receivers = np.array([
    [x0, y0, z0],  # Reference station
    [x1, y1, z1],
    [x2, y2, z2]
])

# Known TDOA measurements with respect to the reference station
tdoa = np.array([delta_t10, delta_t20])

# Speed of the signal (e.g., speed of light)
v = 3e8  # speed of light in m/s

# Known altitude of the node
h = z_true

def equations(vars, receivers, tdoa, v, h):
    x, y = vars
    diffs = []
    for i in range(1, len(receivers)):  # Start from 1 to skip the reference station
        d_i = np.sqrt((x - receivers[i, 0])**2 + (y - receivers[i, 1])**2 + (h - receivers[i, 2])**2)
        d_0 = np.sqrt((x - receivers[0, 0])**2 + (y - receivers[0, 1])**2 + (h - receivers[0, 2])**2)
        diffs.append(v * tdoa[i-1] - (d_i - d_0))
    return diffs

# Use a reasonable initial guess for the position (x, y)
initial_guess = [(x0 + x1 + x2) / 3, (y0 + y1 + y2) / 3]

# Solve the system of equations
result = least_squares(equations, initial_guess, args=(receivers, tdoa, v, h))

# Extract the solution
x, y = result.x
print(f"Estimated position: x={x}, y={y}, h={h}")
print(f"True position: x={x_true}, y={y_true}, h={z_true}")

