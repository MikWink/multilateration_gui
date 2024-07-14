from scipy.optimize import root
import numpy as np

class PressureSolver:
    def __init__(self, BS0, BS1, BS2, h, R10, R20):
        self.x0, self.y0, self.z0 = BS0
        self.x1, self.y1, self.z1 = BS1
        self.x2, self.y2, self.z2 = BS2
        self.h = h
        self.R10 = R10 * 3e8
        self.R20 = R20 * 3e8
        self.h = 5000
        self.a = 6378137
        self.b = 6356752.31425
        # Known parameters and TDOA measurements
        self.X = np.array([self.x0, self.x1, self.x2])  # X coordinates of base stations
        self.Y = np.array([self.y0, self.y1, self.y2])  # Y coordinates of base stations
        self.Z = np.array([self.z0, self.z1, self.z2])  # Z coordinates of base stations
        self.equation = self.equations
        # Initial guess for the unknowns (x, y, z)
        self.initial_guess = [(self.x0 + self.x1 + self.x2) / 3, (self.y0 + self.y1 + self.y2) / 3, (self.z0 + self.z1 + self.z2) / 3]

    # Function representing the system of equations
    def equations(self, vars):
        x, y, z = vars
        eq1 = np.sqrt((x - self.X[1]) ** 2 + (y - self.Y[1]) ** 2 + (z - self.Z[1]) ** 2) - np.sqrt(
            (x - self.X[0]) ** 2 + (y - self.Y[0]) ** 2 + (z - self.Z[0]) ** 2) - self.R10
        eq2 = np.sqrt((x - self.X[2]) ** 2 + (y - self.Y[2]) ** 2 + (z - self.Z[2]) ** 2) - np.sqrt(
            (x - self.X[0]) ** 2 + (y - self.Y[0]) ** 2 + (z - self.Z[0]) ** 2) - self.R20
        eq3 = (x / (self.a + self.h)) ** 2 + (y / (self.a + self.h)) ** 2 + (z / (self.b + self.h)) ** 2 - 1
        return [eq1, eq2, eq3]

    def solve(self):
        result = root(self.equation, self.initial_guess)
        x, y, z =  result.x
        print(f"Estimated position: x={x}, y={y}, z={z}")





