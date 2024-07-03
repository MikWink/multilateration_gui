import logging

import numpy as np
import math
from transformer import CoordinateTransformer

class Foy:
    def __init__(self, bs_list, ms, tdoa_std=0, baro_std=0):
        self.bs_list = bs_list
        self.ms = ms
        self.tdoa_std = tdoa_std
        self.baro_std = baro_std
        self.h = np.zeros(3)
        self.G = np.zeros((3, 3))
        self.deltaXY = np.zeros(3)
        self.R_i_0 = [0, 0, 0, 0]
        self.R_i_real = [0, 0, 0, 0]
        self.R_i_guess = [0, 0, 0, 0]
        self.guesses = [[], [], []]
        self.guessed_position = self.make_init_guess(bs_list)
        self.guesses[0].append(self.guessed_position[0])
        self.guesses[1].append(self.guessed_position[1])
        self.guesses[2].append(self.guessed_position[2])

    def update_x_y(self):
        #print(f"Old position: {self.guessed_position}")
        self.guessed_position[0] += self.deltaXY[0]
        self.guessed_position[1] += self.deltaXY[1]
        self.guessed_position[2] += self.deltaXY[2]
        #print(f"New position: {self.guessed_position}")
        self.guesses[0].append(self.guessed_position[0])
        self.guesses[1].append(self.guessed_position[1])
        self.guesses[2].append(self.guessed_position[2])
        #print(f"Real position: {self.ms}\n")
        #print(f"Difference: {self.guessed_position[0] - self.ms[0]}, {self.guessed_position[1] - self.ms[1]}, {self.guessed_position[2] - self.ms[2]}\n")

    def calculate_deltaXY(self):
        GT_G = np.matmul(np.transpose(self.G), self.G)
        GT_G_inv = np.linalg.inv(GT_G)
        GT_G_inv_GT = np.matmul(GT_G_inv, np.transpose(self.G))
        self.deltaXY = np.matmul(GT_G_inv_GT, self.h)
        #print(f"deltaXY: {self.deltaXY}\n")
        return self.deltaXY

    def calculate_G(self):
        self.G[0, 0] = ((self.bs_list[0][0] - self.guessed_position[0]) / self.R_i_guess[0]) - (
                    (self.bs_list[1][0] - self.guessed_position[0]) / self.R_i_guess[1])
        self.G[0, 1] = ((self.bs_list[0][1] - self.guessed_position[1]) / self.R_i_guess[0]) - (
                    (self.bs_list[1][1] - self.guessed_position[1]) / self.R_i_guess[1])
        self.G[0, 2] = ((self.bs_list[0][2] - self.guessed_position[2]) / self.R_i_guess[0]) - (
                    (self.bs_list[1][2] - self.guessed_position[2]) / self.R_i_guess[1])
        self.G[1, 0] = ((self.bs_list[0][0] - self.guessed_position[0]) / self.R_i_guess[0]) - (
                    (self.bs_list[2][0] - self.guessed_position[0]) / self.R_i_guess[2])
        self.G[1, 1] = ((self.bs_list[0][1] - self.guessed_position[1]) / self.R_i_guess[0]) - (
                    (self.bs_list[2][1] - self.guessed_position[1]) / self.R_i_guess[2])
        self.G[1, 2] = ((self.bs_list[0][2] - self.guessed_position[2]) / self.R_i_guess[0]) - (
                    (self.bs_list[2][2] - self.guessed_position[2]) / self.R_i_guess[2])
        self.G[2, 0] = ((self.bs_list[0][0] - self.guessed_position[0]) / self.R_i_guess[0]) - (
                    (self.bs_list[3][0] - self.guessed_position[0]) / self.R_i_guess[3])
        self.G[2, 1] = ((self.bs_list[0][1] - self.guessed_position[1]) / self.R_i_guess[0]) - (
                    (self.bs_list[3][1] - self.guessed_position[1]) / self.R_i_guess[3])
        self.G[2, 2] = ((self.bs_list[0][2] - self.guessed_position[2]) / self.R_i_guess[0]) - (
                    (self.bs_list[3][2] - self.guessed_position[2]) / self.R_i_guess[3])

        #print(f"G: {self.G}\n")

    def calculate_h(self):
        # Calculate the h vector
        for i in range(1, 4):
            #print(f"R_i_0[i]: {self.R_i_0[i]}, R_iguess[i]: {self.R_i_guess[i]}, R_iguess[i-1]: {self.R_i_guess[i - 1]}")
            self.h[i - 1] = (self.R_i_0[i] - (self.R_i_guess[i] - self.R_i_guess[0]))

        #print(f"h: {self.h}\n")

    def calculate_R_i_guess(self):
        #print(f"Bs list: {self.bs_list}")
        for i in range(4):
            try:
                #print(f"i:{i}, bs_list[i]: {self.bs_list[i]}, guessed_position: {self.guessed_position}")
                self.R_i_guess[i] = math.sqrt(
                    (self.bs_list[i][0] - self.guessed_position[0]) ** 2 + (self.bs_list[i][1] - self.guessed_position[1]) ** 2 + (
                                self.bs_list[i][2] - self.guessed_position[2]) ** 2)
            except Exception as e:
                logging.error(f"Error: {e}")
        #print(f"R_i_guess: {self.R_i_guess}\n")

    def make_init_guess(self, bs_list):
        # Compute the center coordinates
        x_val = sum(bs_list[0]) / len(bs_list[0])
        y_val = sum(bs_list[1]) / len(bs_list[1])
        z_val = sum(bs_list[2]) / len(bs_list[2])
        #print(x_val, y_val, z_val)
        center = np.array([x_val, y_val, z_val])

        # Print the result
        #print(f"Initial guess: {center}\n")
        return center

    def calculate_tdoa_s(self):
        #print(f'TDOA_STD: {self.tdoa_std}')
        for i in range(4):
            self.R_i_real[i] = math.sqrt(
                (self.bs_list[i][0] - self.ms[0]) ** 2 + (self.bs_list[i][1] - self.ms[1]) ** 2 + (
                        self.bs_list[i][2] - self.ms[2]) ** 2)
            self.R_i_0[i] = (math.sqrt(
                (self.bs_list[i][0] - self.ms[0]) ** 2 + (self.bs_list[i][1] - self.ms[1]) ** 2 + (
                        self.bs_list[i][2] - self.ms[2]) ** 2) - math.sqrt(
                (self.bs_list[0][0] - self.ms[0]) ** 2 + (self.bs_list[0][1] - self.ms[1]) ** 2 + (
                        self.bs_list[0][2] - self.ms[2]) ** 2))

        #print(f"R_i_0: {self.R_i_0}\n")
        for i, t in enumerate(self.R_i_0):
            self.R_i_0[i] += self.tdoa_std

        #print(f"R_i: {self.R_i_real}")
        #print(f"R_i_0: {self.R_i_0}\n")

    def convert_coordinates(self):
        for bs in self.bs_list:
            utm_list = list(CoordinateTransformer.long_to_utm(bs))
            utm_list.append(bs[2])
            self.bs_list.append(utm_list)



    def solve(self):
        i = 0
        delta = [100, 100, 100]
        while i < 20 and abs(sum(delta)) > 0.01:
            #print(f"Step {i + 1}")
            self.calculate_R_i_guess()
            #print("Calculated R_i_guess")
            self.calculate_tdoa_s()
            #print("Calculated tdoa_s")
            self.calculate_h()
            #print("Calculated h")
            self.calculate_G()
            #print("Calculated G")
            delta = self.calculate_deltaXY()
            #print(delta[0])
            self.update_x_y()
            i += 1


"""bs = [[5621990, 5616452, 5618652, 5619990], [636646, 636456, 640156, 636346], [100, 0, 0, 200], [5618222, 637900, 180]]
bs = [[3981552.712225135, 3979874.4314526464, 3970987.378112046, 3969313.5507772937], [764924.6490695147, 773609.222315067, 762894.8669536225, 771556.3950646253], [4907185.509521091, 4907185.509521091, 4915995.197143871, 4915995.197143871], [3975898.361382526, 768039.4736985303, 4911254.390409387]]
ms = [3975898.361382526, 768039.4736985303, 4911254.390409387]
print(f"Solver input: {bs, ms}")
solver = Foy(bs, ms)
solver.solve()
print(solver.guesses)
"""
"""
BS0 = np.array([49.449010, 11.064284, 0])
BS1 = np.array([49.457646, 11.088916, 0])
BS2 = np.array([49.447273, 11.088916, 0])
BS3 = np.array([49.452778, 11.077613, 500])
MS = np.array([49.449611, 11.075628, 300.57])

bs_list = [BS0, BS1, BS2, BS3, MS]
converted_bs = []
R_i_0 = [0, 0, 0, 0]
R_i_real = [0, 0, 0, 0]
R_i_guess = [0, 0, 0, 0]

h = np.zeros(3)
G = np.zeros((3, 3))

deltaXY = np.zeros(3)

guessed_position = [650745.995, 5479748.09, 250.]




print(f"Coordinates (lat/lon/alt):\nBS0: {BS0}\nBS1: {BS1}\nBS2: {BS2}\nBS3: {BS3}\nMS: {MS}\n")

points = [[5621990, 636646, 0], [5616452, 636456, 0], [5618652, 640156, 0], [5619990, 636346, 200]]
make_init_guess(points)
i = 0
guesses = [[],[],[]]
delta = [100, 100, 100]
while i < 20 and abs(sum(delta)) > 0.01:
    print(f"Step {i+1}")
    guesses[0].append(guessed_position[0])
    guesses[1].append(guessed_position[1])
    guesses[2].append(guessed_position[2])
    calculate_R_i_guess()
    calculate_tdoa_s()
    calculate_h()
    calculate_G()
    delta = calculate_deltaXY()
    print(delta[0])
    update_x_y()
    i += 1

print(guesses)
"""