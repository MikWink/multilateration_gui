from tdoah import Tdoah
from foy import Foy
from evaluation_functions import read_json, w2k
import numpy as np
import json




bs_data = read_json('finland_real_setup.json')
ms_data = read_json('ms_positions.json')
ms_h = []
print(f'bs_data: {bs_data}')
for ms in ms_data:
    ms_h.append(ms[2])
print(f'ms_data: {ms_data}\n')

# Convert the data to cartesian coordinates
for i in range(len(bs_data)):
    bs_data[i] = w2k(bs_data[i][0], bs_data[i][1], bs_data[i][2])
for i in range(len(ms_data)):
    #ms_h.append(ms_data[i][2])
    ms_data[i] = w2k(ms_data[i][0], ms_data[i][1], ms_data[i][2])
print(f'Converted bs_data: {bs_data}')

print(f'Converted ms_data: {ms_data}')
print(f'Extracted ms_h: {ms_h}\n')

print("TDOAH:")
for i, ms in enumerate(ms_data):
    solver = Tdoah(bs_data, ms, ms_h[i])
    target = solver.solve()
    print(f'ms_{i}: {target}')


print("\nFOY:")
for k, ms in enumerate(ms_data):
    print(f'Foy Input:\nBS: {bs_data}\nMS: {ms}\n')
    foy_solver = Foy(bs_data, ms)
    foy_solver.solve()
    print(f'ms_{k}: {foy_solver.guesses[0][-1], foy_solver.guesses[1][-1], foy_solver.guesses[2][-1]}\n')