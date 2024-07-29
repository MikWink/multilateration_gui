from tdoah import Tdoah
from foy import Foy
from evaluation_functions import read_json, w2k

# Read the bs and ms position data from the json files
bs_data = read_json('finland_real_setup.json')
ms_data = read_json('ms_positions.json')

# Extract the height of the ms in meters before conversion
ms_h = []
for ms in ms_data:
    ms_h.append(ms[2])

# Convert bs and ms coordinates to cartesian coordinates
for i in range(len(bs_data)):
    bs_data[i] = w2k(bs_data[i][0], bs_data[i][1], bs_data[i][2])
for i in range(len(ms_data)):
    ms_data[i] = w2k(ms_data[i][0], ms_data[i][1], ms_data[i][2])

# Solve Localisation problem using TDOAH for each ms
print("TDOAH:")
for i, ms in enumerate(ms_data):
    solver = Tdoah(bs_data, ms)
    target = solver.solve(ms_h[i], 0.0038779747944916753, 0.007525018563308172)
    print(f'ms_{i}: {target}\n')

# Solve Localisation problem using Foy for each ms
print("FOY:")
for k, ms in enumerate(ms_data):
    print(f'Foy Input:\nBS: {bs_data}\nMS: {ms}\n')
    foy_solver = Foy(bs_data, ms)
    foy_solver.solve(0.0038779747944916753, 0.007525018563308172, -0.012970187379503978)
    print(f'ms_{k}: {foy_solver.guesses[0][-1], foy_solver.guesses[1][-1], foy_solver.guesses[2][-1]}\n')
