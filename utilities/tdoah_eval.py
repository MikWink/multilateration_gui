import json
from solver.tdoah import w2k, k2w
import numpy as np
from solver.tdoah_class import Tdoah
from solver.foy import Foy
import plotly.graph_objects as go
import plotly.offline as pyo
from evalution_functions import rmse, std, bias


def read_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        return data


def extract_bs_pos(file_path='../evaluation_data/station_setups/finland_real_setup.json'):
    bs_data = read_json(file_path)
    bs = []
    # Iterate over the values of the dictionary
    for key, value in bs_data.items():
        if len(bs) < 3:
            # Append a tuple of the values to the list
            bs.append((float(value['1']), float(value['2']), float(value['3'])))
    return bs


def extract_ms_pos(file_path='../evaluation_data/station_setups/ms_positions.json'):
    ms_data = read_json(file_path)
    ms = []
    for key, value in ms_data.items():
        ms.append((value['latitude'], value['longitude'], value['altitude']))
    return ms

def convert_coords(arr):
    for i, e in enumerate(arr):
        arr[i] = w2k(e[0], e[1], e[2])
    return arr



bs = extract_bs_pos()
ms = extract_ms_pos()
ms_h = []

for e in ms:
    ms_h.append(e[2])

bs = convert_coords(bs)
ms = convert_coords(ms)

print(ms[0])
print(f'ms_h: {ms_h}\n')

for i in range(5):
    if i == 2:
        continue
    else:
        print(f'\n\n####$####\n{i}\n########\n\n')
        endpoint_id = f'70-b3-d5-67-70-ff-03-4{i}'
        NPZ = np.load(f'../evaluation_data/NPZ_Data/70-b3-d5-67-70-ff-03-4{i}.npz', allow_pickle=True)
        tdoa_0 = NPZ['tdoa_0']
        tdoa_1 = NPZ['tdoa_1']
        print(NPZ)
        tdoa_2 = NPZ['tdoa_2']

        print(f'\ntdoa_0: {len(tdoa_0)}\ntdoa_1: {len(tdoa_1)}\ntdoa_2: {len(tdoa_2)}\n')
        print(f'bs: {bs}\nms: {ms}\nms_h: {ms_h}\n')

        x_values = []
        y_values = []
        z_values = []

        for j in range(len(tdoa_0)):
            print(endpoint_id)
            print(f'\nbs: {bs}\nms[i]: {ms[i]}\nms_h[i]: {ms_h[i]}\ni: {i}')
            solver = Tdoah(bs, ms[i], ms_h[i])
            solution = solver.solve(tdoa_0[j], tdoa_1[j])
            x_values.append(solution[0][0])
            y_values.append(solution[1][0])
            z_values.append(solution[2][0])

        x_values -= ms[i][0]
        y_values -= ms[i][1]
        z_values -= ms[i][2]

        # Calculate distances to the specific coordinate (ms[0])
        distances = np.sqrt((np.array(x_values)) ** 2 + (np.array(y_values)) ** 2 + (np.array(z_values)) ** 2)

        # Define a fixed maximum distance for normalization
        fixed_max_distance = 150  # Adjust this value as needed

        # Normalize distances based on the fixed maximum distance
        norm_distances = distances / fixed_max_distance

        # Clip the normalized distances to ensure they fit within the [0, 1] range
        norm_distances = np.clip(norm_distances, 0, 1)

        print(norm_distances)

        for j in range(len(tdoa_0)):
            foy_solver = Foy(bs, ms[i], tdoa_0[j], tdoa_1[j])






        # Create a 3D scatter plot
        fig = go.Figure(data=[
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode='markers',
                name='Calculated Positions',
                marker=dict(
                    size=3,
                    color=norm_distances,  # Color by z-values
                    colorscale=[[0, 'green'], [1, 'red']],  # Choose a colorscale
                    opacity=0.8
                )
            ),
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode='markers',
                name='Real Position',
                marker=dict(
                    size=5,
                    color='blue'
                )
            )]
        )

        # Update layout to fix the coordinate system
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X Axis', autorange=True),  # Set the range for x-axis
                yaxis=dict(title='Y Axis', autorange=True),  # Set the range for y-axis
                zaxis=dict(title='Z Axis', autorange=True)  # Set the range for z-axis
            ),
            title=endpoint_id
        )

        # Berechne die RMSE, std und Bias für jede Achse
        targets = np.array([0, 0, 0])  # Setze die Zielposition (real) als [0, 0, 0]

        # Konvertiere Listen in numpy Arrays für die Berechnungen
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        z_values = np.array(z_values)

        # Berechnungen für X-Achse
        x_rmse = rmse(x_values, targets[0])
        x_std = std(x_values)
        x_bias = bias(x_std, x_rmse)

        # Berechnungen für Y-Achse
        y_rmse = rmse(y_values, targets[1])
        y_std = std(y_values)
        y_bias = bias(y_std, y_rmse)

        # Berechnungen für Z-Achse
        z_rmse = rmse(z_values, targets[2])
        z_std = (std(z_values))
        z_bias = bias(z_std, z_rmse)

        # Ausgabe der Ergebnisse
        print(f'X-Achse: RMSE = {x_rmse}, Std = {x_std}, Bias = {x_bias}')
        print(f'Y-Achse: RMSE = {y_rmse}, Std = {y_std}, Bias = {y_bias}')
        print(f'Z-Achse: RMSE = {z_rmse}, Std = {z_std}, Bias = {z_bias}')

        # Optional: Ergebnisse in das HTML-Diagramm einfügen
        fig.add_annotation(
            text=f"X-Achse: RMSE = {x_rmse:.2f}, Std = {x_std:.2f}, Bias = {x_bias:.2f}<br>"
                 f"Y-Achse: RMSE = {y_rmse:.2f}, Std = {y_std:.2f}, Bias = {y_bias:.2f}<br>"
                 f"Z-Achse: RMSE = {z_rmse:.2f}, Std = {z_std:.2f}, Bias = {z_bias:.2f}",
            xref="paper", yref="paper",
            x=0.5, y=1.1, showarrow=False
        )



        # Generate the HTML string of the figure
        html_string = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')

        # Update the map window with the new plot
        with open(f'{endpoint_id}.html', 'w') as file:
            file.write(html_string)
