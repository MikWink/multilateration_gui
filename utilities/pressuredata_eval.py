from server_api import ServerApi
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
from evalution_functions import std

# Constants
smarties_base = 565
smarties_relais = 371
R = 287.0  # J/(kg*K) Gasconstant
T = 293.15  # K Temperature
g = 9.81  # m/s^2 Gravitational acceleration


# Plot the graph with std deviation in the title
def plot_graph(xpoints, ypoints, avg_x, avg_y, std_dev_text=""):
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig = go.Figure(data=[go.Scatter(x=xpoints, y=ypoints, mode='lines+markers', name='Height Error')], layout=layout)

    fig.add_trace(go.Scatter(x=avg_x, y=avg_y, mode='lines', name='Average Height Error', line=dict(color='red')))

    fig.update_layout(
        xaxis_title='Time in Unix Timestamp',
        yaxis_title='Height Error in m',
        title=f'Height Error over Time (6.6.24 00:00:00 - 13.6.24 23:59:59)<br>{std_dev_text}',
    )

    html_string = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')

    # Update the map window with the new plot
    with open('../height_error.html', 'w') as file:
        file.write(html_string)


# Assuming your start and end datetimes
start_time = datetime(2024, 6, 6, 0, 0, 0)
end_time = datetime(2024, 6, 13, 23, 59, 59)

# Calculate total time span
total_time = end_time - start_time
desired_ticks = 20
tick_interval = total_time / (desired_ticks - 1)

# Generate tick values and labels
tickvals = [start_time + (i * tick_interval) for i in range(desired_ticks)]
ticktext = [tick.strftime('%Y-%m-%d %H:%M') for tick in tickvals]

server = ServerApi()
result = server.get('mqtt_consumer_pressure_kPa', start_time, end_time, '10m')

points_base = []
points_comp = []
xpoints_base = []
ypoints_base = []
xpoints_comp = []
ypoints_comp = []
delta_h = []

for e in result["result"][0]["values"]:
    points_base.append(e)

for e in result["result"][1]["values"]:
    points_comp.append(e)

progressor = 0
if len(points_base) < len(points_comp):
    temp = points_base
    points_base = points_comp
    points_comp = temp

for i, e in enumerate(points_comp):
    if e[0] == points_base[i + progressor][0]:
        xpoints_base.append(points_base[i + progressor][0])
        ypoints_base.append(points_base[i + progressor][1])
        xpoints_comp.append(e[0])
        ypoints_comp.append(e[1])
    else:
        progressor += 1

for i in range(len(xpoints_base)):
    delta_h.append(np.abs((R * T) / g * np.log(float(ypoints_base[i]) / float(ypoints_comp[i]))))

for i, e in enumerate(delta_h):
    delta_h[i] = e - (smarties_base - smarties_relais)

# Calculate average and standard deviation
avg = sum(delta_h) / len(delta_h)
std_dev = std(delta_h)  # Calculate the standard deviation

avg_x = [xpoints_base[0], xpoints_base[len(xpoints_base) - 1]]
avg_y = [avg, avg]

# Format the standard deviation for display
std_dev_text = f'Standard Deviation: {std_dev:.2f} m'

# Plot the graph with std deviation
plot_graph(xpoints_base, delta_h, avg_x, avg_y)