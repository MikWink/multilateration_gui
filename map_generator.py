import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.offline as pyo

class Map:
    def __init__(self):
        im = Image.open('ilmenau_r5km_c3.tif')

        self.imarray = np.array(im)

        ilmenau_area = np.array([[632621, 5609803], [641086, 5623932]])

        x_step = (ilmenau_area[1][1] - ilmenau_area[0][1]) / self.imarray.shape[1]
        y_step = (ilmenau_area[1][0] - ilmenau_area[0][0]) / self.imarray.shape[0]
        # print(x_step, y_step)

        # Define the real-world coordinates
        y_coords = np.arange(632621, 641086, y_step)
        x_coords = np.arange(5609803, 5623932, x_step)

        print(len(x_coords), len(y_coords))
        print(self.imarray.shape)

        # Create a meshgrid from x and y coordinates
        X, Y = np.meshgrid(x_coords, y_coords)

        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=80)
        )

        # Create a 3D surface plot
        self.fig = go.Figure(data=[go.Surface(x=X, y=Y, z=self.imarray, colorscale='Viridis', showscale=False, name='Terrain')],
                        layout=layout)

        self.points = [[5621990, 636646, 0], [5616452, 636456, 0], [5618652, 640156, 0], [5619990, 636346, 200]]
        self.plot_point()



        # Set the title and axis labels
        self.fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Height'))
        self.fig.update_scenes(aspectmode='manual', aspectratio=dict(x=2, y=1, z=0.2))

        # Generate the HTML string of the figure
        html_string = pyo.plot(self.fig, include_plotlyjs='cdn', output_type='div')

        # Update the map window with the new plot
        with open('terrain_map.html', 'w') as file:
            file.write(html_string)

        self.update([['5621990', '636646', '0'], ['5616452', '636456', '0'], ['5618652', '640156', '0'],
            ['5619990', '636346', '200'], ['5618222', '637900', '180']])

    def update(self, points):
        try:
            if len(self.fig.data) > 3:
                data = list(self.fig.data)
                data.pop(1)
                self.fig.data = data

        except Exception as e:
            print(f"Error: {e}")

        new_data = list(self.fig.data)
        new_data.pop(1)
        new_data.pop(1)

        ms = points.pop(len(points) - 1)

        x = []
        y = []
        z = []

        for point in points:
            x_tmp = point[0]
            y_tmp = point[1]
            x.append(x_tmp)
            y.append(y_tmp)
            mapped_x = round(self.map_value(int(x_tmp), 5609803, 5623932, 0, self.imarray.shape[1]))
            mapped_y = round(self.map_value(int(y_tmp), 632621, 641086, 0, self.imarray.shape[0]))
            z.append(self.imarray[mapped_y][mapped_x] + int(point[2]))

        mapped_ms_x = round(self.map_value(int(ms[0]), 5609803, 5623932, 0, self.imarray.shape[1]))
        mapped_ms_y = round(self.map_value(int(ms[1]), 632621, 641086, 0, self.imarray.shape[0]))
        ms_s = self.imarray[mapped_ms_y][mapped_ms_x] + int(ms[2])
        self.fig.data = new_data
        self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, name='Basestations', mode='markers', marker=dict(symbol='square-open', size=5, color='red'), showlegend=False))
        self.fig.add_trace(go.Scatter3d(x=[ms[0]], y=[ms[1]], z=[ms_s], name='Node', mode='markers', marker=dict(symbol='diamond-open', size=5, color='yellow'), showlegend=False))
        # Generate the HTML string of the figure
        html_string = pyo.plot(self.fig, include_plotlyjs='cdn', output_type='div')

        # Update the map window with the new plot
        with open('terrain_map.html', 'w') as file:
            file.write(html_string)


    def plot_point(self):
        points_x = []
        points_y = []
        points_z = []
        for point in self.points:
            x = point[0]
            y = point[1]
            points_x.append(x)
            points_y.append(y)
            mapped_x = round(self.map_value(x, 5609803, 5623932, 0, self.imarray.shape[1]))
            mapped_y = round(self.map_value(y, 632621, 641086, 0, self.imarray.shape[0]))
            points_z.append(self.imarray[mapped_y][mapped_x] + point[2])

        mapped_ms_x = round(self.map_value(5618222, 5609803, 5623932, 0, self.imarray.shape[1]))
        mapped_ms_y = round(self.map_value(637900, 632621, 641086, 0, self.imarray.shape[0]))
        ms_s = self.imarray[mapped_ms_y][mapped_ms_x] + 180
        self.fig.add_trace(go.Scatter3d(x=[5618222], y=[637900], z=[ms_s], name='Node', mode='markers',
                                        marker=dict(symbol='diamond-open', size=5, color='yellow'), showlegend=False))
        self.fig.add_trace(go.Scatter3d(x=points_x, y=points_y, z=points_z, name='Basestations', mode='markers',
                                   marker=dict(symbol='square-open', size=5, color='red'), showlegend=False))

    def map_value(self, value, min_value, max_value, new_min, new_max):
        """
        Maps a value from one range to another using linear interpolation.

        Args:
        - value (float): The value to be mapped.
        - min_value (float): The minimum value of the original range.
        - max_value (float): The maximum value of the original range.
        - new_min (float): The minimum value of the new range.
        - new_max (float): The maximum value of the new range.

        Returns:
        - mapped_value (float): The mapped value.
        """
        # Perform linear interpolation
        mapped_value = ((value - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
        return mapped_value

    def show_result(self, points):
        try:
            if len(self.fig.data) > 3:
                data = list(self.fig.data)
                print(f"Data: {data}")
                data.pop(1)
                self.fig.data = data

        except Exception as e:
            print(f"Error: {e}")
        print(f"show_result: {points}")
        try:
            points_z = []
            for i, point in enumerate(points[2]):
                mapped_x = round(self.map_value(points[0][i], 5609803, 5623932, 0, self.imarray.shape[1]))
                mapped_y = round(self.map_value(points[1][i], 632621, 641086, 0, self.imarray.shape[0]))

                points_z.append(self.imarray[mapped_y][mapped_x] + point)


        except Exception as e:
            print(f"Error: {e}")
        self.fig.add_trace(go.Scatter3d(x=points[0], y=points[1], z=points_z, name="Localisation", marker=dict(symbol='cross', size=5, color='green'), showlegend=False))
        # Generate the HTML string of the figure
        html_string = pyo.plot(self.fig, include_plotlyjs='cdn', output_type='div')

        # Update the map window with the new plot
        with open('terrain_map.html', 'w') as file:
            file.write(html_string)