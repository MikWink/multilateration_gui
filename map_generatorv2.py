import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.offline as pyo

HEIGHT_MAP_OFFSET = 521.9 - 101

class Map:
    def __init__(self):
        self.height_map = np.array(Image.open('ilmenau_r5km_c3.tif'))
        self.height_map = self.height_map[::-1]
        self.height_map_shape = self.height_map.shape
        self.fig = None

        self.offseted_height_map = [height + HEIGHT_MAP_OFFSET for height in self.height_map]

        ilmenau_real_area = np.array([[10.875, 50.625], [11, 50.75]])

        x_step = (ilmenau_real_area[1][0] - ilmenau_real_area[0][0]) / self.height_map.shape[1]
        y_step = (ilmenau_real_area[1][1] - ilmenau_real_area[0][1]) / self.height_map.shape[0]
        # print(x_step, y_step)

        # Define the real-world coordinates
        x_coords = np.arange(ilmenau_real_area[0][0], ilmenau_real_area[1][0], x_step)
        y_coords = np.arange(ilmenau_real_area[0][1], ilmenau_real_area[1][1], y_step)


        # Create a meshgrid from x and y coordinates
        X, Y = np.meshgrid(x_coords, y_coords)

        # Create a 3D surface plot with layout and scene updates
        self.fig = go.Figure(
            data=[],
            layout=go.Layout(margin=dict(l=0, r=0, b=0, t=20),
                             scene=dict(xaxis_title='Long', yaxis_title='Lat', zaxis_title='Height',
                                        aspectmode='manual', aspectratio=dict(x=1, y=2, z=0.1)))
        )
        self.fig.add_trace(self.make_trace([X, Y, self.offseted_height_map], type='surface'))
        self.fig.data[-1].uid = 'terrain'

        # Generate the HTML string of the figure
        html_string = pyo.plot(self.fig, include_plotlyjs='cdn', output_type='div')

        # Update the map window with the new plot
        with open('terrain_map.html', 'w') as file:
            file.write(html_string)

    def update(self):
        html_string = pyo.plot(self.fig, include_plotlyjs='cdn', output_type='div')
        with open('terrain_map.html', 'w') as file:
            file.write(html_string)

    def add_trace(self, trace, uid=None):
        self.fig.add_trace(trace)
        if uid is not None:
            self.fig.data[-1].uid = uid

    def make_trace(self, points, type='scatter', name='unnamed', color='black', size=3):
        try:

            if type == 'scatter':
                x = [float(point[1]) for point in points]
                y = [float(point[0]) for point in points]
                z = [float(point[2]) for point in points]
                return go.Scatter3d(x=x, y=y, z=z, mode='markers', name=name,
                                            marker=dict(size=size, color=color, opacity=0.8, symbol='x'))
            elif type == 'surface':
                x = points[0]
                y = points[1]
                z = points[2]
                return go.Surface(x=x, y=y, z=z, colorscale='Viridis', showscale=False, name='Terrain',
                                 hovertemplate='<b>Lat:</b> %{y}<br>' +
                                               '<b>Long:</b> %{x}<br>' +
                                               '<b>H:</b> %{z}<extra></extra>')
        except Exception as e:
            print(f'Error: {e}')

    def clear(self):
        terrain = self.fig.data[0]
        self.fig.data = [terrain]
        self.update()

    def remove_trace(self, uid):
        self.fig.data = [trace for trace in self.fig.data if trace.uid != uid]
        self.update()


