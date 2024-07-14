import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.offline as pyo


class Map:
    def __init__(self):
        self.height_map = np.array(Image.open('ilmenau_r5km_c3.tif'))
        self.height_map = self.height_map[::-1]
        self.height_map_shape = self.height_map.shape
        self.fig = None
        print(self.height_map_shape)

        self.offseted_height_map = [height + (521.9-101) for height in self.height_map]

        ilmenau_real_area = np.array([[10.875, 50.625], [11, 50.75]])

        x_step = (ilmenau_real_area[1][0] - ilmenau_real_area[0][0]) / self.height_map.shape[1]
        y_step = (ilmenau_real_area[1][1] - ilmenau_real_area[0][1]) / self.height_map.shape[0]
        # print(x_step, y_step)

        # Define the real-world coordinates
        x_coords = np.arange(ilmenau_real_area[0][0], ilmenau_real_area[1][0], x_step)
        y_coords = np.arange(ilmenau_real_area[0][1], ilmenau_real_area[1][1], y_step)

        [print(i) for i, x in enumerate(x_coords) if x == 10.953333333333346]
        [print(i) for i, y in enumerate(y_coords) if y == 50.67388888888843]
        print(self.height_map[141][176])

        # Create a meshgrid from x and y coordinates
        X, Y = np.meshgrid(x_coords, y_coords)

        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=20)
        )

        # Create a 3D surface plot
        self.fig = go.Figure(
            data=[go.Surface(x=X, y=Y, z=self.offseted_height_map, colorscale='Viridis', showscale=False, name='Terrain',
                             hovertemplate='<b>Lat:</b> %{y}<br>' +
                                           '<b>Long:</b> %{x}<br>' +
                                           '<b>H:</b> %{z}<extra></extra>')],
            layout=layout)
        # Set the title and axis labels
        self.fig.update_layout(scene=dict(xaxis_title='Long', yaxis_title='Lat', zaxis_title='Height'))
        self.fig.update_scenes(aspectmode='manual', aspectratio=dict(x=1, y=2, z=0.1))

        # Generate the HTML string of the figure
        html_string = pyo.plot(self.fig, include_plotlyjs='cdn', output_type='div')

        # Update the map window with the new plot
        with open('terrain_map.html', 'w') as file:
            file.write(html_string)

    def update(self):
        html_string = pyo.plot(self.fig, include_plotlyjs='cdn', output_type='div')
        with open('terrain_map.html', 'w') as file:
            file.write(html_string)

    def add_trace(self, points, uid=None):
        x = [float(point[1]) for point in points]
        y = [float(point[0]) for point in points]
        z = [float(point[2]) for point in points]

        self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', name='BS+MS',
                                        marker=dict(size=3, color='red', opacity=0.8, symbol='x')))


map = Map()
map.add_trace([['50.673881', '10.953427', '521.5'], ['50.674918', '10.906670', '531.5'], ['50.704759', '10.919444', '530.7'], ['50.682106', '10.933819', '700'], ['50.686887', '10.936072', '630']], 'bs+ms')
map.update()