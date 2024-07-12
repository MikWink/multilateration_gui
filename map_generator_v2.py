import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.offline as pyo
from solver.tdoah import w2k, k2w


class Map:
    def __init__(self, points):
        self.points = points

        self.points = self.convert_translate_points()  # Translate the coordinates, so the mobile station is located at the origin
        self.fig = self.create_plot()  # Create the plot

    def create_plot(self):
        fig = go.Figure()
        print(self.points)
        print(self.points[:-1])
        fig.add_trace(
            go.Scatter3d(x=[self.points[-1, 0]], y=[self.points[-1, 1]], z=[self.points[-1, 2]], mode='markers',
                         marker=dict(size=10, color='red')))
        fig.add_trace(
            go.Scatter3d(x=self.points[:-1, 0], y=self.points[:-1, 1], z=self.points[:-1, 2], mode='markers',
                         marker=dict(size=5, color='blue')))

        return fig

    def add_trace(self, trace):
        self.fig.add_trace(trace)

    def show(self):
        # Generate the HTML string of the figure
        html_string = pyo.plot(self.fig, include_plotlyjs='cdn', output_type='div')
        with open('coordinate_map.html', 'w') as file:
            file.write(html_string)

    def convert_translate_points(self):
        conv_values = []
        # convert the points to the correct coordinate system
        for i, point in enumerate(self.points):
            print(float(point[0]), float(point[1]), float(point[2]))
            conv_values.append(w2k(float(point[0]), float(point[1]), float(point[2])))

        # translate the points so the mobile station is located at the origin
        conv_values = np.array(conv_values)
        #conv_values = conv_values - conv_values[-1]

        return conv_values


def plot_ellipsoid(lat_center, lon_center, radius_meters, resolution=50):
    """
        Plots a section of the Earth's ellipsoid centered around given coordinates.
        """
    # WGS-84 parameters (semi-major axis and flattening)
    a = 6378137.0  # semi-major axis in meters
    f = 0.0000669438000426  # flattening

    # Convert latitude and longitude to radians
    lat_rad = np.radians(lat_center)
    lon_rad = np.radians(lon_center)

    # Calculate linear distance for the given radius on Earth's surface
    linear_distance = radius_meters / a

    # Generate grid points in latitude and longitude (restrict to the section)
    lat_range = np.linspace(lat_rad - linear_distance, lat_rad + linear_distance, resolution)
    lon_range = np.linspace(lon_rad - linear_distance, lon_rad + linear_distance, resolution)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

    # Calculate Cartesian coordinates (ECEF)
    N = a / np.sqrt(1 - f * (2 - f) * np.sin(lat_grid) ** 2)
    x = (N + 0) * np.cos(lat_grid) * np.cos(lon_grid)
    y = (N + 0) * np.cos(lat_grid) * np.sin(lon_grid)
    z = (N * (1 - f) ** 2 + 0) * np.sin(lat_grid)

    # Create Surface trace
    earth_trace = go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale='Blues',
        showscale=False,
        lighting=dict(
            ambient=0.8,
            diffuse=0.8,
            specular=0.2,
            roughness=0.1
        )
    )
    return earth_trace
    """html_string = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
    with open('coordinate_map_earth.html', 'w') as file:
        file.write(html_string)"""



points = [['50.673881', '10.953427', '0'], ['50.674918', '10.906670', '0'], ['50.704759', '10.919444', '0'],
          ['50.682106', '10.933819', '300'], ['50.686887', '10.936072', '150']]
map = Map(points)
map.add_trace(plot_ellipsoid(50.686887, 10.936072, 1000, 50))
map.show()





