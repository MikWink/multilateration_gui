import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.offline as pyo
from solver.tdoah import w2k, k2w
from scipy.interpolate import griddata
from pyproj import CRS, Transformer
import time


class Map:
    def __init__(self, points):
        self.points = points
        self.ms_point = points[-1]
        self.ms_trace = None
        self.bs_trace = None

        self.points = self.convert_translate_points()  # Translate the coordinates, so the mobile station is located at the origin
        self.fig = self.create_plot()  # Create the plot
        self.earth_on = False

    def create_plot(self):
        fig = go.Figure()
        print(self.points)
        print(self.points[:-1])
        self.ms_trace = go.Scatter3d(x=[self.points[-1, 0]], y=[self.points[-1, 1]], z=[self.points[-1, 2]],
                                     mode='markers',
                                     name='Mobile Station',
                                     marker=dict(size=10, color='red'))
        fig.add_trace(self.ms_trace)
        fig.data[-1].uid = 'ms'
        self.bs_trace = go.Scatter3d(x=self.points[:-1, 0], y=self.points[:-1, 1], z=self.points[:-1, 2],
                                     mode='markers',
                                     name='Basestations',
                                     marker=dict(size=5, color='blue'))
        fig.add_trace(self.bs_trace)
        fig.data[-1].uid = 'bs'

        return fig

    def add_trace(self, trace):
        self.fig.add_trace(trace)

    def get_points(self):
        return self.points

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
        conv_values = conv_values - conv_values[-1]

        return conv_values

    def calculate_ellipsoid_height(self, lat):
        """Calculates the height of the WGS-84 ellipsoid at a given latitude."""
        a = 6378137.0  # semi-major axis in meters
        b = 6356752.3142  # semi-minor axis in meters
        lat_rad = np.radians(lat)
        return (a ** 2 * np.cos(lat_rad) ** 2 + b ** 2 * np.sin(lat_rad) ** 2) ** 0.5

    def plot_ellipsoid(self, points, resolution=50):
        """
        Plots an interpolated surface based on given points in ECEF coordinates.
        """
        print(f'points: {points}')
        # Get coordinates at sea level and convert to ECEF
        points_ecef = np.array([w2k(float(p[0]), float(p[1]), 0.0) for p in points])
        print(f'Converted and lowered points: {points_ecef}')

        points_conv = np.array([w2k(float(p[0]), float(p[1]), float(p[2])) for p in points])
        points_ecef = np.array([p - points_conv[0] for p in points_ecef])

        # Create a grid for interpolation
        x_min, x_max = points_ecef[:, 0].min(), points_ecef[:, 0].max()
        y_min, y_max = points_ecef[:, 1].min(), points_ecef[:, 1].max()
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))

        # Interpolate z-values using griddata
        z_grid = griddata(points_ecef[:, :2], points_ecef[:, 2], (x_grid, y_grid), method='cubic')

        # Create Surface trace
        earth_trace = go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            colorscale='Blues',
            showscale=False,
            lighting=dict(ambient=0.8, diffuse=0.8, specular=0.2, roughness=0.1)
        )

        fig = go.Figure(data=[earth_trace])

        # Update layout (adjust labels and zoom as needed)
        fig.update_layout(
            scene=dict(xaxis_title='X (meters)', yaxis_title='Y (meters)', zaxis_title='Z (meters)'),
            title='Interpolated Earth Surface Section'
        )
        html_string = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
        # Update the map window with the new plot
        with open('coordinate_map_earth.html', 'w') as file:
            file.write(html_string)

        return earth_trace

    def wgs84_square_points(self, wgs_center, side_length_km=10):
        """
        Generates the corners of a square centered at a WGS84 coordinate.

        Args:
            wgs_center: List or tuple [latitude, longitude] in degrees.
            side_length_km: Length of each side of the square in kilometers (default: 10 km).

        Returns:
            List of lists, each representing a corner of the square:
            [[lat1, lon1], [lat2, lon2], [lat3, lon3], [lat4, lon4]]
        """
        alt = wgs_center[-1]
        wgs_center = wgs_center[:2]
        print(wgs_center)

        lat, lon = map(float, wgs_center)

        print(lat, lon, alt)

        # Azimuthal Equidistant projection centered at the point
        crs_aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon}")

        # Transformer for WGS84 to AEQD and back
        transformer_to_aeqd = Transformer.from_crs(CRS.from_epsg(4326), crs_aeqd)
        transformer_to_wgs84 = Transformer.from_crs(crs_aeqd, CRS.from_epsg(4326))

        # Calculate corners in AEQD (meters)
        half_side = side_length_km * 500  # Half the side length in meters
        corner_offsets = [(-half_side, -half_side), (half_side, -half_side), (half_side, half_side),
                          (-half_side, half_side)]
        corners_aeqd = [(0, 0)]  # Start with the center point
        corners_aeqd.extend([(x + 0, y + 0) for x, y in corner_offsets])  # Add offsets to get corners

        # Convert corners back to WGS84 (lat, lon)
        corners_wgs84 = [list(transformer_to_wgs84.transform(*xy)) for xy in corners_aeqd]
        for i, corner in enumerate(corners_wgs84):
            if i == 0:
                corners_wgs84[i].append(alt)
            else:
                corners_wgs84[i].append(0.0)

        return corners_wgs84

    def init_earth(self):
        self.earth_on = True
        print()
        earth_area = self.wgs84_square_points(self.ms_point)
        earth_trace = self.plot_ellipsoid(earth_area)
        self.add_trace(earth_trace)
        self.fig.data[-1].uid = 'earth'
        self.earth_uid = self.fig.data[-1].uid

    def toggle_earth(self):
        if self.earth_on:
            self.fig.data = [trace for trace in self.fig.data if trace.uid != 'earth']
            self.earth_on = False
        else:
            self.init_earth()

    def toggle_stations(self, station):
        trace = False
        for trace in self.fig.data:
            if trace.uid == station:
                trace = True

        if trace:
            self.fig.data = [trace for trace in self.fig.data if trace.uid != station]
        else:
            if station == 'ms':
                self.add_trace(self.ms_trace)
                self.fig.data[-1].uid = station
            else:
                self.add_trace(self.bs_trace)
                self.fig.data[-1].uid = station


points = [['50.673881', '10.953427', '0'], ['50.674918', '10.906670', '0'], ['50.704759', '10.919444', '0'],
          ['50.682106', '10.933819', '300'], ['50.686887', '10.936072', '150']]
map2 = Map(points)
map2.init_earth()
map2.show()
