import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.offline as pyo
from solver.tdoah import w2k, k2w
from scipy.interpolate import griddata
from pyproj import CRS, Transformer
import time


class EvalPlot:
    def __init__(self, points):
        self.points = points
        self.ms_point = points[-1]
        self.ms_trace = None
        self.bs_trace = None
        print(f'Points: {self.points}')
        print(f'MS: {self.ms_point}')

        self.points = self.convert_translate_points()  # Translate the coordinates, so the mobile station is located at the origin
        self.fig = self.create_plot()  # Create the plot
        self.earth_on = False

    def create_plot(self):
        fig = go.Figure()
        #print(self.points)
        #print(self.points[:-1])
        self.ms_trace = go.Scatter3d(x=[self.points[-1, 0]], y=[self.points[-1, 1]], z=[self.points[-1, 2]],
                                     mode='markers',
                                     name='Mobile Station',
                                     marker=dict(size=5, color='red', symbol='x'))
        fig.add_trace(self.ms_trace)
        fig.data[-1].uid = 'ms'
        self.bs_trace = go.Scatter3d(x=self.points[:-1, 0], y=self.points[:-1, 1], z=self.points[:-1, 2],
                                     mode='markers',
                                     name='Basestations',
                                     marker=dict(size=5, color='blue'))
        fig.add_trace(self.bs_trace)
        fig.data[-1].uid = 'bs'

        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(xaxis_title='X in m',
                       yaxis_title='Y in m',
                       zaxis_title='Z in m'),
            title='Localization Error Map'
        )

        return fig

    def add_trace(self, trace, uid=None):
        self.fig.add_trace(trace)
        if uid is not None:
            self.fig.data[-1].uid = uid

    def get_points(self):
        return self.points

    def get_ms(self):
        return self.ms_point

    def show(self):
        # Generate the HTML string of the figure
        html_string = pyo.plot(self.fig, include_plotlyjs='cdn', output_type='div')
        with open('coordinate_map.html', 'w') as file:
            file.write(html_string)

    def convert_translate_points(self):
        conv_values = []
        # convert the points to the correct coordinate system
        for i, point in enumerate(self.points):
            #print(float(point[0]), float(point[1]), float(point[2]))
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
        #print(f'points: {points}')
        # Get coordinates at sea level and convert to ECEF
        points_ecef = np.array([w2k(float(p[0]), float(p[1]), 0.0) for p in points])
        #print(f'Converted and lowered points: {points_ecef}')

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

        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=20)
        )

        fig = go.Figure(data=[earth_trace], layout=layout)

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
        #print(wgs_center)

        lat, lon = map(float, wgs_center)

        #print(lat, lon, alt)

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
        trace_there = False
        for trace in self.fig.data:
            if trace.uid == station:
                trace_there = True
        if trace_there:
            self.fig.data = [trace for trace in self.fig.data if trace.uid != station]
        else:
            if station == 'ms':
                self.add_trace(self.ms_trace)
                self.fig.data[-1].uid = station
            else:
                self.add_trace(self.bs_trace)
                self.fig.data[-1].uid = station
