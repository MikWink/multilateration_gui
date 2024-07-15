import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.offline as pyo
from solver.tdoah import w2k, k2w


class Map:
    def __init__(self):
        im = Image.open('ilmenau_r5km_c3.tif')

        self.imarray = np.array(im)
        print(self.imarray)
        self.imarray = self.imarray[::-1]
        print(self.imarray)

        ilmenau_area = np.array([[632621, 5609803], [641086, 5623932]])
        ilmenau_real_area = np.array([[10.875, 50.625], [11, 50.75]])

        x_step = (ilmenau_real_area[1][0] - ilmenau_real_area[0][0]) / self.imarray.shape[1]
        y_step = (ilmenau_real_area[1][1] - ilmenau_real_area[0][1]) / self.imarray.shape[0]
        # print(x_step, y_step)

        # Define the real-world coordinates
        x_coords = np.arange(ilmenau_real_area[0][0], ilmenau_real_area[1][0], x_step)
        y_coords = np.arange(ilmenau_real_area[0][1], ilmenau_real_area[1][1], y_step)

        # Create a meshgrid from x and y coordinates
        X, Y = np.meshgrid(x_coords, y_coords)

        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=20)
        )

        # Create a 3D surface plot
        self.fig = go.Figure(
            data=[go.Surface(x=X, y=Y, z=self.imarray, colorscale='Viridis', showscale=False, name='Terrain',
                             hovertemplate='<b>Lat:</b> %{y}<br>' +
                                           '<b>Long:</b> %{x}<br>' +
                                           '<b>H:</b> %{z}<extra></extra>')],
            layout=layout)

        self.points = [[50.69419444444444, 10.916666666666668, 0], [50.68852777777777, 10.93011111111111, 0],
                       [50.68908333333333, 10.940277777777778, 0], [50.69275, 10.938694444444444, 0]]

        # Set the title and axis labels
        self.fig.update_layout(scene=dict(xaxis_title='Long', yaxis_title='Lat', zaxis_title='Height'))
        self.fig.update_scenes(aspectmode='manual', aspectratio=dict(x=1, y=2, z=0.2))

        # Generate the HTML string of the figure
        html_string = pyo.plot(self.fig, include_plotlyjs='cdn', output_type='div')

        # Update the map window with the new plot
        with open('terrain_map.html', 'w') as file:
            file.write(html_string)

    def update(self, points, empty_map=False):
        print(points)
        new_data = list(self.fig.data)
        if not empty_map:
            new_data = new_data[:1]
        else:
            new_data = new_data[:0]

        ms = points.pop(len(points) - 1)
        temp = ms[0]
        ms[0] = ms[1]
        ms[1] = temp

        x = []
        y = []
        z = []

        for point in points:
            x_tmp = point[1]
            y_tmp = point[0]
            x.append(x_tmp)
            y.append(y_tmp)
            # print(f'x_tmp: {x_tmp}, {type(x_tmp)}, y_tmp: {y_tmp}')
            if not empty_map:
                mapped_x = round(self.map_value(float(x_tmp), 10.875, 11.0, 0, self.imarray.shape[1] - 1))
                mapped_y = round(self.map_value(float(y_tmp), 50.625, 50.75, 0, self.imarray.shape[0] - 1))
                # print(f"mapped_x: {mapped_x}, mapped_y: {mapped_y}")
                z.append(self.imarray[mapped_y][mapped_x] + float(point[2]))
            else:
                z.append(float(point[2]))
                print("1")

        if not empty_map:
            mapped_ms_x = round(self.map_value(float(ms[0]), 10.875, 11.0, 0, self.imarray.shape[1] - 1))
            mapped_ms_y = round(self.map_value(float(ms[1]), 50.625, 50.75, 0, self.imarray.shape[0] - 1))
            # print(f'ms[0]: {ms[0]}, ms[1]: {ms[1]}')
            # print(f'mapped_ms_x: {mapped_ms_x}, mapped_ms_y: {mapped_ms_y}')
            ms_s = self.imarray[mapped_ms_y][mapped_ms_x] + int(ms[2])
        else:
            ms_s = float(ms[2])
            print("2")
        # print(f"ms_s: {ms}")
        self.fig.data = new_data
        self.fig.add_trace(go.Scatter3d(x=sorted(x), y=y, z=z, name='Basestations', mode='markers',
                                        marker=dict(symbol='square-open', size=8, color='red'), showlegend=False,
                                        hovertemplate='<b>Lat:</b> %{y}<br>' +
                                                      '<b>Long:</b> %{x}<br>' +
                                                      '<b>H:</b> %{z}<extra></extra>'))
        self.fig.add_trace(go.Scatter3d(x=[ms[0]], y=[ms[1]], z=[ms_s], name='Node', mode='markers',
                                        marker=dict(symbol='diamond-open', size=8, color='yellow'), showlegend=False,
                                        hovertemplate=
                                        '<b>Lat:</b> %{y}<br>' +
                                        '<b>Long:</b> %{x}<br>' +
                                        '<b>H:</b> %{z}<extra></extra>'))
        # Generate the HTML string of the figure
        html_string = pyo.plot(self.fig, include_plotlyjs='cdn', output_type='div')

        # Update the map window with the new plot
        with open('terrain_map.html', 'w') as file:
            file.write(html_string)

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

    def show_result(self, points, color='green', mode='lines+markers', eval_flag=False):
        try:
            if len(self.fig.data) > 3 and not eval_flag == True:
                data = list(self.fig.data)
                # print(f"Data: {data}")
                #data.pop(1)
                self.fig.data = data
                #print(f'fig_data: {data}')

        except Exception as e:
            print(f"Error: {e}")
        # print(f"show_result: {points}")

        # print('\n#######')
        # print(f'Points: {points}')
        points_conv = [[], [], []]
        # Converting coordinates in lat, lon
        for i, point in enumerate(points[0]):
            # print(f'i: {i}')
            fi, la, h = k2w(points[0][i], points[1][i], points[2][i])
            points_conv[0].append(fi)
            points_conv[1].append(la)
            points_conv[2].append(h)
            # print(f'Conv: {fi}, {la}, {h}')

        # print(f'Points_conv: {points_conv}')
        points_z = []

        for i, point in enumerate(points_conv[0]):
            mapped_x = round(self.map_value(points_conv[1][i], 10.875, 11, 0, self.imarray.shape[1]))
            mapped_y = round(self.map_value(points_conv[0][i], 50.625, 50.75, 0, self.imarray.shape[0]))
            # print(f'Test: {self.imarray[mapped_y][mapped_x] + point}')
            points_z.append(self.imarray[mapped_y][mapped_x] + points_conv[2][i])

        print(f'z: {points_z}')

        self.fig.add_trace(go.Scatter3d(x=points_conv[1], y=points_conv[0], z=points_z, name="Localisation", mode=mode,
                                        marker=dict(symbol='cross', size=8, color=color), showlegend=False))
        # Generate the HTML string of the figure
        html_string = pyo.plot(self.fig, include_plotlyjs='cdn', output_type='div')

        # Update the map window with the new plot
        with open('terrain_map.html', 'w') as file:
            file.write(html_string)


