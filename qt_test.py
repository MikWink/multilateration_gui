import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPlainTextEdit, QGridLayout, QLineEdit
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt
from pyproj import Transformer, CRS
import numpy as np

import math

class BSCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = "BS 1"
        self.lat = 0
        self.long = 0
        self.alt = 0
        self.pos_x = 0
        self.pos_y = 0
        self.width = 400
        self.height = 300
        self.border_color = QColor(0, 0, 0)
        self.bg_color = QColor(255, 255, 255, 20)

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(self.border_color, 3))
        painter.setBrush(self.bg_color)
        painter.drawRect(self.pos_x, self.pos_y, self.width, self.height)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.web_widget = QWebEngineView(self)
        self.web_widget.load(QUrl.fromLocalFile("/terrain_map.html"))
        self.web_widget.setMinimumSize(1600, 1080)  # Or adjust to your desired minimum

        self.bs_widget = QWidget(self)
        self.bs_layout = self.layout_bs()
        self.bs_widget.setLayout(self.bs_layout)

        self.calc_widget = QWidget(self)
        self.calc_layout = self.layout_calc()
        self.calc_widget.setLayout(self.calc_layout)

        self.info_widget = QWidget(self)
        self.info_layout = self.layout_info()
        self.info_widget.setLayout(self.info_layout)

        self.action_screen = QWidget(self)
        self.action_screen_layout = QHBoxLayout()
        self.action_screen_layout.addWidget(self.bs_widget, 0)
        self.action_screen_layout.addWidget(self.web_widget, 2)
        self.action_screen_layout.addWidget(self.calc_widget, 0)
        self.action_screen.setLayout(self.action_screen_layout)

        self.main_layout = QVBoxLayout()  # Create the main layout
        self.main_layout.addWidget(self.action_screen, 1)
        self.main_layout.addWidget(self.info_widget, 0)

        self.central_widget = QWidget(self)
        self.central_widget.setLayout(self.main_layout)  # Set the main layout on the central widget
        self.setCentralWidget(self.central_widget)

        self.showMaximized()


        line_edits = self.bs_widget.findChildren(QLineEdit)
        print(line_edits)
        for i, line in enumerate(line_edits):
            line.textChanged.connect(lambda text, line_edits=line_edits, line=line, i=i: self.on_input_changed(line_edits, line, i))

    def on_input_changed(self, line_edits, line, index):
        print((index /3) % 2)
        if(int((index /3) % 2) == 0):
            print("Polar")
            count = int(index/3) * 3
            for i in range(3):
                print(f"Val: {line_edits[count+i].text()}, type {type(line_edits[count+i].text())}")
            if(line_edits[count].text() != "" and line_edits[count+1].text() != "" and line_edits[count+2].text() != ""):
                print("Calculating")
                coords = self.lla_to_xyz(int(line_edits[count].text()), int(line_edits[count+1].text()), int(line_edits[count+2].text()))
            for j in range(3):
                line_edits[count+3+j].setText(coords[j])
        else:
            print("Cartesian")
            count = int(index / 3) * 3
            for i in range(3):
                print(f"Val: {line_edits[count + i].text()}")
        print(index, type(index))
        print(f"Index: {index}")
        print(type(line.text()))
        print(f"Line: {line.text()}\n")

    def lla_to_xyz(self, lat, lon, alt):
        """Converts Latitude, Longitude, Altitude (LLA) coordinates (in degrees) to
        XYZ Cartesian coordinates (in meters) using the WGS84 reference model and UTM zone 32N.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            alt: Altitude in meters above the WGS84 ellipsoid (NOT sea level).

        Returns:
            A tuple (x, y, z) representing the XYZ coordinates in meters.
        """

        wgs84 = CRS.from_epsg(4326)  # WGS84 geographic coordinate system
        utm_zone_32N = CRS.from_epsg(32632)  # UTM zone 32N (northern hemisphere)
        transformer = Transformer.from_crs(wgs84, utm_zone_32N, always_xy=True)

        # First, convert to XYZ on the ellipsoid (z will be ellipsoidal height)
        x, y, z = transformer.transform(lon, lat, alt)

        # Now, adjust z for geoid undulation (difference between ellipsoid and mean sea level)
        geoid_undulation = self.get_geoid_undulation(lat, lon)  # Get undulation for the location
        z += geoid_undulation

        return x, y, z

    def get_geoid_undulation(self, lat, lon):
        """Fetches geoid undulation (height difference between ellipsoid and mean sea level)
        for a given latitude and longitude using EGM96 model.

        This is a simplified implementation. In real-world scenarios, you'd likely use a library
        or online service to get accurate geoid data.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.

        Returns:
            Geoid undulation in meters (approximate).
        """
        # EGM96 coefficients for Nuremberg area (very rough approximation)
        # For accurate results, you'd need a full geoid model or a service like:
        # https://geographiclib.sourceforge.io/cgi-bin/GeoidEval
        n_coeff = 360
        c, s = np.mgrid[-180:180, -90:90] / n_coeff
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        geoid = (50 * np.sin(2 * lat_rad) * np.cos(lon_rad) +
                 10 * np.cos(4 * lat_rad)) * np.exp(-(lat_rad ** 2 + lon_rad ** 2))
        return geoid
    def calculate_lla_for_bs(self, bs_index):
        def calculate():
            try:
                x = float(self.get_bs_field(bs_index, "x").text())
                y = float(self.get_bs_field(bs_index, "y").text())
                z = float(self.get_bs_field(bs_index, "z").text())

                # Replace with your actual XYZ to LLA conversion logic here
                lat, lon, alt = x / 111139, y / 111139, z
                self.get_bs_field(bs_index, "Lat").setText(str(math.degrees(lat)))
                self.get_bs_field(bs_index, "Long").setText(str(math.degrees(lon)))
                self.get_bs_field(bs_index, "Alt").setText(str(alt))
            except ValueError:
                pass  # Handle invalid input gracefully

        return calculate

    def get_bs_field(self, bs_index, field_name):
        return self.bs_layout.itemAt(
            bs_index * 7 + ["Lat", "Long", "Alt", "x", "y", "z"].index(field_name) + 2).widget()

    def layout_bs(self):
        labels = ["Lat: ", "Long: ", "Alt: ", "x: ", "y: ", "z: "]  # Labels for the text fields
        final_layout = QVBoxLayout()
        for j in range(3):
            bs_input = QWidget()
            bs_layout = QGridLayout()
            bs_layout.addWidget(QLabel(f"Base Station {j}"), 0, 0)
            for i in range(6):
                bs_layout.addWidget(QLabel(labels[i]), i+1, 0)
                bs_layout.addWidget(QLineEdit(), i+1, 1)
            bs_input.setLayout(bs_layout)
            final_layout.addWidget(bs_input)

        return final_layout

    def layout_calc(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Base Stations"))
        return layout

    def layout_info(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Base Stations"))
        return layout



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
