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
        try:
            print((index /3) % 2)
            if(int((index /3) % 2) == 0):
                print("Polar")
                count = int(index/3) * 3
                for i in range(3):
                    print(f"Val: {line_edits[count+i].text()}, type {type(line_edits[count+i].text())}")
                if(line_edits[count].text() != "" and line_edits[count+1].text() != "" and line_edits[count+2].text() != ""):
                    print("Calculating")
                    coords = self.w2k(float(line_edits[count].text()), float(line_edits[count + 1].text()), float(line_edits[count + 2].text()))
                for j in range(3):
                    line_edits[count+3+j].setText(str(coords[j]))
            else:
                print("Cartesian")
                count = int(index / 3) * 3
                for i in range(3):
                    print(f"Val: {line_edits[count + i].text()}")
                if (line_edits[count].text() != "" and line_edits[count + 1].text() != "" and line_edits[count + 2].text() != ""):
                    print("Calculating")
                    coords = self.k2w(float(line_edits[count].text()), float(line_edits[count + 1].text()),
                                      float(line_edits[count + 2].text()))
                for j in range(3):
                    line_edits[count - 2 + j].setText(str(coords[j]))
            print(index, type(index))
            print(f"Index: {index}")
            print(type(line.text()))
            print(f"Line: {line.text()}\n")
        except Exception as e:
            print(f"Error: {e}")
            pass

    def w2k(self, fi, la, h):
        """Converts WGS-84 coordinates (lat, lon, height) to Cartesian (x, y, z)."""
        #print(f'W2K:::: type of fi: {type(fi)} # {fi:.40f} # , la: {type(la)} # {la:.40f} # , h: {type(h)} # {h:.40f} # ')
        K = np.pi / 180
        #print(f'W2K:::: K: {K}')

        f = fi * K
        l = la * K

        A = 6378137
        B = 0.00669438000426
        C = 0.99330561999574

        #print(f'W2K:::: f: {f}')

        a = np.cos(f)
        b = np.sin(f)
        c = np.cos(l)
        d = np.sin(l)

        n = A / np.sqrt(1 - B * b ** 2)

        X = (n+h)*a*c
        Y = (n+h)*a*d;
        Z = (n*C+h)*b;


        #print(f'W2K::::\n{X}\n{Y}\n{Z}\n')
        return X, Y, Z

    def k2w(self, X, Y, Z):
        """
        Converts Cartesian coordinates (X, Y, Z) to WGS-84
        latitude (fi), longitude (la), and height (h) in degrees and meters.
        """

        # WGS-84 ellipsoid parameters
        A = 6378137.0  # Semi-major axis in meters
        B = 0.00669438002290  # Flattening

        # Longitude Calculation
        la_rad = np.arctan2(Y, X)
        la = np.degrees(la_rad)

        # Latitude and Height Calculation (Iterative Method)
        p = np.sqrt(X ** 2 + Y ** 2)
        fi_rad = np.arctan2(Z, p * (1 - B))  # Initial approximation

        for _ in range(5):  # Iterate for better accuracy
            N = A / np.sqrt(1 - B * np.sin(fi_rad) ** 2)
            h = p / np.cos(fi_rad) - N
            fi_rad = np.arctan2(Z, p * (1 - B * (N / (N + h))))

        fi = np.degrees(fi_rad)

        return fi, la, h

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
                self.get_bs_field(bs_index, "Lat").setText('hallo')
                self.get_bs_field(bs_index, "Long").setText(str(math.degrees(lon)))
                self.get_bs_field(bs_index, "Alt").setText(str(alt))
            except Exception as e:
                print(f"Error: {e}")
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
