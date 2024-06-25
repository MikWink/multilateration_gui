from PyQt5.QtWidgets import QApplication, QVBoxLayout, QMainWindow, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView
from map_generator import Map
import json
import sys
import os
from solver.foy import Foy
import pvlib
from test_3_eq import PressureSolver
from solver.tdoah import *
from solver.tdoah_class import Tdoah


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.web = QWebEngineView()
        self.master_layout = QGridLayout()
        self.web.load(QUrl.fromLocalFile("/terrain_map.html"))
        self.mode = "4BS"
        self.unit = "Deg"
        self.unit_switch = None
        self.bs3_elements = []
        self.lat_labels = []
        self.long_labels = []
        self.height_label = []
        self.conv_labels = []
        self.res_labels = []
        self.user_input = QWidget()
        self.user_info = QWidget()
        self.calculation_input = QWidget()
        self.input_fields = []
        self.bs_labels = []
        self.result_labels = []
        self.generated_map = Map()
        self.setup()

    def setup(self):
        if self.unit == "Deg":
            print("Mode switch")
        else:
            print("Mode switch")
        self.user_input.setLayout(self.initUI())
        self.user_info.setLayout(self.initUserInfo())
        self.calculation_input.setLayout(self.initCalculationInput())


        self.master_layout.addWidget(self.user_input, 0, 0)
        self.master_layout.addWidget(self.web, 2, 0)
        self.master_layout.addWidget(self.user_info, 2, 1)
        self.master_layout.addWidget(self.calculation_input, 0, 1)

        central_widget = QWidget()
        central_widget.setLayout(self.master_layout)
        self.setCentralWidget(central_widget)

    def initUI(self):
        layout = QGridLayout()

        # Define the labels for each block
        block_labels = ['BS0:', 'BS1:', 'BS2:', 'BS3:', 'MS:']
        temp = [['5621990', '636646', '100'], ['5616452', '636456', '0'], ['5618652', '640156', '0'],
                ['5619990', '636346', '200'], ['5618222', '637900', '180']]
        temp = [['49.46941666666667', '11.067777777777778', '100'], ['49.45769444444445', '11.08913888888889', '0'], ['49.459694444444445', '11.060194444444445', '0'],
                  ['49.459694444444445', '11.060194444444445', '200'], ['49.459250000000004', '11.074583333333333', '180']]
        coord_labels = ['Lat', 'Long', 'Alt']
        for i, block_label in enumerate(block_labels):
            if len(block_label) - 1 == i:
                bs3_label = QLabel(block_label)
                self.bs3_elements.append(bs3_label)
                layout.addWidget(bs3_label, 0, i * 4)

                # Add the 'x:', 'y:', 'z:' labels
                for j in range(1, 6):
                    if j < 4:
                        label = QLabel(f'{coord_labels[j - 1]}:')
                        if label.text() == 'Lat:':
                            self.bs3_elements.append(label)
                            self.lat_labels.append(label)
                        if label.text() == 'Long:':
                            self.bs3_elements.append(label)
                            self.long_labels.append(label)
                        if label.text() == 'Alt:':
                            self.bs3_elements.append(label)
                            self.height_label.append(label)
                        layout.addWidget(label, j, i * 4)

                # Add the text inputs for 'x', 'y', 'z'
                for j in range(1, 4):
                    input = QLineEdit()
                    input.setText(temp[i][j - 1])
                    layout.addWidget(input, j, i * 4 + 1)
                    self.input_fields.append(input)  # Keep track of input fields
                    self.bs3_elements.append(input)
            else:
                # Add the block label
                layout.addWidget(QLabel(block_label), 0, i * 4)

                # Add the 'x:', 'y:', 'z:' labels
                for j in range(1, 6):
                    if j < 4:
                        label = QLabel(f'{coord_labels[j-1]}:')
                        if label.text() == 'Lat:':
                            self.lat_labels.append(label)
                        if label.text() == 'Long:':
                            self.long_labels.append(label)
                        if label.text() == 'Alt:':
                            self.height_label.append(label)
                        layout.addWidget(label, j, i * 4)

                # Add the text inputs for 'x', 'y', 'z'
                for j in range(1, 4):
                    input = QLineEdit()
                    input.setText(temp[i][j - 1])
                    layout.addWidget(input, j, i * 4 + 1)
                    self.input_fields.append(input)  # Keep track of input fields

        save_button = QPushButton("Save")
        save_button.clicked.connect(lambda: self.on_save_clicked())  # Connect the button to the slot
        load_button = QPushButton("Load")
        load_button.clicked.connect(lambda: self.on_load_clicked())
        update_button = QPushButton("Update")
        update_button.clicked.connect(lambda: self.on_update_clicked(self.web))
        layout.addWidget(save_button, 1, 22)
        layout.addWidget(load_button, 2, 22)
        layout.addWidget(update_button, 3, 22)

        return layout

    def initUserInfo(self):
        layout = QGridLayout()
        user_info = QLabel("Info:")
        layout.addWidget(user_info, 0, 0)
        block_labels = ['BS0:', 'BS1:', 'BS2:', 'BS3:', 'MS:']
        """values = [['5621990', '636646', '100'], ['5616452', '636456', '0'], ['5618652', '640156', '0'],
                  ['5619990', '636346', '200'], ['5618222', '637900', '180']]"""
        values = [['49.46941666666667', '11.067777777777778', '100'], ['49.45769444444445', '11.08913888888889', '0'], ['49.459694444444445', '11.060194444444445', '0'],
                  ['49.459694444444445', '11.060194444444445', '200'], ['49.459250000000004', '11.074583333333333', '180']]
        deg_values = [[50.683093, 10.933442, 100], [50.690786, 10.933712, 400], [50.689108, 10.928814, 250]]

        for i, block_label in enumerate(block_labels):
            # Add the block label
            layout.addWidget(QLabel(block_label), i * 4 + 1, 0)

            bs_labels_tmp = []
            # Add the 'x:', 'y:', 'z:' labels
            for j in range(1, 4):
                layout.addWidget(QLabel(f'{chr(120 + j - 1)}:'), i * 4 + j + 1, 0)
                if i < 4:
                    label = QLabel(f'{values[i][j - 1]}')
                    bs_labels_tmp.append(label)
                    layout.addWidget(label, i * 4 + j + 1, 1)
                else:
                    self.result_labels.append(QLabel('-'))
                    layout.addWidget(self.result_labels[j-1], i * 4 + j + 1, 1)
            self.bs_labels.append(bs_labels_tmp)
        return layout

    def initCalculationInput(self):
        layout = QGridLayout()
        calc_button = QPushButton("Calculate")
        calc_button.clicked.connect(lambda: self.on_calc_clicked(self.web))
        #dropdown = QComboBox()
        #dropdown.addItem("Foy")
        mode_switch = QPushButton("Mode: 4BS")
        mode_switch.clicked.connect(lambda: self.on_mode_switch_clicked(mode_switch))
        self.unit_switch = QPushButton("Unit: Deg")
        self.unit_switch.clicked.connect(lambda: self.on_unit_switch_clicked(self.unit_switch))
        #layout.addWidget(dropdown, 0, 0)
        layout.addWidget(mode_switch, 1, 0)
        layout.addWidget(self.unit_switch, 2, 0)
        layout.addWidget(calc_button, 3, 0)
        return layout

    def on_mode_switch_clicked(self, button):
        try:
            if button.text() == "Mode: 4BS":
                button.setText("Mode: 3BS")
                self.mode = "3BS"
                for e in self.bs3_elements:
                    e.setVisible(False)
            else:
                button.setText("Mode: 4BS")
                self.mode = "4BS"
                for e in self.bs3_elements:
                    e.setVisible(True)
        except Exception as e:
            print(f"Error: {e}")

    def on_unit_switch_clicked(self, button):
        try:
            if button.text() == "Unit: Deg":
                button.setText("Unit: XYZ")
                self.unit = "XYZ"
                for label in self.lat_labels:
                    label.setText("X:")
                for label in self.long_labels:
                    label.setText("Y:")
                for label in self.height_label:
                    label.setText("Z:")
                for i, input in enumerate(self.input_fields):
                    if i % 3 == 0:
                        x, y, z = w2k(float(self.input_fields[i].text()), float(self.input_fields[i+1].text()), float(self.input_fields[i+2].text()))
                        print(f'x: {x}, y: {y}, z: {z}')
                        self.input_fields[i].setText(str(x))
                        self.input_fields[i+1].setText(str(y))
                        self.input_fields[i+2].setText(str(z))

            else:
                button.setText("Unit: Deg")
                self.unit = "Deg"
                for label in self.lat_labels:
                    label.setText("Lat:")
                for label in self.long_labels:
                    label.setText("Long:")
                for label in self.height_label:
                    label.setText("Alt:")
                for i, input in enumerate(self.input_fields):
                    if i % 3 == 0:
                        fi, la, alt = k2w(float(self.input_fields[i].text()), float(self.input_fields[i + 1].text()),
                                      float(self.input_fields[i + 2].text()))
                        print(f'fi: {fi}, la: {la}, alt: {alt}')
                        self.input_fields[i].setText(str(fi))
                        self.input_fields[i + 1].setText(str(la))
                        self.input_fields[i + 2].setText(str(alt))

        except Exception as e:
            print(f"Error: {e}")

    def on_load_clicked(self):
        # Open file dialog to select JSON file
        file_path, _ = QFileDialog.getOpenFileName(None, "Open File", "", "JSON Files (*.json);;All Files (*)")
        try:
            temp = [['5621990', '636646', '0'], ['5616452', '636456', '0'], ['5618652', '640156', '0'],
                    ['5619990', '636346', '200']]
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                print(f'data: {data}')
                for i, input_field in enumerate(self.input_fields):
                    i1 = i // 3
                    i2 = i % 3 + 1
                    print(f'i1: {i1}, i2: {i2}')
                    # temp.append(data[str(i1)][str(i2)])
                    test = data[f'{i1}'][f'{i2}']
                    # Temporary solution
                    input_field.setText(str(test))

        except FileNotFoundError:
            print("File not found.")
        except json.JSONDecodeError:
            print("Invalid JSON format.")
        except Exception as e:
            print(f"Error: {e}")

    def on_save_clicked(self):
        # Collect data from input fields
        data = []
        for i in range(len(self.input_fields)):
            data.append(self.input_fields[i].text())

        # Save data to a JSON file
        self.save_json(data)

    def save_json(self, data):
        data = {
            '0': {
                '1': data[0],
                '2': data[1],
                '3': data[2]
            },
            '1': {
                '1': data[3],
                '2': data[4],
                '3': data[5]
            },
            '2': {
                '1': data[6],
                '2': data[7],
                '3': data[8]
            },
            '3': {
                '1': data[9],
                '2': data[10],
                '3': data[11]
            },
            '4': {
                '1': data[12],
                '2': data[13],
                '3': data[14]
            }
        }
        file_path, _ = QFileDialog.getSaveFileName(None, "Save File", "", "JSON Files (*.json);;All Files (*)",
                                                   "*.json")
        with open(file_path, "w") as file:
            json.dump(data, file)
        print("JSON file saved!")

    def on_update_clicked(self, web):
        try:
            if self.mode == "4BS":
                num = 5
            else:
                num = 4
            points = [[] for _ in range(num)]
            for i in range(num):
                print(f'i: {i}')
                points[i] = [0 for _ in range(3)]
            count = 0
            print(f'points: {points}')
            print(f'last input: {self.input_fields[len(self.input_fields)-1].text()}')
            points[num-1] = [float(self.input_fields[len(self.input_fields)-3].text()), float(self.input_fields[len(self.input_fields)-2].text()), float(self.input_fields[len(self.input_fields)-1].text())]
            print(f'Points: {points}')
            for i, input_field in enumerate(self.input_fields):
                if self.mode == "3BS" and i > 8:
                    break
                else:
                    i1 = i // 3
                    i2 = i % 3
                    # why did i do that????
                    """if i1 == 4:
                        i1 = i1 - 1"""
                    print(f'i1: {i1}, i2: {i2}')
                    points[i1][i2] = input_field.text()

                if i1 < 4 and i2 < 3:
                    self.bs_labels[i1][i2].setText(input_field.text())

            print(f"Points: {points}")
            self.generated_map.update(points)
            web.reload()
        except Exception as e:
            print(f"Errror: {e}")

    def on_calc_clicked(self, web):
        try:
            if self.mode == "4BS":
                if self.unit == "Deg":
                    self.on_unit_switch_clicked(self.unit_switch)
                bs = [[], [], []]
                ms = []
                print(f"Input fields: {self.input_fields}")
                for i, input in enumerate(self.input_fields):
                    print(f'i: {i}, input: {input.text()}')
                    if i < len(self.input_fields)-3:
                        if i%3 == 0:
                            bs[0].append(float(input.text()))
                        elif i%3 == 1:
                            bs[1].append(float(input.text()))
                        elif i%3 == 2:
                            bs[2].append(float(input.text()))
                    if i == len(self.input_fields)-3:
                        ms = [float(self.input_fields[i].text()), float(self.input_fields[i+1].text()), float(self.input_fields[i+2].text())]
                bs.append(ms)
                print(f"Solver input: {bs, ms}")
                solver = Foy(bs, ms)
                solver.solve()
                print(f"Solver output: {solver.guesses}")
                for i,label in enumerate(self.result_labels):
                    label.setText(str(solver.guesses[i][len(self.result_labels)-1]))

                print(f"Solver output: {solver.guesses}")
                self.generated_map.show_result(solver.guesses)
                # Call the calculation function
                # Update the map
                web.reload()
            elif self.mode == "3BS":
                os.system('cls')
                #self.calculate_height()
                bs = [[], [], []]
                ms = []
                print(f"Input fields: {self.input_fields}")
                for i, input in enumerate(self.input_fields):
                    print(f'i: {i}, input: {input.text()}')
                    if i < len(self.input_fields) - 6:
                        if i % 3 == 0:
                            bs[0].append(float(input.text()))
                        elif i % 3 == 1:
                            bs[1].append(float(input.text()))
                        elif i % 3 == 2:
                            bs[2].append(float(input.text()))
                    if i == len(self.input_fields) - 3:
                        ms = [float(self.input_fields[i].text()), float(self.input_fields[i + 1].text()),
                              float(self.input_fields[i + 2].text())]
                print(f"Solver input: {bs, ms}")
                solver = Tdoah(bs, ms)
                target = solver.solve()
                print(f'Final: {target}')
                self.generated_map.show_result(target)
                web.reload()

        except Exception as e:
            print(f'Error: {e}')

    def calculate_height(self):
        heights = []
        try:
            for i in range(1, len(self.input_fields)+1):
                if i % 3 == 0 and not i == 0:
                    heights.append(int(self.input_fields[i-1].text()))

            for i, height in enumerate(heights):
                self.res_labels[i].setText(str(pvlib.atmosphere.pres2alt(height)))
        except Exception as e:
            print(f"Error: {e}")


class MainWindow2(QMainWindow):
    def __init__(self):
        super().__init__()
        self.showMaximized()

        self.view = QWebEngineView(self)
        self.view.load(QUrl.fromLocalFile("/terrain_map.html"))

        #self.setCentralWidget(self.view)
        self.create_base_station_cards()

    def create_base_station_cards(self):
        # Example base station data
        base_stations = [
            ("Base Station 1", 100, 200, 50, 30),
            ("Base Station 2", 350, 150, 80, 45),
            # ... more base stations
        ]

        for name, x, y, z, height in base_stations:
            card = BaseStationCard(name, x, y, z, height)

            # Calculate card position relative to the chart/web view
            card_x = self.view.x() + x  # Adjust as needed
            card_y = self.view.y() + y

            #card.setGeometry(card_x, card_y, card.sizeHint().width(), card.sizeHint().height())
            card.show()


class BaseStationCard(QWidget):
    def __init__(self, name, x, y, z, height):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"{name}:"))
        layout.addWidget(QLabel(f"X: {x}, Y: {y}, Z: {z}"))
        layout.addWidget(QLabel(f"Height: {height}"))
        self.setLayout(layout)

        self.setStyleSheet("""
            BaseStationCard {
                background-color: rgba(255, 255, 255, 180); /* Semi-transparent white */
                border: 1px solid lightgray;
                padding: 10px;
                border-radius: 5px;
            }
        """)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())