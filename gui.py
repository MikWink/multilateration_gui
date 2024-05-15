from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from map_generator import Map
import json
import sys
from solver.foy import Foy
import pvlib
from test_3_eq import PressureSolver

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.web = QWebEngineView()
        self.master_layout = QGridLayout()
        self.web.load(QUrl.fromLocalFile("/terrain_map.html"))
        self.mode = "4BS"
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
        if self.mode == "4BS":
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
        for i, block_label in enumerate(block_labels):
            # Add the block label
            layout.addWidget(QLabel(block_label), 0, i * 4)

            # Add the 'x:', 'y:', 'z:' labels
            for j in range(1, 5):
                if j < 4:
                    label = QLabel(f'{chr(120 + j - 1)}:')
                    if label.text() == 'z:':
                        self.height_label.append(label)
                    layout.addWidget(label, j, i * 4)
                else:
                    conv_label = QLabel("z:")
                    self.conv_labels.append(conv_label)
                    conv_label.setVisible(False)
                    res_label = QLabel("-")
                    self.res_labels.append(res_label)
                    res_label.setVisible(False)
                    layout.addWidget(conv_label, j, i * 4)
                    layout.addWidget(res_label, j, i * 4 + 1)

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
        values = [['5621990', '636646', '100'], ['5616452', '636456', '0'], ['5618652', '640156', '0'],
                  ['5619990', '636346', '200'], ['5618222', '637900', '180']]

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
        dropdown = QComboBox()
        dropdown.addItem("Foy")
        mode_switch = QPushButton("Mode: 4BS")
        mode_switch.clicked.connect(lambda: self.on_mode_switch_clicked(mode_switch))
        layout.addWidget(dropdown, 0, 0)
        layout.addWidget(mode_switch, 1, 0)
        layout.addWidget(calc_button, 2, 0)
        return layout

    def on_mode_switch_clicked(self, button):
        try:
            if button.text() == "Mode: 4BS":
                button.setText("Mode: PSI")
                self.mode = "PSI"
                for label in self.height_label:
                    label.setText("PSI:")
                for label in self.conv_labels:
                    label.setVisible(True)
                for label in self.res_labels:
                    label.setVisible(True)
            else:
                button.setText("Mode: 4BS")
                self.mode = "4BS"
                for label in self.height_label:
                    label.setText("z:")
                for label in self.conv_labels:
                    label.setVisible(False)
                for label in self.res_labels:
                    label.setVisible(False)
        except Exception as e:
            print(f"Error: {e}")

    def on_load_clicked(self):
        global input_fields
        # Open file dialog to select JSON file
        file_path, _ = QFileDialog.getOpenFileName(None, "Open File", "", "JSON Files (*.json);;All Files (*)")
        try:
            temp = [['5621990', '636646', '0'], ['5616452', '636456', '0'], ['5618652', '640156', '0'],
                    ['5619990', '636346', '200']]
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                for i, input_field in enumerate(input_fields):
                    i1 = i // 3
                    i2 = i % 3 + 1
                    # temp.append(data[str(i1)][str(i2)])
                    test = temp[i1][i2 - 1]
                    # Temporary solution
                    input_field.setText(test)

        except FileNotFoundError:
            print("File not found.")
        except json.JSONDecodeError:
            print("Invalid JSON format.")

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
            points = [[] for _ in range(5)]
            for i in range(5):
                points[i] = [0 for _ in range(3)]
            count = 0
            for i, input_field in enumerate(self.input_fields):
                i1 = i // 3
                i2 = i % 3
                print(i2)
                print(f"Mode: {self.mode}")
                if self.mode == "PSI" and i2 == 2:
                    z_value = self.res_labels[count].text()
                    count += 1
                    print(f"points[{i1}][{i2}]: {z_value}")
                    points[i1][i2] = str(round(float(z_value)))
                else:
                    points[i1][i2] = input_field.text()


                if i1 < 4 and i2 < 3:
                    self.bs_labels[i1][i2].setText(input_field.text())

            print(f"Points: {points}")
            self.generated_map.update(points)
            web.reload()
        except Exception as e:
            print(f"Error: {e}")

    def on_calc_clicked(self, web):
        if self.mode == "4BS":
            bs = [[], [], []]
            ms = []
            for i, input in enumerate(self.input_fields):
                if i < len(self.input_fields)-3:
                    if i%3 == 0:
                        bs[0].append(int(input.text()))
                    elif i%3 == 1:
                        bs[1].append(int(input.text()))
                    elif i%3 == 2:
                        bs[2].append(int(input.text()))
                if i == len(self.input_fields)-3:
                    ms = [int(self.input_fields[i].text()), int(self.input_fields[i+1].text()), int(self.input_fields[i+2].text())]
            bs.append(ms)
            print(f"Solver input: {bs, ms}")
            solver = Foy(bs, ms)
            solver.solve()
            print(f"Solver output: {solver.guesses}")
            for i,label in enumerate(self.result_labels):
                label.setText(str(solver.guesses[i][len(self.result_labels)-1]))
            self.generated_map.show_result(solver.guesses)
            # Call the calculation function
            # Update the map
            web.reload()
        elif self.mode == "PSI":
            self.calculate_height()
            # Define the known positions of the receivers and the reference station
            BS0 = 4039139.89, 897222.76, 4838608.59  # Reference station
            BS1 = 4027031.42, 890211.32, 4849775.99
            BS2 = 4047433.94, 904276.02, 4830281.62
            solver = PressureSolver(BS0, BS1, BS2, 5000, -6.044e-6, 6.916e-6)
            solver.solve()

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

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())