from PyQt5.QtWidgets import QApplication, QSizePolicy, QSlider, QVBoxLayout, QMainWindow, QWidget, QGridLayout, QLabel, \
    QLineEdit, QPushButton, QComboBox, QFileDialog, QDialog
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go

from map_generator import Map
from map_generator_v2 import Map as Map2
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
        self.eval_window = None
        self.web = QWebEngineView()
        self.master_layout = QGridLayout()
        self.web.load(QUrl.fromLocalFile("/terrain_map.html"))
        self.json_file = None
        self.mode = "4BS"
        self.unit = "Deg"
        self.unit_switch = None
        self.conv_values = []
        self.bs3_elements = []
        self.lat_labels = []
        self.long_labels = []
        self.height_label = []
        self.conv_labels = []
        self.res_labels = []
        self.user_input = QWidget()
        self.user_info = QWidget()
        self.calculation_input = QWidget()
        self.eval_input = QWidget()
        self.input_fields = []
        self.bs_labels = []
        self.result_labels = []
        self.generated_map = Map()
        self.setup()
        self.on_update_clicked(self.web)

    def setup(self):
        self.user_input.setLayout(self.initUI())
        self.user_info.setLayout(self.initUserInfo())
        self.calculation_input.setLayout(self.initCalculationInput())
        self.eval_input.setLayout(self.initEvalInput())

        self.master_layout.addWidget(self.user_input, 0, 0)
        self.master_layout.addWidget(self.web, 2, 0)
        self.master_layout.addWidget(self.user_info, 2, 1, 2, 1)
        self.master_layout.addWidget(self.calculation_input, 0, 1)
        self.master_layout.addWidget(self.eval_input, 3, 0)

        self.web.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        central_widget = QWidget()
        central_widget.setLayout(self.master_layout)
        self.setCentralWidget(central_widget)

    def initEvalInput(self):
        layout = QGridLayout()
        layout_right = QGridLayout()
        layout_left = QGridLayout()
        widget_right = QWidget()
        widget_left = QWidget()

        # TDOA Std Slider
        tdoa_std_slider = QSlider(Qt.Horizontal)
        tdoa_std_slider.setMinimum(0)
        tdoa_std_slider.setMaximum(10)
        tdoa_std_label = QLabel("0")
        tdoa_std_unit = QLabel(" m")
        tdoa_std_slider.sliderReleased.connect(lambda: self.on_tdoa_std_changed(tdoa_std_slider, tdoa_std_label))
        tdoa_std_slider.valueChanged.connect(lambda: self.on_value_changed(tdoa_std_slider, tdoa_std_label))

        # Baro Std Slider
        baro_std_slider = QSlider(Qt.Horizontal)
        baro_std_slider.setMinimum(0)
        baro_std_slider.setMaximum(10)
        baro_std_label = QLabel("0,25")
        baro_std_unit = QLabel(" x Std")
        baro_std_slider.sliderReleased.connect(lambda: self.on_baro_std_changed(baro_std_slider, baro_std_label))
        baro_std_slider.valueChanged.connect(lambda: self.on_value_changed(baro_std_slider, baro_std_label, True))

        # Iterations Input
        iterations_label = QLabel("Iterations:")
        iterations_input = QLineEdit()
        iterations_input.setText("100")

        # Start button
        start_button = QPushButton("Start")
        start_button.clicked.connect(
            lambda: self.on_eval_clicked(self.web, int(tdoa_std_label.text()), float(baro_std_label.text())*12.48, int(iterations_input.text())))

        # Eval output
        tdoa_deviation = QLabel("-")
        baro_deviation = QLabel("-")
        start_eval_btn = QPushButton("Start")
        start_eval_btn.clicked.connect(
            lambda: self.on_eval_clicked(self.web, int(tdoa_std_label.text()), float(baro_std_label.text())*12.48, int(iterations_input.text())))

        layout_right.addWidget(QLabel("Eval output:"), 0, 3, 1, 2)
        layout_right.addWidget(QLabel("TDOA"), 1, 3)
        layout_right.addWidget(QLabel("Baro"), 1, 4)
        layout_right.addWidget(tdoa_deviation, 2, 3)
        layout_right.addWidget(baro_deviation, 2, 4)
        layout_right.addWidget(start_eval_btn, 3, 4)
        widget_right.setLayout(layout_right)

        # Adding Widgets
        layout_left.addWidget(QLabel("TDOA Std:"), 0, 0)
        layout_left.addWidget(iterations_label, 0, 3)
        layout_left.addWidget(tdoa_std_slider, 1, 0)
        layout_left.addWidget(tdoa_std_label, 1, 1)
        layout_left.addWidget(tdoa_std_unit, 1, 2)
        layout_left.addWidget(iterations_input, 1, 3)
        layout_left.addWidget(QLabel("Baro Std:"), 2, 0)
        layout_left.addWidget(baro_std_slider, 3, 0)
        layout_left.addWidget(baro_std_label, 3, 1)
        layout_left.addWidget(baro_std_unit, 3, 2)
        layout_left.addWidget(start_button, 3, 3)
        widget_left.setLayout(layout_left)

        layout.addWidget(widget_left, 0, 0)

        return layout

    def on_eval_clicked(self, web, tdoa_std, baro_std, n):

        try:
            # do the tdoah evaluation multiple times
            target_list = [[] for _ in range(3)]
            print(f'Test: {baro_std}')


            #print(f'BS: {bs}\nMS: {ms}\nMS_H: {ms_h}')
            tdoa_vals = np.random.normal(0, tdoa_std, n*2)
            baro_vals = np.random.normal(0, baro_std, n)
            for i in range(n):
                self.conv_values = self.convert_input_field(baro_vals[i])
                ms = self.conv_values.pop()
                bs = self.conv_values
                ms_h = float(self.input_fields[len(self.input_fields) - 1].text())
                print(f'Anforderungen:\nbs: {bs}\nms: {ms}\nms_h: {ms_h}\ntdoa: {tdoa_vals}')
                tdoah_solver = Tdoah(bs, ms, ms_h, tdoa_vals[i], tdoa_vals[i+n], baro_vals[i])
                target = tdoah_solver.solve()

                #print(target_list)
                target_list[0].append(target[0][0])
                target_list[1].append(target[1][0])
                target_list[2].append(target[2][0])
                print(f'Target_z: {target[2][0]}')

            #print(f'Target_list: {target_list}\n')
            self.generated_map.show_result(target_list, 'blue', 'markers')
            tdoah_targets = target_list

            # do the 4bs evaluation multiple times
            target_list = [[] for _ in range(3)]
            # bringing the input in the right form for foy
            foy_bs = [[] for _ in range(4)]
            for e in bs:
                foy_bs[0].append(e[0])
                foy_bs[1].append(e[1])
                foy_bs[2].append(e[2])
            for e in ms:
                foy_bs[3].append(e)
            try:
                ms = [ms[0], ms[1], ms[2]]
            except Exception as e:
                print(f'Error: {e}')
            for i in range(n):
                #print(f'Loop: {i}')
                try:
                    foy_solver = Foy(foy_bs, ms, tdoa_vals[i], baro_vals[i])
                    foy_solver.solve()
                except Exception as e:
                    print(f'Error: {e}')
                target_list[0].append(foy_solver.guesses[0].pop())
                target_list[1].append(foy_solver.guesses[1].pop())
                target_list[2].append(foy_solver.guesses[2].pop())

            #print(f'Foy Solver: {target_list}')
            self.generated_map.show_result(target_list, 'green', 'markers')
            foy_targets = target_list
            web.reload()


            # Load Eval window
            if self.eval_window is None:
                print(f'input_fields: {self.input_fields}')
                points = []
                i = 0
                try:
                    for k, field in enumerate(self.input_fields):
                        if i % 3 == 0 and i < len(self.input_fields) - 2:
                            points.append([self.input_fields[k].text(), self.input_fields[k + 1].text(),
                                           self.input_fields[k + 2].text()])

                        i += 1
                except Exception as e:
                    print(f'Error: {e}')
                print(f'POINTS: {points}')
                self.eval_window = EvalWindow(points, foy_targets, tdoah_targets)
                self.eval_window.show()
                self.eval_window.raise_()
                self.eval_window.activateWindow()

        except Exception as e:
            print(f'Error: {e}')

    def on_value_changed(self, slider, label, map=False):
        if map == True:
            slider_vals = ["0", "0.25", "0.5", "0.75", "1", "1.25", "1.5", "1.75", "2", "2.25", "2.5"]
            label.setText(slider_vals[slider.value()])
        else:
            label.setText(str(slider.value()))

    def on_tdoa_std_changed(self, slider, label):
        pass

    def on_baro_std_changed(self, slider, label):
        pass

    def initUI(self):
        layout = QGridLayout()

        # Define the labels for each block
        block_labels = ['BS0:', 'BS1:', 'BS2:', 'BS3:', 'MS:']
        coord_labels = ['Lat', 'Long', 'Alt']

        # iterate basestation labels and create labels and inputs for each basestation
        for i, block_label in enumerate(block_labels):
            # if element part of label BS3:, save element in variable for modeswitch access
            if block_label == 'BS3:':
                bs3_label = QLabel(block_label)
                self.bs3_elements.append(bs3_label)
                layout.addWidget(bs3_label, 0, i * 4)
            else:
                layout.addWidget(QLabel(block_label), 0, i * 4)

            # Add the 'lat:', 'long:', 'alt:' labels
            for j in range(1, 6):
                if j < 4:
                    label = QLabel(f'{coord_labels[j - 1]}:')
                    if block_label == 'BS3:':
                        self.bs3_elements.append(label)
                        self.bs3_elements.append(label)
                        self.bs3_elements.append(label)
                    layout.addWidget(label, j, i * 4)

            # Add the text inputs for 'lat', 'long', 'alt'
            for j in range(1, 4):
                input = QLineEdit()
                layout.addWidget(input, j, i * 4 + 1)
                self.input_fields.append(input)  # Keep track of input fields
                if block_label == 'BS3:':
                    self.bs3_elements.append(input)

        self.load_file('bs_setups/realistic_ilmenau_setup.json')

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
        user_info = QLabel("User Info:")
        layout.addWidget(user_info, 0, 0)
        block_labels = ['BS0:', 'BS1:', 'BS2:', 'BS3:', 'MS:', 'MS_Conv:']
        coord_labels = ['Lat', 'Long', 'Alt']

        self.conv_values = self.convert_input_field()

        # result_label counter
        c = 0
        for i, block_label in enumerate(block_labels):
            # Add the block label
            label = QLabel(block_label)
            if block_label == 'BS3:':
                self.bs3_elements.append(label)

            layout.addWidget(label, i * 4 + 1, 0)
            bs_labels_tmp = []
            # Add the 'x:', 'y:', 'z:' labels
            for j in range(1, 4):
                if block_label == 'BS3:':
                    label = QLabel(f'{chr(120 + j - 1)}:')
                    layout.addWidget(label, i * 4 + j + 1, 0)
                    self.bs3_elements.append(label)
                elif i == len(block_labels) - 1:
                    layout.addWidget(QLabel(coord_labels[j - 1]), i * 4 + j + 1, 0)
                else:
                    layout.addWidget(QLabel(f'{chr(120 + j - 1)}:'), i * 4 + j + 1, 0)
                if i < 4 and not block_label == 'BS3:':
                    label = QLabel(str(self.conv_values[i][j - 1]))
                    bs_labels_tmp.append(label)
                    layout.addWidget(label, i * 4 + j + 1, 1)
                elif i < 4 and block_label == 'BS3:':
                    label = QLabel(str(self.conv_values[i][j - 1]))
                    bs_labels_tmp.append(label)
                    self.bs3_elements.append(label)
                    layout.addWidget(label, i * 4 + j + 1, 1)

                else:
                    self.result_labels.append(QLabel('-'))
                    layout.addWidget(self.result_labels[c], i * 4 + j + 1, 1)
                    c += 1

            self.bs_labels.append(bs_labels_tmp)
        return layout

    def convert_input_field(self, std=0):
        values = []
        conv_values = []
        for i, input in enumerate(self.input_fields):
            if i % 3 == 0 and i < len(self.input_fields) - 2:
                values.append([float(self.input_fields[i].text()), float(self.input_fields[i + 1].text()),
                               float(self.input_fields[i + 2].text())])
        for i, value in enumerate(values):
            #print(f'ms_h_real: {value[2]}')
            conv_values.append(w2k(value[0], value[1], value[2]))
        return conv_values

    def initCalculationInput(self):
        layout = QGridLayout()
        calc_button = QPushButton("Calculate")
        calc_button.clicked.connect(lambda: self.on_calc_clicked(self.web))
        comp_button = QPushButton("Compare")
        comp_button.clicked.connect(lambda: self.on_compare_clicked(self.web))
        # dropdown = QComboBox()
        # dropdown.addItem("Foy")
        mode_switch = QPushButton("Mode: 4BS")
        mode_switch.clicked.connect(lambda: self.on_mode_switch_clicked(mode_switch))
        # layout.addWidget(dropdown, 0, 0)
        layout.addWidget(mode_switch, 1, 0)
        layout.addWidget(calc_button, 3, 0)
        layout.addWidget(comp_button, 4, 0)
        return layout

    def read_json(self, file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data

    def on_compare_clicked(self, web):
        # load json file
        try:
            ms_pos = self.read_json('finland_eval/ms_positions.json')
        except Exception as e:
            print(f"Error: {e}")

        print(ms_pos)

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
                        x, y, z = w2k(float(self.input_fields[i].text()), float(self.input_fields[i + 1].text()),
                                      float(self.input_fields[i + 2].text()))
                        print(f'x: {x}, y: {y}, z: {z}')
                        self.input_fields[i].setText(str(x))
                        self.input_fields[i + 1].setText(str(y))
                        self.input_fields[i + 2].setText(str(z))

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
        # Open file dialog to select JSON file and safe filename as variable
        file_path, _ = QFileDialog.getOpenFileName(None, "Open File", "", "JSON Files (*.json);;All Files (*)")
        if file_path: self.json_file = os.path.basename(file_path)
        self.load_file(file_path)

    def load_file(self, file_path):
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                # print(f'data: {data}')
                for i, input_field in enumerate(self.input_fields):
                    i1 = i // 3
                    i2 = i % 3 + 1
                    # print(f'i1: {i1}, i2: {i2}')
                    # temp.append(data[str(i1)][str(i2)])
                    test = data[f'{i1}'][f'{i2}']
                    # Temporary solution
                    input_field.setText(str(test))
        except FileNotFoundError as e:
            print("File not found.")
        except json.JSONDecodeError:
            print("Invalid JSON format")

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
                points[i] = [0 for _ in range(3)]
            points[num - 1] = [float(self.input_fields[len(self.input_fields) - 3].text()),
                               float(self.input_fields[len(self.input_fields) - 2].text()),
                               float(self.input_fields[len(self.input_fields) - 1].text())]
            self.conv_values = self.convert_input_field()
            for i, input_field in enumerate(self.input_fields):
                if self.mode == "3BS" and i > 8:
                    break
                else:
                    i1 = i // 3
                    i2 = i % 3
                    points[i1][i2] = input_field.text()

                if i1 < 4 and i2 < 3:
                    self.bs_labels[i1][i2].setText(str(self.conv_values[i1][i2]))

            #print(f"Points: {points}")
            if float(self.input_fields[0].text()) > 65.0:
                print(float(self.input_fields[0].text()))
                self.generated_map.update(points, True)
                print("gogogo")
            else:
                self.generated_map.update(points)
            web.reload()
        except Exception as e:
            print(f"Error: {e}")

    def on_calc_clicked(self, web):
        try:
            ms = []
            ms = self.conv_values.pop()
            bs = self.conv_values
            ms_h = float(self.input_fields[len(self.input_fields) - 1].text())
            if self.mode == "4BS":
                # bringing the input in the right form for foy
                foy_bs = [[] for _ in range(4)]
                for e in bs:
                    foy_bs[0].append(e[0])
                    foy_bs[1].append(e[1])
                    foy_bs[2].append(e[2])
                for e in ms:
                    foy_bs[3].append(e)
                ms = [ms[0], ms[1], ms[2]]
                #print(f"Solver input: {foy_bs, ms}")
                solver = Foy(foy_bs, ms)
                solver.solve()
                #print(f"Solver output: {solver.guesses}")


                #print(f"Solver output: {solver.guesses}")
                self.generated_map.show_result(solver.guesses)
                self.print_solution(solver.guesses)
                # Call the calculation function
                # Update the map
                web.reload()
            elif self.mode == "3BS":
                #print(f"Solver input: {bs, ms}")
                solver = Tdoah(bs, ms, ms_h)
                target = solver.solve()
                #print(f'Final: {target}')
                self.generated_map.show_result(target)
                self.print_solution(target)
                web.reload()

        except Exception as e:
            print(f'Error: {e}')

    def print_solution(self, target):
        if self.mode == "4BS":
            x, y, z = target[0].pop(), target[1].pop(), target[2].pop()
        else:
            x, y, z = target[0][0], target[1][0], target[2][0]
        for i in range(4):
            if i < 3:
                self.result_labels[i].setText(str(target[i].pop()))
            else:
                lat, long, h = k2w(x, y, z)
                self.result_labels[i].setText(str(lat))
                self.result_labels[i+1].setText(str(long))
                self.result_labels[i+2].setText(str(h))

    def get_solver_input(self):
        bs = [[], [], []]
        ms = []
        #print(f"Input fields: {self.input_fields}")
        for i, input in enumerate(self.input_fields):
            #print(f'i: {i}, input: {input.text()}')
            if i < len(self.input_fields) - 3:
                if i % 3 == 0:
                    bs[0].append(float(input.text()))
                elif i % 3 == 1:
                    bs[1].append(float(input.text()))
                elif i % 3 == 2:
                    bs[2].append(float(input.text()))
            if i == len(self.input_fields) - 3:
                ms = [float(self.input_fields[i].text()), float(self.input_fields[i + 1].text()),
                      float(self.input_fields[i + 2].text())]
        if self.mode == '3BS':
            for e in bs:
                e.pop()

        return bs, ms

    def calculate_height(self):
        heights = []
        try:
            for i in range(1, len(self.input_fields) + 1):
                if i % 3 == 0 and not i == 0:
                    heights.append(int(self.input_fields[i - 1].text()))

            for i, height in enumerate(heights):
                self.res_labels[i].setText(str(pvlib.atmosphere.pres2alt(height)))
        except Exception as e:
            print(f"Error: {e}")


class EvalWindow(QDialog):
    def __init__(self, input_fields, foy_targets, tdoah_targets):
        self.input_fields = input_fields
        self.foy_targets = foy_targets
        self.tdoah_targets = tdoah_targets
        self.ms = None

        print(f'foy_targets: {foy_targets}\ntdoah_targets: {tdoah_targets}')

        self.map = None
        try:
            super().__init__(None)
            self.web_view = None
            self.input_fields = self.input_fields


            self.init_map()
            self.init_ui()
        except Exception as e:
            print(f'Error: {e}')

    def init_map(self):
        print(f'$input field: {self.input_fields}')
        self.map = Map2(self.input_fields)
        self.map.init_earth()
        self.ms = self.map.get_ms()
        self.ms = w2k(float(self.ms[0]), float(self.ms[1]), float(self.ms[2]))
        self.foy_targets[0] -= self.ms[0]
        self.foy_targets[1] -= self.ms[1]
        self.foy_targets[2] -= self.ms[2]
        self.tdoah_targets[0] -= self.ms[0]
        self.tdoah_targets[1] -= self.ms[1]
        self.tdoah_targets[2] -= self.ms[2]
        self.map.add_trace(go.Scatter3d(x=self.foy_targets[1], y=self.foy_targets[0], z=self.foy_targets[2], name="Foy", mode='markers', marker=dict(color='green')), 'foy')
        self.map.add_trace(go.Scatter3d(x=self.tdoah_targets[1], y=self.tdoah_targets[0], z=self.tdoah_targets[2], name="TDOAH", mode='markers', marker=dict(color='blue')), 'tdoah')
        self.map.show()


    def init_ui(self):
        self.setWindowTitle("Evaluation")
        self.resize(800, 600)

        self.web_view = QWebEngineView(self)
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        interface = QWidget()
        interface.setLayout(self.init_interface())


        # Load HTML file directly into the web view
        with open('coordinate_map.html', 'r') as f:
            html = f.read()
            self.web_view.setHtml(html)

        layout = QVBoxLayout()
        layout.addWidget(interface)
        layout.addWidget(self.web_view)

        self.setLayout(layout)  # Set layout to the QDialog itself

    def init_interface(self):
        layout = QGridLayout()
        update_button = QPushButton("Update")
        update_button.clicked.connect(lambda: self.on_update_clicked())
        layout.addWidget(update_button, 0, 0)

        earth_button = QPushButton("Earth on/off")
        earth_button.clicked.connect(lambda: self.on_earth_clicked())
        layout.addWidget(earth_button, 0, 1)

        bs_button = QPushButton("BS on/off")
        bs_button.clicked.connect(lambda: self.on_bs_clicked())
        layout.addWidget(bs_button, 0, 2)

        ms_button = QPushButton("MS on/off")
        ms_button.clicked.connect(lambda: self.on_ms_clicked())
        layout.addWidget(ms_button, 0, 3)

        return layout

    def on_update_clicked(self):
        # Load HTML file directly into the web view
        with open('coordinate_map.html', 'r') as f:
            html = f.read()
            self.web_view.setHtml(html)

    def on_earth_clicked(self):
        self.map.toggle_earth()
        self.map.show()
        self.on_update_clicked()

    def on_bs_clicked(self):
        self.map.toggle_stations('bs')
        self.map.show()
        self.on_update_clicked()

    def on_ms_clicked(self):
        self.map.toggle_stations('ms')
        self.map.show()
        self.on_update_clicked()



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
