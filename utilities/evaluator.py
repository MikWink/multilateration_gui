import numpy as np
import json
from solver.tdoah import w2k
from solver.tdoah_class import Tdoah
import plotly.graph_objects as go
import os


class Evaluator:
    def __init__(self, dir_path, bs_setup, ms_postion, file_name):
        self.dir_path = dir_path
        self.bs_name = file_name.split('.')[0]
        self.bs_setup = self.extract_bs_pos(bs_setup)
        self.ms_position, self.ms_h = self.extract_ms_pos(ms_postion)
        NPZ = np.load(os.path.join(self.dir_path, "evaluation_data", "NPZ_Data", file_name), allow_pickle=True)
        self.tdoa_0 = NPZ['tdoa_0']
        self.tdoa_1 = NPZ['tdoa_1']

        print(f'bs_setup: {self.bs_setup}')
        print(f'ms_position: {self.ms_position}')
        print(f'ms_h: {self.ms_h}\n')

    def extract_bs_pos(self, file_path):
        bs_data = self.read_json(os.path.join(self.dir_path, "evaluation_data", "station_setups", file_path))
        bs = []
        for key, value in bs_data.items():
            if len(bs) < 3:
                bs.append(w2k(float(value['1']), float(value['2']), float(value['3'])))
        return bs

    def extract_ms_pos(self, file_path):
        ms_data = self.read_json(os.path.join(self.dir_path, "evaluation_data", "station_setups", file_path))
        ms = []
        ms_h = None
        for key, value in ms_data.items():
            if key == self.bs_name:
                ms_h = value['altitude']
                ms.append((w2k(value['latitude'], value['longitude'], value['altitude'])))
        return ms, ms_h

    def read_json(self, file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data

    def eval_tdoah(self):
        x = []
        y = []
        z = []
        print(f'Eval starting...\n{len(self.tdoa_0)}')
        bs_plus_ms = self.bs_setup
        bs_plus_ms.append(self.ms_position[0])
        try:
            for i in range(len(self.tdoa_0)):
                print(f'i: {i}')
                print(f'Anforderungen_unsure:\nbs: {bs_plus_ms}\nms: {self.ms_position}\nms_h: {self.ms_h}\n')
                solver = Tdoah(bs_plus_ms, self.ms_position[0], self.ms_h)
                solution = solver.solve(self.tdoa_0[i], self.tdoa_1[i])
                print(f'solution: {solution}')
                x.append(solution[0][0] - self.ms_position[0][0])
                y.append(solution[1][0] - self.ms_position[0][1])
                z.append(solution[2][0] - self.ms_position[0][2])
        except Exception as e:
            print(f'Eval error: {e}')

        return go.Scatter3d(x=x, y=y, z=z, mode='markers', name='Localization Error', marker=dict(size=1, color='orange', symbol='x'))



