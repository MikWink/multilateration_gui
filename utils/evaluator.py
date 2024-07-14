import numpy as np
import json
from solver.tdoah import w2k


class Evaluator:
    def __init__(self, bs_setup, ms_postion, file_name):
        self.bs_name = file_name.split('.')[0]
        self.bs_setup = self.extract_bs_pos(bs_setup)
        self.ms_position, self.ms_h = self.extract_ms_pos(ms_postion)
        NPZ = np.load('../evaluation_data/NPZ_Data/' + file_name, allow_pickle=True)
        self.tdoa_0 = NPZ['tdoa_0']
        self.tdoa_1 = NPZ['tdoa_1']

        print(f'bs_setup: {self.bs_setup}')
        print(f'ms_position: {self.ms_position}')
        print(f'ms_h: {self.ms_h}\n')

    def extract_bs_pos(self, file_path):
        bs_data = self.read_json('../evaluation_data/station_setups/' + file_path)
        bs = []
        for key, value in bs_data.items():
            if len(bs) < 3:
                bs.append(w2k(float(value['1']), float(value['2']), float(value['3'])))
        return bs

    def extract_ms_pos(self, file_path):
        ms_data = self.read_json('../evaluation_data/station_setups/' + file_path)
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


evaluator = Evaluator('finland_real_setup.json', 'ms_positions.json', '70-b3-d5-67-70-ff-03-40.npz')
