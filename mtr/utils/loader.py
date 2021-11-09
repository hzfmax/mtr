import os.path as osp

import numpy as np
from numpy.lib.npyio import save

ROOT = osp.abspath(osp.dirname(osp.dirname(__file__)))
DATA_DIR = osp.join(ROOT, 'data')
DEFAULT_LOG_DIR = osp.join(ROOT, 'log')


def get_data(line_name='Victoria', read=True):
    filepath = osp.join(DATA_DIR, f'{line_name}.npy')
    data = np.load(filepath, allow_pickle=True).item()
    return data


def save_data(data, name):
    filepath = osp.join(DATA_DIR, f'{name}.npy')
    np.save(filepath, data)


def get_EAL(read=True, store=False):
    "Generate EAL dataset following 1060C file"

    fp = osp.join(DATA_DIR, "EAL.npy")
    if read and osp.exists(fp):
        return get_data("EAL")
    else:
        data = get_data("Victoria")
        data['name'] = "EAL"
        data['routes'] = ['HUH', 'MKK', 'KOT', 'TAW', 'SHT', 'FOT', 'UNI', 'TAP', 'TWO', 'FAN', 'SHS', 'LOW']
        data['run'] = np.array([
            303, 164, 261, 110, 147, 192, 343, 121, 322, 121, 234, 219, 124, 354, 110, 339, 174, 151, 114, 259, 162, 319
        ])  # turnaround time not included
        d = [2.41, 1.79, 4.39, 1.40, 1.86, 2.56, 6.34, 1.27, 6.15, 1.53, 3.46]
        data['distance'] = np.array(d + d[::-1]) * 1000
        data['dwell'] = np.array([
            0, 30, 47, 30, 30, 30, 30, 35, 25, 30, 40, 0,
            0, 47, 35, 35, 40, 30, 35, 40, 47, 47, 37, 0,
        ])  # terminal station included
        data['t_start'] = 7
        data['t_end'] = 8
        data['turnaround'] = np.array([315, 208])  # [LOW, HUH]
        data['train_weight'] = 45  # metric tones
        data['psg_weight'] = 0.08  # metric tones
        data['capacity'] = 250  # psx
        data['stock_size'] = 31
        data['sft_hdw'] = 60  # seconds, used to be 250 meters, derived with a third of average speed 4.167m/s
        data['dft_hdw'] = 210  # seconds
        num_stat = len(data['routes'])
        data['demand'] = data['demand'][:num_stat, :num_stat]
        if store:
            save_data(data, "EAL")
        return data


if __name__ == '__main__':
    data = get_EAL(False, True)
    for k, v in data.items():
        print(k, type(v))
