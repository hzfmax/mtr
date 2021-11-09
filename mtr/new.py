import numpy as np
from utils.loader import get_EAL
import matplotlib.pyplot as plt
from scipy import stats
import scipy.stats as stats
import random
from utils.loader import get_data

def random_list(start,stop,length):
    if length>=0:
        length=int(length)
    start, stop = ((start), (stop)) if start <= stop else ((stop), (start))
    random_list = []
    for i in range(length):
        random_list.append(random.uniform(start, stop))
    return random_list

# data = get_EAL()

# name = data['name']
# length = data['length']
# routes = data['routes']
# run = data['run']
# distance = data['distance']
# demand = data['demand']

# name = 'EastRailLine(EAL)'
# data['name'] = name

# length = 12
# data['length'] = length

# routes = ['HuH','MKK','KOT','TAM','SHT','FOT','UNI','TAP','TWO','FAN','SHS','LOW']
# data['routes'] = routes

# run = np.array([219, 124, 354, 110, 339, 174, 151, 114, 259, 162, 319, 200, 303, 164, 261, 110, 147, 192, 343, 121, 322, 121, 234])
# data['run'] = run
# # 多加一个200

# distance = np.array([3460, 1530, 6150, 1270, 6340, 2560, 1860, 1400, 4390, 1790, 2410, 1000, 2410, 1790, 4390, 1400, 1860, 2560, 6340, 1270, 6150, 1530, 3460])
# # 多加一个1000
# data['distance'] = distance

# t_start=7
# data['t_start'] = t_start
# t_end = 9
# data['t_end'] = t_end

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
np.save('EAL.npy',data)
# print (data['t_start']) 


