import numpy as np
from utils.loader import get_EAL_data
import matplotlib.pyplot as plt
from scipy import stats
import scipy.stats as stats
import random
from utils.loader import get_victoria_data

def random_list(start,stop,length):
    if length>=0:
        length=int(length)
    start, stop = ((start), (stop)) if start <= stop else ((stop), (start))
    random_list = []
    for i in range(length):
        random_list.append(random.uniform(start, stop))
    return random_list

data = get_EAL_data()

name = data['name']
length = data['length']
routes = data['routes']
run = data['run']
distance = data['distance']
demand = data['demand']

name = 'EastRailLine(EAL)'
data['name'] = name

length = 12
data['length'] = length

routes = ['HuH','MKK','KOT','TAM','SHT','FOT','UNI','TAP','TWO','FAN','SHS','LOW']
data['routes'] = routes

run = np.array([219, 124, 354, 110, 339, 174, 151, 114, 259, 162, 319, 200, 303, 164, 261, 110, 147, 192, 343, 121, 322, 121, 234])
data['run'] = run
# 多加一个200

distance = np.array([3460, 1530, 6150, 1270, 6340, 2560, 1860, 1400, 4390, 1790, 2410, 1000, 2410, 1790, 4390, 1400, 1860, 2560, 6340, 1270, 6150, 1530, 3460])
# 多加一个1000
data['distance'] = distance

np.save('/home/hzfmax/DRL_TRB/data/EAL.npy',data)


