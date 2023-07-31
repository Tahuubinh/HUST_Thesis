from data_util import read_energy_data
import numpy as np
with open('../data/SoDa_HC3-METEO_lat0.329_lon32.499_2005-01-01_2005-12-31_1833724734.csv', 'r') as f:
    readlines = f.readlines()
    result = list(readlines)[32:]

GHI = np.zeros(len(result))
for i in range(len(GHI)):
    # Only get Clear-Sky 
    GHI[i] = int(result[i].split(";")[2])
print(GHI.shape)
for i in range(1000):
    print(GHI[i])