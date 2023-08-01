import random as rd
import numpy as np
from pathlib import Path
import os
from config import DATA_LOCATION, NUM_TASKS_PER_TIME_SLOT
path = os.path.abspath(__file__)
path = Path(path).parent.parent.parent

for NUM_TASKS_PER_TIME_SLOT in [800, 1400, 2000, 2600, 3200, 3800]:
    try:
        os.makedirs('../data_task/data' + str(NUM_TASKS_PER_TIME_SLOT))
    except OSError as e:
        print(e)

    DATA_LOCATION = "data_task/data" + str(NUM_TASKS_PER_TIME_SLOT) + "/"
    for i in range(200):
        with open("{}/{}/datatask{}.csv".format(str(path), DATA_LOCATION, i), "w") as output:

            indexs = NUM_TASKS_PER_TIME_SLOT
            seconds = 30

            m = np.sort(np.random.randint(
                i*100*seconds, (i+1)*100*seconds, indexs)/100)
            # Computational resource Gigacycle
            m1 = np.random.randint(500, 600, indexs)/1000
            m2 = np.random.randint(1500, 2000, indexs)/1000  # p in Mb
            m3 = np.random.randint(15, 20, indexs)/1000  # p out Mb
            m4 = 1+np.random.rand(indexs)/2  # deadline
            
            for j in range(indexs):
                output.write("{},{},{},{},{}\n".format(
                    m[j], m1[j], m2[j], m3[j], m4[j]))
        # import pdb;pdb.set_trace()
