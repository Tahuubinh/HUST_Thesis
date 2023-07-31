import random as rd
import numpy as np
from pathlib import Path
import os
from config import DATA_LOCATION, NUM_TASKS_PER_TIME_SLOT
path = os.path.abspath(__file__)
path = Path(path).parent.parent

# try:
#     os.makedirs('../data_task/data' + str(NUM_TASKS_PER_TIME_SLOT))
# except OSError as e:
#     print(e)

# for i in range(200):
#     with open("{}/{}/datatask{}.csv".format(str(path),DATA_LOCATION,i),"w") as output:

#         indexs=NUM_TASKS_PER_TIME_SLOT
#         seconds = 30
#         # To make the diff as 0.1s
#         # m = np.sort(np.random.randint(i*10*seconds,(i+1)*10*seconds,indexs)/10)
#         # m1 = np.random.randint(500,1000,indexs)/1000 #Computational resource Gigacycle
#         # m2 = np.random.randint(1000,2000,indexs)/1000 # p in Mb
#         # m3 = np.random.randint(100,200,indexs)/1000 # p out Mb
#         # m4 = 0.5+np.random.rand(indexs) #deadline

#         # indexs=NUM_TASKS_PER_TIME_SLOT
#         # seconds = 10
#         # # To make the diff as 0.1s
#         # m = np.sort(np.random.randint(i*100*seconds,(i+1)*100*seconds,indexs)/100)
#         # m1 = np.random.randint(200,300,indexs)/1000 #Computational resource Gigacycle
#         # m2 = np.random.randint(150,200,indexs)/1000 # p in Mb
#         # m3 = np.random.randint(15,20,indexs)/1000 # p out Mb
#         # m4 = 0.05+np.random.rand(indexs)/20 #deadline

#         m = np.sort(np.random.randint(i*100*seconds,(i+1)*100*seconds,indexs)/100)
#         m1 = np.random.randint(500,600,indexs)/1000 #Computational resource Gigacycle
#         m2 = np.random.randint(1500,2000,indexs)/1000 # p in Mb
#         m3 = np.random.randint(15,20,indexs)/1000 # p out Mb
#         m4 = 1+np.random.rand(indexs)/2 #deadline


#        # indexs=rd.randint(1000,1000)
#        # m = np.sort(np.random.randint(i*300,(i+1)*300,indexs))
#        # m1 = np.random.randint(1000,1200,indexs)
#        # m2 = np.random.randint(100,110,indexs)
#        # m3 = np.random.randint(2,3,indexs)
#        # m4 = 5+np.random.rand(indexs)*2

#         for j in range(indexs):
#             output.write("{},{},{},{},{}\n".format(m[j],m1[j],m2[j],m3[j],m4[j]))
#     #import pdb;pdb.set_trace()
for NUM_TASKS_PER_TIME_SLOT in [2000]:
    try:
        os.makedirs('../data_task/data' + str(NUM_TASKS_PER_TIME_SLOT))
    except OSError as e:
        print(e)

    DATA_LOCATION = "data_task/data" + str(NUM_TASKS_PER_TIME_SLOT) + "/"
    for i in range(200):
        with open("{}/{}/datatask{}.csv".format(str(path), DATA_LOCATION, i), "w") as output:

            indexs = NUM_TASKS_PER_TIME_SLOT
            seconds = 30
            # To make the diff as 0.1s
            # m = np.sort(np.random.randint(i*10*seconds,(i+1)*10*seconds,indexs)/10)
            # m1 = np.random.randint(500,1000,indexs)/1000 #Computational resource Gigacycle
            # m2 = np.random.randint(1000,2000,indexs)/1000 # p in Mb
            # m3 = np.random.randint(100,200,indexs)/1000 # p out Mb
            # m4 = 0.5+np.random.rand(indexs) #deadline

            # indexs=NUM_TASKS_PER_TIME_SLOT
            # seconds = 10
            # # To make the diff as 0.1s
            # m = np.sort(np.random.randint(i*100*seconds,(i+1)*100*seconds,indexs)/100)
            # m1 = np.random.randint(200,300,indexs)/1000 #Computational resource Gigacycle
            # m2 = np.random.randint(150,200,indexs)/1000 # p in Mb
            # m3 = np.random.randint(15,20,indexs)/1000 # p out Mb
            # m4 = 0.05+np.random.rand(indexs)/20 #deadline

            m = np.sort(np.random.randint(
                i*100*seconds, (i+1)*100*seconds, indexs)/100)
            # Computational resource Gigacycle
            m1 = np.random.randint(500, 600, indexs)/1000
            m2 = np.random.randint(1500, 2000, indexs)/1000  # p in Mb
            m3 = np.random.randint(15, 20, indexs)/1000  # p out Mb
            m4 = 1+np.random.rand(indexs)/2  # deadline

           # indexs=rd.randint(1000,1000)
           # m = np.sort(np.random.randint(i*300,(i+1)*300,indexs))
           # m1 = np.random.randint(1000,1200,indexs)
           # m2 = np.random.randint(100,110,indexs)
           # m3 = np.random.randint(2,3,indexs)
           # m4 = 5+np.random.rand(indexs)*2

            for j in range(indexs):
                output.write("{},{},{},{},{}\n".format(
                    m[j], m1[j], m2[j], m3[j], m4[j]))
        # import pdb;pdb.set_trace()
