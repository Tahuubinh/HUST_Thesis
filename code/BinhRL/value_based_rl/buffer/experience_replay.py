import numpy as np
import random
# replay memory for NAFA
class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, s0, a, r, s1, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s0[None, :], a, r, s1[None, :], done))

    def sample(self, batch_size):
        s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        # print(len(s0), len(a), len(r))
        # print(s0[0], a[0], r[0])
        # print()
        # #print(np.concatenate(s0), a, r, np.concatenate(s1), done)
        # print(np.concatenate(s0)[0])
        # assert 2 == 3
        return np.concatenate(s0), a, r, np.concatenate(s1), done

    def size(self):
        return len(self.buffer)