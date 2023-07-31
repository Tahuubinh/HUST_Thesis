import random
import numpy as np


class EpsGreedyQPolicy:
    def __init__(self, agent, eps=.1):
        self.agent
        self.eps = eps

    def selectAction(self, state):
        if random.random() > self.eps:
            self.agent.exploit(state)
        else:
            actions = self.agent.env.getPossibleActionGivenState(state)
            action = np.random.choice(list(actions))
        return action
