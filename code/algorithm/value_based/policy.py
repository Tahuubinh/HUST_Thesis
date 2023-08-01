from __future__ import division
import numpy as np

from rl.util import *


class Policy(object):
    """Abstract base class for all implemented policies.

    Each policy helps with selection of action to take on an environment.

    Do not use this abstract base class directly but instead use one of the concrete policies implemented.
    To implement your own policy, you have to implement the following methods:

    - `select_action`

    # Arguments
        agent (rl.core.Agent): Agent used
    """
    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        """Return configuration of the policy

        # Returns
            Configuration as dict
        """
        return {}

class EpsGreedyQPolicy(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, eps=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy

        # Returns
            Dict of config
        """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config
    
class EpsLinearDecreaseQPolicy(Policy):
    """Implement the epsilon greedy policy
    Eps Greedy policy either:
    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, maxeps = 1, mineps = 0, subtrahend = 0):
        super(EpsLinearDecreaseQPolicy, self).__init__()
        self.maxeps = maxeps
        self.mineps = mineps
        self.subtrahend = subtrahend
        self.eps = self.maxeps

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        self.eps = max(self.eps - self.subtrahend, self.mineps)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy
        # Returns
            Dict of config
        """
        config = super(EpsLinearDecreaseQPolicy, self).get_config()
        config['maxeps'] = self.maxeps
        config['mineps'] = self.mineps
        config['subtrahend'] = self.subtrahend
        config['eps'] = self.eps
        return config
    
class EpsGreedyHardDecreasedQPolicy(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, eps=.1, decreased_quantity=0,nb_hard_decreased_steps=1):
        super(EpsGreedyHardDecreasedQPolicy, self).__init__()
        self.eps = eps
        self.cnt = 0
        self.decreased_quantity = decreased_quantity
        self.nb_hard_decreased_steps = nb_hard_decreased_steps

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        self.cnt += 1
        if (self.cnt == self.nb_hard_decreased_steps):
            self.cnt = 0
            self.eps = max(0, self.eps - self.decreased_quantity) 
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy

        # Returns
            Dict of config
        """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config
    
class EpsGreedyBoltzmannDecreasedQPolicy(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, eps=.1, decreased_quantity=0,nb_hard_decreased_steps=1, tau=1., clip=(-500., 500.)):
        super(EpsGreedyBoltzmannDecreasedQPolicy, self).__init__()
        self.eps = eps
        self.cnt = 0
        self.decreased_quantity = decreased_quantity
        self.nb_hard_decreased_steps = nb_hard_decreased_steps
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        self.cnt += 1
        if (self.cnt == self.nb_hard_decreased_steps):
            self.cnt = 0
            self.eps = max(0, self.eps - self.decreased_quantity) 
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            assert q_values.ndim == 1
            q_values = q_values.astype('float64')
            nb_actions = q_values.shape[0]

            exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
            probs = exp_values / np.sum(exp_values)
            action = np.random.choice(range(nb_actions), p=probs)
            if self.cnt > self.nb_hard_decreased_steps - 50:
                print(q_values)
                print(probs)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy

        # Returns
            Dict of config
        """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class GreedyQPolicy(Policy):
    """Implement the greedy policy
    Greedy policy returns the current best action according to q_values
    """
    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action


class BoltzmannQPolicy(Policy):
    """Implement the Boltzmann Q Policy
    Boltzmann Q Policy builds a probability law on q values and returns
    an action selected randomly according to this law.
    """
    def __init__(self, tau=1., clip=(-500., 500.)):
        super(BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        """Return configurations of BoltzmannQPolicy
        # Returns
            Dict of config
        """
        config = super(BoltzmannQPolicy, self).get_config()
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config
