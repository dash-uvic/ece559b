######################################################################
# Replay Memory
# -------------
#

import random
import pickle
import replay
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

def load_memory(fn="memory.pkl"):
    with open(fn, 'rb') as fp:
        return pickle.load(fp)

def save_memory(mem, fn="memory.pkl"):
    with open(fn, 'wb') as fp:
        pickle.dump(mem, fp, pickle.HIGHEST_PROTOCOL)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    #dummy function to match prioritized memory api
    def update(self, idx, error):
        pass  
