import random
import numpy as np
import os
import pickle

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        #Kluge fix because of time constraints
        if action.shape == (1,4):
            action = action.squeeze(0)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        try:
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
        except Exception as e:
            pickle.dump(batch, file = open("batch.pickle", "wb"))
            #batch = pickle.load(open("batch.pickle", "rb"))
            raise e
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, save_path):
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                self.buffer = pickle.load(f)
                self.position = len(self.buffer) % self.capacity

if __name__ == "__main__":
    fn = "sac_buffer_boxfinder_draw.pkl"
    buff = ReplayMemory(1000000, 123456)
    buff.load_buffer(fn)
    buff.save_buffer(fn)

