import torch 
import numpy as np 

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size = int(1e5)):
        self.state = np.zeros((max_size, *state_dim), dtype = np.float32) 
        self.action = np.zeros((max_size, *action_dim), dtype = np.int64) 
        self.reward = np.zeros((max_size, 1), dtype = np.float32) 
        self.next_state = np.zeros((max_size, *state_dim), dtype = np.float32) 
        self.terminated = np.zeros((max_size, 1), dtype = np.float32) 

        self.pointer = 0 
        self.size = 0
        self.max_size = max_size 

    def update(self, state, action, reward, next_state, terminated):
        self.state[self.pointer] = state 
        self.action[self.action] = action 
        self.reward[self.pointer] = reward 
        self.next_state[self.pointer] = next_state 
        self.terminated[self.pointer] = terminated

        self.pointer = (self.pointer + 1) % self.max_size 
        self.size = min(self.size + 1, self.max_size) 

    def sample(self, batch_size):
        random_index = np.random.randint(0, self.size, batch_size) 

        return (
            torch.FloatTensor(self.state[random_index]),
            torch.FloatTensor(self.action[random_index]), 
            torch.FloatTensor(self.reward[random_index]),
            torch.FloatTensor(self.next_state[random_index]),
            torch.FloatTensor(self.terminated[random_index])
        )