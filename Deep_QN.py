import torch 
import numpy as np 
from Q_Network import Network 
from Replay import ReplayBuffer
import torch.nn.functional as F

class DQN:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 lr = 0.00025, 
                 epsilon = 0.2,
                 epsilon_min = 0.1,
                 gamma = 0.99,
                 batch_size = 1,
                 warmup_steps = 5000,
                 buffer_size = int(1e5),
                 target_update_interval = 1000
                 ):
        self.action_dim = action_dim 
        self.epsilon = epsilon 
        self.gamma = gamma 
        self.batch_size = batch_size 
        self.warmup_steps = warmup_steps 
        self.target_update_interval = target_update_interval

        self.network = Network(state_dim[0], action_dim) 
        self.target_network = Network(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr) 

        self.buffer = ReplayBuffer(state_dim, (1, ), buffer_size) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        
        self.network.to(device = self.device) 
        self.target_network.to(device = self.device) 

        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e6

    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training) 
        if training and (np.random.rand() < self.epsilon or self.total_steps < self.warmup_steps):
            max_index = np.random.randint(0, self.action_dim)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.network(state)
            max_index = torch.argmax(q_values).item()
        return max_index
    
    def learn(self):
        state, action, reward, next_state, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size)) 

        next_q_values = self.target_network(next_state).detach() 
        target_values = reward + (1 - terminated) * self.gamma * next_q_values.max(dim = 1, keepdim = True).values
        loss = F.mse_loss(self.network(state).gather(1, action.long()), target_values)
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()

        result = {
            'Total_steps' : self.total_steps,
            'value_loss' : loss.item()
        }

        return result 
    
    def process(self, transition):
        # print('Replay Buffer updated...')
        result = {} 
        self.total_steps += 1
        self.buffer.update(*transition) 

        if self.total_steps > self.warmup_steps:
            result = self.learn()

        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        self.epsilon -= self.epsilon_decay 
        return result 