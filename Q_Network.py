from torch import nn 
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Network, self).__init__() 
        self.conv_1 = nn.Conv2d(state_dim, 16, kernel_size = 8, stride = 4)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)
        self.in_features = 32 * 14 * 14
        self.fc_layers = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, action_dim) 
        )

    def forward(self, input):
        input = F.relu(self.conv_1(input)) 
        input = F.relu(self.conv_2(input)) 
        input = input.view((-1, self.in_features)) 
        input = self.fc_layers(input)  
        return input 