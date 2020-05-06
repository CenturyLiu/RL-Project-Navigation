import torch
import torch.nn as nn
import torch.nn.functional as F



class DuelingDQN(nn.Module):
    def __init__(self, state_size, num_outputs, seed, fc1_units=64, fc2_units=64):
        super(DuelingDQN, self).__init__()
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        self.val1 = nn.Linear(fc2_units, fc2_units)
        self.val2 = nn.Linear(fc2_units, 1)
        
        self.advantage1 = nn.Linear(fc2_units, fc2_units)
        
        self.advantage2 = nn.Linear(fc2_units, num_outputs)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))



        adv = F.relu(self.advantage1(x))
        adv = self.advantage2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()

