import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Qnetwork(torch.nn.Module):
    def __init__(self,num_actions):
        super(Qnetwork,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0).cuda()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0).cuda()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0).cuda()
        self.fc1 = nn.Linear(64 * 7 * 7, 512).cuda()
        self.fc2 = nn.Linear(512, num_actions).cuda()
    
    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        # Resize from (batch_size, 64, 7, 7) to (batch_size,64*7*7)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        return self.fc2(out)




class DQNagent():
    
    def __init__(
        self,
        epsilon=1,
        disc_factor = 0.99,
        num_actions=4
        ):
        self.Q = Qnetwork(num_actions)
        self.Q.cuda()
        self.QTarget = self.Q 
        self.disc_factor = disc_factor
        self.epsilon = epsilon
        self.num_actions = num_actions

    
    def getQvalues(self,state):
        return self.Q(state)

    def getAction(self,state):

        if self.Q == None:
            return np.random.randint(0,high=4)

        else:

            Qvalues = self.getQvalues(state)
            probability = np.random.random_sample()

            if self.epsilon <= probability:
                maxq, action = Qvalues.max(1)
            else:
                action = np.random.randint(0,high=4)
            return action

