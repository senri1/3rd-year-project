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
        self.QTarget = Qnetwork(num_actions).cuda()
        self.QTarget.load_state_dict(self.Q.state_dict())
        self.disc_factor = disc_factor
        self.epsilon = epsilon
        self.num_actions = num_actions

    
    def getQvalues(self,state):
        with torch.no_grad():
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

    def decrease_epsilon(self,steps):
        min_epsilon = 0.1
        init_epsilon = 1.0
        eps_decay_steps = 1000000.0
        self.epsilon = max(min_epsilon, init_epsilon - (init_epsilon-min_epsilon) * float(steps)/eps_decay_steps)



"""
done = False
initial_state = env.reset()
action = agent.getAction(LazyFrame2Torch(initial_state)) 
state, reward, done, _ = env.step(action)
memory.add(initial_state,action,reward,state,done )

action = agent.getAction(LazyFrame2Torch(state)) 
next_state,reward,done,_ = env.step(action)
memory.add(state,action,reward,next_state,done)
state = next_state
"""