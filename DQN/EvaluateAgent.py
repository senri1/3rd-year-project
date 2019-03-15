from ReplayMemory import *
from DQNAgent import DQNagent
import os

TestAgent = DQNagent()
evaluation_data=[]
idx = int(np.load('idx.npy'))
env_name = 'Breakout-v0'

for i in range(idx):

    print(i)
    state_dict = torch.load('saved_models/Qnetwork' + str(i) + '.pth')
    TestAgent.Q.load_state_dict(state_dict)
    evaluation_data.append([ collectMeanScore(TestAgent,7,0.05,env_name), evaluateStateQvalues(TestAgent) ])

np.savetxt("log/eval_data.csv", evaluation_data, delimiter=",")