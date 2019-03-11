from agent import myAgent
from helperFuncs import *

samples = 2000
iterations = 40                       
episodic_rewards = np.zeros((20000,1))
total_ep = 0
drate =  0.93              
avrg_reward = np.zeros((iterations,1))
data = []
agent = myAgent()

#agent.load_policy()
agent.create_CNN()

for n in range(iterations):
    
