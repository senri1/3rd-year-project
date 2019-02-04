from agent import myAgent
from VAE2 import *

samples = 2000
episodic_rewards = np.zeros((20000,1))
total_ep = 0              
avrg_reward = np.zeros((50,1))

agent = myAgent()
agent.epsilon = 0
agent.load_encoder()
agent.load_policy()


observation,actions,rewards,num_episodes,ep_reward = collectObs(samples,4,'Breakout-v0',agent) # 1) Get data      

avrg_reward[0,0] = np.mean(ep_reward[0:num_episodes,0])

print('Iteration: ',0)
print('Number of episodes this iteration: ',num_episodes)
print('Average reward per episode: ', np.mean(ep_reward[0:num_episodes,0]))
print('Standard deviation of rewards: ', np.std(ep_reward[0:num_episodes,0]))
print('Current epsilon: ',agent.epsilon)