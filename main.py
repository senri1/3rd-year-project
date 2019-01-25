from agent import myAgent
from VAE2 import*
import matplotlib.pyplot as plt

samples = 1000                          #roughly 13 episodes
encoder_samples = 100000
iterations = 150                        #150 iterations totals roughly 1950 episodes
episodic_rewards = np.zeros((3000,1))
total_ep = 0

agent = myAgent()



agent.create_encoder(encoder_samples)
#agent.load_encoder()

for n in range(iterations):
    
    agent.create_encoder(encoder_samples)       #Create and train encoder
    
    observation,actions,rewards,num_episodes,ep_reward = collectObs(samples,4,'Breakout-v0',agent)      
    episodic_rewards[total_ep:total_ep+num_episodes,1] = ep_reward
    total_ep += num_episodes
    

    states = agent.getState(observation,samples)
    
    agent.improve_policy(states,actions[4:,:],rewards[4:,:])

    if agent.epsilon>0.1:
        agent.epsilon -= 0.009

plt.plot(episodic_rewards)
ply.ylabel('Reward per episode')
plt.xlabel('Episodes')
plt.show()
                                   
