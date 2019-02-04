from agent import myAgent
from VAE2 import*
import matplotlib.pyplot as plt
import pickle
import numpy as np

samples = 2000                         #roughly 65 episodes
encoder_samples = 100000
iterations = 100                       
episodic_rewards = np.zeros((20000,1))
total_ep = 0
drate =  0.93              
avrg_reward = np.zeros((iterations,1))

agent = myAgent()



#agent.create_encoder(encoder_samples)
#agent.load_policy()
agent.load_encoder()

for n in range(iterations):
    
    try:
        observation,actions,rewards,num_episodes,ep_reward = collectObs(samples,4,'Breakout-v0',agent) # 1) Get data      
        episodic_rewards[total_ep:total_ep+num_episodes,0] = ep_reward[0:num_episodes,0]
        total_ep += num_episodes
        avrg_reward[n,0] = np.mean(ep_reward[0:num_episodes,0])
    
        print('Iteration: ',n)
        print('Number of episodes this iteration: ',num_episodes)
        print('Average reward per episode: ', np.mean(ep_reward[0:num_episodes,0]))
        print('Standard deviation of rewards: ', np.std(ep_reward[0:num_episodes,0]))
        print('Current epsilon: ',agent.epsilon)

        states = agent.getState(observation,samples) # 2) Get state from the observations
    
        agent.improve_policy(states,actions[4:,:],rewards[4:,:]) # 3) Improve policy using collected data

        if agent.epsilon>0.01:         # 4) Decrease chance of taking random action
                agent.epsilon *= drate
        else:
                agent.epsilon = 0.01
    
    except:
           agent.save_policy()
           np.save('observation',observation)
           np.save('actions',actions)
           np.save('rewards',rewards)
           np.save('num_episodes',num_episodes)
           np.save('ep_reward',ep_reward)


agent.save_policy()
print("Total number of episodes: ",total_ep)
plt.scatter(np.arange(1,iterations+1),avrg_reward)
plt.ylabel('Average reward per episode')
plt.xlabel('Iteration')
plt.show()

#fig = plt.figure(figsize=(8, 8))
fig1 = plt.figure(figsize=(8, 8))
ax = []
ax1 = []

plot_img = np.random.randint(low=0,high = observation.shape[0]-4,size=18)
columns = 6
rows = 6       
               
"""for j in range(1,int(columns*rows/2) ):
        ax.append ( fig.add_subplot(rows, columns, j) )
        ax[-1].set_title("Observation: "+str(j))
        plt.imshow(observation[plot_img[j-1],:,:,0])

plt.show()
"""

for j in range(1,int(columns*rows/2) ):
        ax1.append(fig1.add_subplot(rows, columns, j))
        ax1[-1].set_title("Reconstruction: "+str(j))
        plt.imshow( agent.CAE.predict( Img2Frame( observation[plot_img[j-1]:plot_img[j-1]+4,:,:,:]))[0,:,:,0] )

plt.show()