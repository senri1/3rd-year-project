from agent import myAgent
from helperFuncs import*
import matplotlib.pyplot as plt
import pickle
import numpy as np

# samples - The amount of data used each iteration to improve the policy.
#           One sample is 4 frames of an observation from the game. 

# encoder_samples - The number of samples to train the encoder. One sample
#                   4 frames from the game. 
# 
# drate - The rate at which we decay the chance of taking random action.
#         
samples = 2000
encoder_samples = 100000
iterations = 40                       
episodic_rewards = np.zeros((20000,1))
total_ep = 0
drate =  0.93              
avrg_reward = np.zeros((iterations,1))
data = []
agent = myAgent()

# Uncomment and comment as required.
#agent.create_encoder(encoder_samples)
#agent.load_policy()
agent.load_encoder()

for n in range(iterations):

    # Try taining the agent, if theres an error save the policy and data.
    try:
        
        # 1) Collect training data to use to train the agent. If agent has no policy, random
        #    actions will be used to collect the data. 
        observation,actions,rewards,num_episodes,ep_reward = collectObs(samples,4,'Breakout-v0',agent)     
        episodic_rewards[total_ep:total_ep+num_episodes,0] = ep_reward[0:num_episodes,0]
        total_ep += num_episodes
        avrg_reward[n,0] = np.mean(ep_reward[0:num_episodes,0])
        
        # 2) Get states that will be used to train the policy.
        #    observation            -> encoder(observation)   -> flatten(encoder(observation))    
        #    (samples*4, 84, 84, 1) -> (samples*4, 16, 16, 5) -> (samples*4-4, 400)
        #    After getting the states, get the training data.
        states = agent.getState(observation) 
        X,Y = agent.getTrainingData(states,actions[4:,:],rewards[4:,:])
        data.append([X[0], agent.myPolicy[0].predict(X[0]), Y[0],agent.myPolicy[0].getSquaredError(X[0],Y[0])])

        # Print useful information
        print('Iteration: ',n)
        print('Number of episodes this iteration: ',num_episodes)
        print('Average reward per episode: ', np.mean(ep_reward[0:num_episodes,0]))
        print('Standard deviation of rewards: ', np.std(ep_reward[0:num_episodes,0]))
        print('Current epsilon: ',agent.epsilon)
        for i in range(4):
                weight = agent.myPolicy[i].getWeights()
                print('Sum of state vectors: ',np.sum(states[i,:]))
                print('Sum of current weights: ', np.sum(weight) )
                #print('Squared error: ', agent.myPolicy[i].getSquaredError(X[i],Y[i]))

        # 3) Improve policy
        agent.improve_policy(X,Y) 

        # 4) Decrease epsilon untill it is 0.01. Porbability of random action decreases
        #    down to 0.01.
        if agent.epsilon>0.01:         
                agent.epsilon *= drate
        else:
                agent.epsilon = 0.01
    
    except:
           agent.save_policy()
           np.save(os.getcwd() + '/debug/observation',observation)
           np.save(os.getcwd() + '/debug/states', states)
           np.save(os.getcwd() + '/debug/actions',actions)
           np.save(os.getcwd() + '/debug/rewards',rewards)
           np.save(os.getcwd() + '/debug/num_episodes',num_episodes)
           np.save(os.getcwd() + '/debug/ep_reward',ep_reward)

# Save policy, print useful info and plot average reward per episode vs iterations
agent.save_policy()
print("Total number of episodes: ",total_ep)
plt.scatter(np.arange(1,iterations+1),avrg_reward)
plt.ylabel('Average reward per episode')
plt.xlabel('Iteration')
plt.show()

with open(os.getcwd() +'/debug/data.pckl', "wb") as f:
        pickle.dump(data, f)




""" IGNOREEEEEEEEE
fig = plt.figure(figsize=(8, 8))
fig1 = plt.figure(figsize=(8, 8))
ax = []
ax1 = []

plot_img = np.random.randint(low=0,high = observation.shape[0]-4,size=18)
columns = 6
rows = 6       
               
for j in range(1,int(columns*rows/2) ):
        ax.append ( fig.add_subplot(rows, columns, j) )
        ax[-1].set_title("Observation: "+str(j))
        plt.imshow(observation[plot_img[j-1],:,:,0])

plt.show()


for j in range(1,int(columns*rows/2) ):
        ax1.append(fig1.add_subplot(rows, columns, j))
        ax1[-1].set_title("Reconstruction: "+str(j))
        plt.imshow( agent.CAE.predict( Img2Frame( observation[plot_img[j-1]:plot_img[j-1]+4,:,:,:]))[0,:,:,0] )

plt.show() 

plt.scatter(np.arange(Y[0].size),Y[0],color = 'red')
plt.scatter(np.arange(Y[0].size),agent.myPolicy[0].predict(X[0]),color='blue')
plt.scatter(np.arange(Y[0].size),(agent.myPolicy[0].predict(X[0])-Y[0]),color='green')
plt.show()

np.save(os.getcwd() + '/debug/X'+str(n),X[0])
np.save(os.getcwd() + '/debug/Y'+str(n),Y[0])
np.save(os.getcwd() + '/debug/Xw'+str(n),agent.myPolicy[0].predict(X[0]))
"""