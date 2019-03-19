import numpy as np 
import gym
import torch
import torch.nn.functional as F
from gym import wrappers
from atari_wrappers import wrap_deepmind
from atari_wrappers import make_atari
from DQNAgent import DQNagent
from ReplayMemory import *
import time
from datetime import timedelta

env_name = 'Breakout-v0'
env = make_atari(env_name)
env = wrap_deepmind(env)
"""
frames - number of frames for algorithm to run on
episodes - stores number of episodes
batch_size - size of batch to train q network with, as well as how many data points to sample
memory_size - size of experience replay memory
memory_start_size - size of initial random memory in experience replay memory
learning_rate - learning rate for SGD
update_frequency - how frequently to update target q network
discount - discount factor 
evaluation_frequency - how often to evaluate the agents performance
evaluation_data - list of evaluation data 

"""
frames = 1000
episodes = 0
batch_size = 32
memory_size = 500000
memory_start_size = int(memory_size/1000)
learning_rate = 0.00025
update_frequency = 10000
evaluation_frequency = frames/250

memory = ReplayMemory(memory_size, batch_size)
agent = DQNagent()
#optimizer = torch.optim.adam(agent.Qnetwork.parameters(), lr = learning_rate)
optimizer = torch.optim.RMSprop(agent.Q.parameters(), lr=learning_rate, eps=0.01, alpha=0.95)
collectRandomData(memory,memory_start_size,env_name)
print(memory.current_size)

n = 0
j = 0

try:
    while n in range(frames):

        done = False
        initial_state = env.reset()
        action = agent.getAction(LazyFrame2Torch(initial_state)) 
        state, reward, done, _ = env.step(action)
        memory.add(initial_state,action,reward,state,done )
        agent.decrease_epsilon(n)
        n += 1
            
        while (not done) and (n<frames):

            action = agent.getAction(LazyFrame2Torch(state)) 
            next_state,reward,done,_ = env.step(action)
            memory.add(state,action,reward,next_state,done)
            state = next_state
            agent.decrease_epsilon(n)
            n += 1

            if memory.current_size >= batch_size:

                # Get batch
                state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = memory.get_batch()

                # Zero any graidents
                optimizer.zero_grad()
                    
                # Get the q values corresponding to action taken
                qvalues = agent.Q(state_batch)[range(batch_size), action_batch]

                # Get the target q values 
                qtargetValues, _ = torch.max(agent.QTarget(next_state_batch), 1)

                # set final frame in episode to have q value equal to reward
                qtargetValues = not_done_batch * qtargetValues

                # calculate target q value r + y * Qt
                qtarget = reward_batch + agent.disc_factor * qtargetValues

                # don't calculate gradients of target network
                qtarget = qtarget.detach()

                # loss is mean squared loss 
                loss = F.mse_loss(qvalues,qtarget)

                # calculate gradients of q network parameters
                loss.backward()

                # update paramters a single step
                optimizer.step()

            if n % update_frequency == 0:
                #start_time = time.monotonic()    
                agent.QTarget.load_state_dict(agent.Q.state_dict())
                #end_time = time.monotonic()
                #print('Block 4 time: ',timedelta(seconds=end_time - start_time))
            

            if n % evaluation_frequency == 0:
                torch.save(agent.Q.state_dict(),'saved_models/Qnetwork' + str(j) + '.pth')
                print(j)
                j+=1

    
            
    episodes += 1
    print(episodes)

except:
    torch.save(agent.Q.state_dict(),'saved_models/QnetworkBACKUP.pth')
    np.save('log/idx',j)
    print('Final avergae score: ', collectMeanScore(agent,5,0.005,env_name))
    print("Total number of frames: ", frames)
    print("Total number of episodes: ", episodes)

torch.save(agent.Q.state_dict(),'saved_models/QnetworkFINAL.pth')
np.save('idx',j)
print('Final avergae score: ', collectMeanScore(agent,5,0.005,env_name))
print("Total number of frames: ", frames)
print("Total number of episodes: ", episodes)