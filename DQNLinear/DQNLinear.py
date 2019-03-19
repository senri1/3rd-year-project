import numpy as np 
import gym
import torch
import torch.nn.functional as F
from gym import wrappers
from atari_wrappers import wrap_deepmind
from atari_wrappers import make_atari
from LinearAgent import Linearagent
from ReplayMemory import ReplayMemory
from EvaluateAgent import collectRandomData
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

memory = ReplayMemory(memory_size)
agent = Linearagent()
collectRandomData(memory,memory_start_size,env_name)
agent.train_autoencoder(memory, memory_start_size)
#agent.load_agent()
#memory.load_replay()

n = 0
j = agent.training_steps

try:
    while n in range(frames):

        done = False
        initial_state = env.reset()
        action = agent.getAction(np.array(initial_state.__array__()[np.newaxis,:,:,:])) 
        state, reward, done, _ = env.step(action)
        memory.add(initial_state,action,reward,state,done )
        agent.decrease_epsilon()
        n += 1
            
        while (not done) and (n<frames):

            action = agent.getAction(np.array(state.__array__()[np.newaxis,:,:,:])) 
            next_state,reward,done,_ = env.step(action)
            memory.add(state,action,reward,next_state,done)
            state = next_state
            agent.decrease_epsilon()
            n += 1

            if memory.current_size >= batch_size:

                # get batch of size 32 from replay memory
                state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = memory.get_batch(batch_size)
                # get qtargets
                qtarget = agent.getQtargets(next_state_batch, reward_batch, not_done_batch)
                # get data that matches qtarget and states to corresponding action 
                state_batch, qtarget = agent.getTrainingData(state_batch,qtarget,action_batch)
                # train agent for 1 step
                agent.train(state_batch, qtarget, 1)

            if n % update_frequency == 0:
                #start_time = time.monotonic()    
                agent.updateQTarget()
                #end_time = time.monotonic()
                #print('Block 4 time: ',timedelta(seconds=end_time - start_time))
            

            if n % evaluation_frequency == 0:
                agent.save_agent(j)
                print(j)
                j+=1

    episodes += 1
    print(episodes)

except:
    agent.save_agent('BACKUP')
    #memory.save_replay()
    np.save('idx',j)
    print("Total number of frames: ", frames)
    print("Total number of episodes: ", episodes)

agent.save_agent('FINAL')
#memory.save_replay()
np.save('idx',j)
print("Total number of frames: ", frames)
print("Total number of episodes: ", episodes)



"""
done = False
initial_state = env.reset()
action = agent.getAction(np.array(initial_state.__array__()[np.newaxis,:,:,:])) 
state, reward, done, _ = env.step(action)
memory.add(initial_state,action,reward,state,done )
agent.decrease_epsilon()
n += 1

action = agent.getAction(np.array(state.__array__()[np.newaxis,:,:,:])) 
next_state,reward,done,_ = env.step(action)
memory.add(state,action,reward,next_state,done)
state = next_state
agent.decrease_epsilon()
n += 1

# get batch of size 32 from replay memory
state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = memory.get_batch(batch_size)
# get qtargets
qtarget = agent.getQtargets(next_state_batch, reward_batch, not_done_batch)
# get data that matches qtarget and states to corresponding action 
state_batch, qtarget = agent.getTrainingData(state_batch,qtarget,action_batch)
# train agent for 1 step
agent.train(state_batch, qtarget, 1)

if n % update_frequency == 0:
#start_time = time.monotonic()    
agent.updateQTarget()
#end_time = time.monotonic()
#print('Block 4 time: ',timedelta(seconds=end_time - start_time))


if n % evaluation_frequency == 0:
agent.save_agent(j)
print(j)
j+=1
"""
