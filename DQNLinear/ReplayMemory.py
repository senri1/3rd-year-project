from collections import deque
import random
import numpy as np
import torch 
import torch.nn.functional as F
import gym
from gym import wrappers
from atari_wrappers import wrap_deepmind
from atari_wrappers import make_atari
from DQNAgent import DQNagent


class ReplayMemory():
    def __init__(self, size):
        self.max_size = size
        self.current_size = 0
        self.data = deque(maxlen=size)
    
    def add(self, state, action, reward, next_state, done):
        if self.current_size < self.max_size:
            self.data.append((state, int(action), reward, next_state, not done))
            self.current_size += 1
        else: 
            self.data.popleft()
            self.data.append((state, int(action), reward, next_state, not done))
    
    def get_batch(self,batch_size):
        batch = random.sample(self.data, batch_size)
        batch = np.array(batch)
        statenp = np.zeros((batch_size,4,84,84),dtype = 'uint8')
        next_statenp = np.zeros((batch_size,4,84,84), dtype = 'uint8')

        for n in range(batch_size):
            statenp[n,:,:,:] = batch[n,0].__array__()
            next_statenp[n,:,:,:] = batch[n,3].__array__()

        state = statenp
        action = batch[:,1]
        reward = batch[:,2]
        next_state = next_statenp
        not_done = batch[:,4]
        
        return state, action, reward, next_state, not_done

def collectRandomData(replayMemory,steps,env_name):
    env = make_atari(env_name)
    env = wrap_deepmind(env)
    i=0
    while i in range(steps):
        
        done = False
        initial_state = env.reset()
        action = np.random.randint(0,high=4) 
        state, reward, done, _ = env.step(action)
        replayMemory.add(initial_state,action,reward,state,done )
        i += 1
    
        while (not done) and (i < steps) :
        
            action = np.random.randint(0,high=4)
            next_state,reward,done,_ = env.step(action)
            replayMemory.add(state,action,reward,next_state,done)
            state = next_state
            i += 1
    env.close()

def collectMeanScore(agent,steps,epsilon,env_name):
    env = make_atari(env_name)
    env = wrap_deepmind(env)
    evalAgent = DQNagent()
    evalAgent.Q.load_state_dict(agent.Q.state_dict())
    evalAgent.epsilon = epsilon
    rewards_sum = 0.0
    episodes = 0

    state = env.reset()
    while episodes in range(steps):
        
        action = evalAgent.getAction(LazyFrame2Torch(state))
        state, reward, done, _ = env.step(action)
        rewards_sum += reward
        if done:
            env.reset()
            episodes += 1

    average_score = rewards_sum/episodes
    env.close()
    return float(average_score)

def evaluateStateQvalues(agent):
    s = np.load('log/stateEval.npy') 
    s = torch.from_numpy(s).float().to('cuda')
    with torch.no_grad():
        q = agent.Q(s)
        q = torch.mean(q,1)
        q = torch.mean(q)
    return float(q)

def LazyFrame2Torch(x):
        y = x.__array__()[np.newaxis,:,:,:]
        y = np.moveaxis(y,3,1)
        y = torch.from_numpy(y).float().to('cuda')
        return y