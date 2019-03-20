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
import os 
import pickle

class ReplayMemory():
    def __init__(self, size, batch_size):
        self.max_size = size
        self.current_size = 0
        self.data = deque(maxlen=size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        if self.current_size < self.max_size:
            self.data.append((state, int(action), reward, next_state, not done))
            self.current_size += 1
        else: 
            self.data.popleft()
            self.data.append((state, int(action), reward, next_state, not done))
    
    def get_batch(self):
        batch = random.sample(self.data, self.batch_size)
        batch = np.array(batch)
        statenp = np.zeros((self.batch_size,4,84,84))
        next_statenp = np.zeros((self.batch_size,4,84,84))

        for n in range(batch.shape[0]):
            statenp[n,:,:,:] = (np.moveaxis(batch[n,0].__array__()[np.newaxis,:,:,:],3,1)).astype(float)
            next_statenp[n,:,:,:] = (np.moveaxis(batch[n,3].__array__()[np.newaxis,:,:,:],3,1)).astype(float)

        state = torch.from_numpy(statenp).float().to('cuda')
        action = torch.from_numpy(batch[:,1].astype(float)).long().to('cuda')
        reward = torch.from_numpy(batch[:,2].astype(float)).float().to('cuda')
        next_state = torch.from_numpy(next_statenp).float().to('cuda')
        not_done = torch.from_numpy(batch[:,4].astype(float)).float().to('cuda')
        
        return state, action, reward, next_state, not_done
        
    def save_replay(self,j):
        agent_name = 'agent' + str(j)
        dir = 'saved_agents/' + agent_name + '/'
        try:
            os.mkdir(dir)
        except FileExistsError:
            print("Directory " , dir ,  " already exists")
        
        with open(os.getcwd() + '/' + dir + 'replay_memory.pckl' , "wb") as f:
            pickle.dump([self.max_size, self.current_size, self.data], f)
        
    def load_replay(self,j):
        agent_name = 'agent' + str(j)
        dir = 'saved_agents/' + agent_name + '/'
        with open(os.getcwd() +'/' + dir +'replay_memory.pckl', "rb") as f:
            replay_memory = pickle.load(f)
        self.max_size = replay_memory[0]
        self.current_size = replay_memory[1]
        self.data = replay_memory[2]

