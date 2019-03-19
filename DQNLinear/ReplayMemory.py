from collections import deque
import random
import numpy as np
import os 
import pickle


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
        statenp = np.zeros((batch_size,84,84,4),dtype = 'uint8')
        next_statenp = np.zeros((batch_size,84,84,4), dtype = 'uint8')

        for n in range(batch_size):
            statenp[n,:,:,:] = batch[n,0].__array__()
            next_statenp[n,:,:,:] = batch[n,3].__array__()

        state = statenp
        action = batch[:,1]
        reward = batch[:,2]
        next_state = next_statenp
        not_done = batch[:,4]
        
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