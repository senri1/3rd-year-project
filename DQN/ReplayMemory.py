from collections import deque
import random
import numpy as np
import torch 

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

        for n in range(batch.shape[0]):
            batch[n,1] = LazyFrame2Torch(batch[n,1])
            batch[n,3] = LazyFrame2Torch(batch[n,3])

        state = torch.from_numpy(batch[:,0]).to('cuda',torch.float)
        action = torch.from_numpy(batch[:,1]).to('cuda',torch.float)
        reward = torch.from_numpy(batch[:,2]).to('cuda',torch.float)
        next_state = torch.from_numpy(batch[:,3]).to('cuda',torch.float)
        not_done = torch.from_numpy(batch[:,4]).to('cuda',torch.float)
        
        return state, action, reward, next_state, not_done


def LazyFrame2Torch(x):
        x = x.__array__()[np.newaxis,:,:,:]
        x = np.moveaxis(x,3,1)
        x = torch.from_numpy(x).to('cuda',torch.float)
        return x