import numpy as np 
import gym
import torch
import torch.nn.functional as F
from gym import wrappers
from atari_wrappers import wrap_deepmind
from atari_wrappers import make_atari
from DQNAgent import DQNagent
from ReplayMemory import ReplayMemory
from ReplayMemory import LazyFrame2Torch

env_name = 'Breakout-v0'
env = make_atari(env_name)
env = wrap_deepmind(env)

frames = 100
episodes = 0
batch_size = 32
memory_size = 500000
update_target_frequency = 1000
learning_rate = 0.00025
update_frequency = 1000
discount = 0.99

memory = ReplayMemory(memory_size, batch_size)
DQNagent0 = DQNagent()
#optimizer = torch.optim.adam(DQNagent0.Qnetwork.parameters(), lr = learning_rate)
optimizer = torch.optim.RMSprop(DQNagent0.Q.parameters(), lr=learning_rate, eps=0.01, alpha=0.95)

for n in range(frames):
  
    done = False
    initial_state = env.reset()
    action = DQNagent0.getAction(LazyFrame2Torch(initial_state)) 
    state, reward, done, _ = env.step(action)
    memory.add(initial_state,action,reward,state,done )
    
    while not done:
        
        action = DQNagent0.getAction(LazyFrame2Torch(state)) 
        next_state,reward,done,_ = env.step(action)
        memory.add(state,action,reward,next_state,done)
        state = next_state

        if memory.current_size >= batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = memory.get_batch()
            optimizer.zero_grad()
            qvalues = DQNagent0.Q(state_batch)[range(batch_size), action_batch]
            qtargetValues, _ = torch.max(DQNagent0.QTarget(next_state_batch), 1)
            qtargetValues = not_done_batch * qtargetValues
            qtarget = reward_batch + discount * qtargetValues
            qtarget = qtarget.detach()
            loss = F.mse_loss(qvalues,qtarget)
            loss.backward()
            optimizer.step()

        if n % update_frequency == 0:
            DQNagent0.QTarget.load_state_dict(DQNagent0.Q.state_dict())

    episodes += 1


"""
done = False
initial_state = env.reset()
action = DQNagent0.getAction(LazyFrame2Torch(initial_state)) 
state, reward, done, _ = env.step(action)
memory.add(initial_state,action,reward,state,done )

action = DQNagent0.getAction(LazyFrame2Torch(state)) 
next_state,reward,done,_ = env.step(action)
memory.add(state,action,reward,next_state,done)
state = next_state
"""