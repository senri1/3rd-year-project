import numpy as np 
import gym
import torch
from gym import wrappers
from atari_wrappers import wrap_deepmind
from atari_wrappers import make_atari
from DQNAgent import DQNagent
from ReplayMemory import ReplayMemory
from ReplayMemory import LazyFrame2Torch

env_name = 'Breakout-v0'
env = make_atari(env_name)
env = wrap_deepmind(env)

frames = 1000
episodes = 0
batch_size = 32
memory_size = 500000
update_target_frequency = 1000
learning_rate = 0.000025

memory = ReplayMemory(memory_size, batch_size)
DQNagent0 = DQNagent()

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



action = DQNagent0.getAction(LazyFrame2Torch(state)) 
next_state,reward,done,_ = env.step(action)
memory.add(state,action,reward,next_state,done)
state = next_state