import numpy as np 
import gym
from gym import wrappers
from atari_wrappers import wrap_deepmind
from atari_wrappers import make_atari
from DQNAgent import DQNagent

env_name = 'Breakout-v0'
env = make_atari(env_name)
env = wrap_deepmind(env)
DQNagent0 = DQNagent()

iterations = 1000
batch_size = 32
memory_size = 1000000
update_target_frequency = 1000
learning_rate = 0.000025

for n in range(iterations):

    istate = env.reset()
    done = false
    
    while not done:
        
        action = DQNagent0.getAction(istate) 
        states,rewards,done,_ = env.step(action)
        