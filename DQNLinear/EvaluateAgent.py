from LinearAgent import Linearagent
from atari_wrappers import wrap_deepmind
from atari_wrappers import make_atari
import numpy as np
import os
import torch

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
    evalAgent = agent
    evalAgent.epsilon = epsilon
    rewards_sum = 0.0
    episodes = 0

    state = env.reset()
    while episodes in range(steps):
        
        action = evalAgent.getAction(np.array(state.__array__()[np.newaxis,:,:,:]))
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
    s = np.moveaxis(s,1,3)
    q = agent.getQvalues(s)
    q = np.mean(q,axis=1)
    q = np.mean(q,axis=0)
    return float(q)

def evaluate():

    TestAgent = Linearagent()
    evaluation_data=[]
    idx = int(np.load('log/idx.npy'))
    env_name = 'Breakout-v0'

    for i in range(idx):

        print(i)
        TestAgent.load_agent(str(i))            
        evaluation_data.append([ collectMeanScore(TestAgent,7,0.05,env_name), evaluateStateQvalues(TestAgent) ])

    np.savetxt("log/eval_data.csv", evaluation_data, delimiter=",")