#from LinearAgent import Linearagent
from atari_wrappers import wrap_deepmind
from atari_wrappers import make_atari
import numpy as np
import os
import torch
from LinearAgent import Linearagent


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
        print(i)
        
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
        
        action = evalAgent.getAction(LazyFrame2Torch(state))
        state, reward, done, _ = env.step(action)
        rewards_sum += reward
        if done:
            env.reset()
            print(episodes)
            episodes += 1

    average_score = rewards_sum/episodes
    env.close()
    return float(average_score)

def evaluateStateQvalues(agent):
    
    s = np.load('log/stateEval.npy')
    q = np.zeros((s.shape[0],agent.num_actions))
    s = torch.from_numpy(s).float().to('cuda')
    with torch.no_grad():
        z = agent.encoder(s)
    _,_,z = getStandard(z.detach().cpu().numpy())
    #z = z.view(-1,400).detach().cpu().numpy()
    z = z - agent.mean
    z = np.divide(z, agent.sd, out=np.zeros_like(z), where=agent.sd!=0)
    for i in range(agent.num_actions):
        q[:,i] = agent.Linear[i].predict(z)
    q = np.mean(q,axis=1)
    q = np.mean(q,axis=0)
    return float(q)


def getStandard(state):

    """ Input: State of shape (samples,16,16,5)
        Returns: Standard deviation and mean of states and flattened state of shape (samples,400) """
    state_temp = np.zeros((state.shape[0],400))     
    state = np.moveaxis(state,3,1)
    for i in range(state.shape[0]):
        state_temp[i,:] = state[i,:,:,:].reshape((1,400))
    mean = np.mean(state_temp,axis=0)
    sd = np.std(state_temp,axis=0)
    return mean, sd, state_temp

def evaluate():

    TestAgent = Linearagent()
    evaluation_data=[]
    idx = int(np.load('log/idx.npy'))
    env_name = 'Breakout-v0'

    for i in range(449):

        print(i)
        print(TestAgent.training_steps)
        TestAgent.load_agent(str(i))            
        evaluation_data.append([ TestAgent.training_steps, collectMeanScore(TestAgent,20,0.05,env_name), evaluateStateQvalues(TestAgent) ])

    np.savetxt("log/eval_data.csv", evaluation_data, delimiter=",")


def LazyFrame2Torch(x):
        y = x.__array__()[np.newaxis,:,:,:]
        y = np.moveaxis(y,3,1)
        y = torch.from_numpy(y).float().to('cuda')
        return y






