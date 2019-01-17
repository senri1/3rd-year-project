import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess(observation):              # converts rgb to black and white
    observation = cv2.cvtColor(cv2.resize(observation,(84,110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]         # get rid of first 27 rows of image
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84))

def collectData(steps):          # collects steps amount data, using random actions
    env = gym.make('breakout-v0')
    env.reset()
    steps = steps
    tdata = np.zeros( (steps,84,84) )

    for i in range(steps):

        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        imge[i,:,:,:] = observation
        tdata[i,:,:] = preprocess(observation)
        if done:
            env.reset()

    return tdata


