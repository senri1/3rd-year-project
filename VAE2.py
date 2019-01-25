import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import gym
import numpy as np
import cv2
import torch 

def ConvAE(train_samples,envName,agent):

    """ Creats a convolutional autoencoder.

        Input:  train_samples is the number of samples to train on. envName is the name of the
                environment to get the samples from. agent is an object with method getAction
                that determines the actions taken in the environment to take samples.

        Returns: Returns encoder, decoder and CAE objects as well as the history of training and
                 validation losses every epoch."""


    X,_,_,_,_ = collectObs(train_samples,1,envName,agent)
    X = Img2Frame(X)

    height = X.shape[1]
    width = X.shape[2]
    channels = X.shape[3]   
    X_train, X_test = train_test_split(X,test_size=0.2)


    encoder_input = tf.keras.layers.Input( shape = (height,width,channels))
    encoded = layers.Conv2D( filters = 32, kernel_size = 8, padding = 'valid', strides = 4, activation = 'relu', input_shape = (height,width,channels) ) (encoder_input)
    encoded = layers.Conv2D( filters = 64, kernel_size = 4, padding = 'valid', strides = 2, activation = 'relu' ) (encoded)
    encoded = layers.Conv2D( filters = 64, kernel_size = 3, padding = 'valid', strides = 1, activation = 'relu' ) (encoded)
    encoded = layers.Conv2D( filters = 16, kernel_size = 3, padding = 'valid', strides = 1, activation = 'relu', name='encoder_output' ) (encoded)
    encoder = tf.keras.Model( encoder_input, encoded )

    decoder_input = tf.keras.layers.Input( shape = ( encoder.output_shape[1], encoder.output_shape[2], encoder.output_shape[3] ) )
    decoded = layers.Conv2DTranspose( filters = 16, kernel_size = 3, padding = 'valid', strides = 1, activation = 'relu', input_shape = (encoder.output_shape[1], encoder.output_shape[2], encoder.output_shape[3]) )(decoder_input)
    decoded = layers.Conv2DTranspose( filters = 64, kernel_size = 3, padding = 'valid', strides = 1, activation = 'relu' ) (decoded)
    decoded = layers.Conv2DTranspose( filters = 64, kernel_size = 4, padding = 'valid', strides = 2, activation = 'relu' ) (decoded)
    decoded = layers.Conv2DTranspose( filters = 32, kernel_size = 8, padding = 'valid', strides = 4, activation = 'relu', name='decoder_output' ) (decoded)
    decoded = layers.Conv2DTranspose( filters = 4, kernel_size = 1, padding = 'valid', strides = 1, activation = 'sigmoid' ) (decoded)
    decoder = tf.keras.Model( decoder_input, decoded )

    autoencoder = decoder( encoder(encoder_input) )
    CAE = tf.keras.Model( encoder_input, autoencoder )

    print(CAE.summary())

    CAE.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

    History = CAE.fit(X_train,
                X_train,
                batch_size=32,
                epochs=10,
                verbose=1,
                validation_data=(X_test, X_test)
                )

    print("\nSuccess.")
    
    return CAE,encoder,decoder,History.history


def preprocess(observation):    

    """ Converts single RGB image into an image with 1 bit colour of size 84,84

        Input: An image of of shape (210,160,3) with 8 bit colour.

        Returns: An image of shape shape (84,84) with 1 bit colour. """

    observation = cv2.cvtColor(cv2.resize(observation,(84,110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]                                                         # get rid of first 27 rows of image
    ret, observation = cv2.threshold(observation,1,1,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))
    

def collectObs(samples,k,envName,agent):            

    """ Collects observations to be used as training data for the convolutional autoencoder.

        Input: samples, the number data points needed in the training set one sample is 4 frames
               of 1 bit (84,84) images. envName is the environment to collect the observations from.
               agent is on abject with method getAction which determines how the data will be collected.

        Returns: Obs, 4 dimensional array of shape (steps,84,84,1) of observations 1st dimension is number of steps.
                 actions, a column vector of shape (steps,1) containing the action made at each step.
                 rewards, a column vector of shape (steps,1) containing the reward received at each step.
                 Also returns the number of episodes needed to collect the observaitons.  
                 Note: Steps is equal to number of frames and samples*4 """

    samples = samples + 1
    steps = samples*4                                          # 4 steps is 1 sample
    obs = np.zeros( (steps,84,84,1), dtype = 'uint8' )
    actions = np.zeros( (steps,1) , dtype = 'uint8' )
    rewards = np.zeros( (steps,1) , dtype = 'uint8' )
    ep_rewards = np.zeros((20000,1)) 
    num_episodes = 0

    env = gym.make(envName)
    env.reset()

    for n in range(4):
        observation, reward, done, _ = env.step(0)
        obs[n,:,:,:] = preprocess(observation)[np.newaxis]
        actions[n,0] = 0
        rewards[n,0] = reward
        ep_rewards[0,0] += reward

    for i in range(steps-4):

        if i%k == 0:
            action = agent.getAction( Img2Frame(obs[i:i+4,:,:,:]) )                                                     

        observation, reward, done, info = env.step(action)

        obs[i+4,:,:,:] = preprocess(observation)[np.newaxis]
        actions[i+4,0] = action
        rewards[i+4,0] = reward
        ep_rewards[num_episodes,0] += reward 

        if done:
            num_episodes = num_episodes + 1
            env.reset()
        
    return obs,actions,rewards,num_episodes,ep_rewards                                                  


def createPolicy(policyType ,num_actions):

    """ Initialises linear model objects from OLS linear regression, LASSO and ridge regression.
        One model is initalised for each action.

        Input: policyType, a string from 'lr', 'l1' and 'l2' for OLS, LASSO and ridge respectively. 
               num_actions specifies how many policy objects to initialise.

        Returns: An array containing 4 linear models one for each action."""

    initialisex = np.zeros((1,400))
    initialisey = np.zeros((1,1))

    policy = []
    if policyType == 'lr':
        for i in range(num_actions):
            policy.append([lm.LinearRegression(),StandardScaler()])     
            policy[i][0].fit(initialisex,initialisey)                                # Initialises weights to zero
    if policyType == 'l1':
        for i in range(num_actions):
            policy.append([lm.Lasso(),StandardScaler()])
            policy[i][0].fit(initialisex,initialisey)
    if policyType == 'l2':
        for i in range(num_actions):
            policy.append([lm.Ridge(),StandardScaler()])
            policy[i][0].fit(initialisex,initialisey)
    return policy        

def Img2Frame(observation):
    observations = np.moveaxis(observation,3,1).reshape((-1,4,84,84))
    observation = np.moveaxis(observations,1,3)
    return observation


