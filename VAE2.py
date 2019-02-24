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
import time
import OLSModel

def ConvAE(train_samples,envName,agent):

    """ Creats a convolutional autoencoder.

        Input:  train_samples is the number of samples to train on 1 sample is 4 frames concatenated together 
                to have shape of (1,84,84,4). envName is the name of the environment to get the samples from.
                agent is an object with method getAction that determines the actions taken in the environment 
                to take samples.                 
                
        Returns: Returns encoder, decoder and CAE objects, the history of training and validation losses every
                 epoch and the mean and standard deviation of the states produced by encoder from the training 
                 to be used for standardization. """

    # Collect observations in X of shape (train_samples*4,84,84,1), then group observations into stacks of 4
    # using Img2Frame to turn X into shape (train_samples,84,84,4).
    X,_,_,_,_ = collectObs(train_samples,1,envName,agent)
    X = Img2Frame(X)

    # Get height width and channels and split the training data into train and test. 
    height = X.shape[1]
    width = X.shape[2]
    channels = X.shape[3]   
    X_train, X_test = train_test_split(X,test_size=0.2)

    # Define the structure of the encoder
    encoder_input = tf.keras.layers.Input( shape = (height,width,channels))
    encoded = layers.Conv2D( filters = 32, kernel_size = 8, padding = 'valid', strides = 4, activation = 'relu', input_shape = (height,width,channels) ) (encoder_input)
    encoded = layers.Conv2D( filters = 64, kernel_size = 4, padding = 'valid', strides = 2, activation = 'relu' ) (encoded)
    encoded = layers.Conv2D( filters = 64, kernel_size = 3, padding = 'valid', strides = 1, activation = 'relu' ) (encoded)
    encoded = layers.Conv2D( filters = 16, kernel_size = 3, padding = 'valid', strides = 1, activation = 'relu', name='encoder_output' ) (encoded)
    encoder = tf.keras.Model( encoder_input, encoded )

    # Define the structure of the decoder
    decoder_input = tf.keras.layers.Input( shape = ( encoder.output_shape[1], encoder.output_shape[2], encoder.output_shape[3] ) )
    decoded = layers.Conv2DTranspose( filters = 16, kernel_size = 3, padding = 'valid', strides = 1, activation = 'relu', input_shape = (encoder.output_shape[1], encoder.output_shape[2], encoder.output_shape[3]) )(decoder_input)
    decoded = layers.Conv2DTranspose( filters = 64, kernel_size = 3, padding = 'valid', strides = 1, activation = 'relu' ) (decoded)
    decoded = layers.Conv2DTranspose( filters = 64, kernel_size = 4, padding = 'valid', strides = 2, activation = 'relu' ) (decoded)
    decoded = layers.Conv2DTranspose( filters = 32, kernel_size = 8, padding = 'valid', strides = 4, activation = 'relu', name='decoder_output' ) (decoded)
    decoded = layers.Conv2DTranspose( filters = 4, kernel_size = 1, padding = 'valid', strides = 1, activation = 'sigmoid' ) (decoded)
    decoder = tf.keras.Model( decoder_input, decoded )

    # Join encoder and decoder together to get a convolutional autoencoder.
    autoencoder = decoder( encoder(encoder_input) )
    CAE = tf.keras.Model( encoder_input, autoencoder )

    # Print the summary, then compile and fit (train).
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

    # Get the standard deviation and mean of the states produced by the encoder from the training data.    
    state = encoder.predict(X_train)
    mean,sd,_ = standardize(state)

    return CAE,encoder,decoder,History.history,mean,sd

def collectObs(samples,k,envName,agent):            

    """ Input: samples, the number data points needed in the training set one sample is shape (1,84,84,4).
               k is a parameter that decides for how many frames an action is repeated till a new one is chosen 
               through getAction. 
               envName is the environment to collect the observations from.
               agent is on abject with method getAction which determines how the data will be collected.

        Returns: obs, an array of observations of shape (samples*4,84,84,1).
                 actions, a column vector of shape (samples*4,1) containing the action made at each observation.
                 rewards, a column vector of shape (samples*4,1) containing the reward received at each observation.
                 num_ep, the number of episodes needed to collect the samples.
                 ep_rewards, array of shape (num_ep,1) with total reward received each episode. """

    # Increase samples by 1 to accomodate for the initial 4 observations required to get an action from the policy
    # since a single state is 4 frames. 
    samples = samples + 1
    # Steps is total number of frames/observations.
    steps = samples*4
    #Create arrays to store values of interest.                                          
    obs = np.zeros( (steps,84,84,1), dtype = 'uint8' )
    actions = np.zeros( (steps,1) , dtype = 'uint8' )
    rewards = np.zeros( (steps,1) , dtype = 'uint8' )
    ep_rewards = np.zeros((4000,1)) 
    num_episodes = 0

    # Initialise game environment.
    env = gym.make(envName)
    env.reset()

    # Agents policy needs 4 observations to make action (since 1 state = 4 frames).
    for n in range(4):
        observation, reward, done, _ = env.step(0)
        obs[n,:,:,:] = preprocess(observation)[np.newaxis]
        actions[n,0] = 0
        rewards[n,0] = reward
        ep_rewards[0,0] += reward

    # Generate the observations
    for i in range(steps-4):

        # Decide on a new action every k steps.
        if i%k == 0:
            action = agent.getAction( Img2Frame(obs[i:i+4,:,:,:]) )      

        # Uncomment to render the gameplay at 60fps.
        #time.sleep(0.016)                                               
        #env.render()
        
        # Perform the action and save relevant values.
        observation, reward, done, info = env.step(action)
        obs[i+4,:,:,:] = preprocess(observation)[np.newaxis]
        actions[i+4,0] = action
        rewards[i+4,0] = reward
        ep_rewards[num_episodes,0] += reward 

        #If an episode is over increment num_episodes and reset the envirinment.
        if done:
            num_episodes = num_episodes + 1
            env.reset()
    env.close()
    return obs,actions,rewards,num_episodes,ep_rewards                                                  


def createPolicy(policyType ,num_actions):

    """ Input: policyType, a string from 'lr', 'l1' and 'l2' for OLS, LASSO and ridge respectively. 
               num_actions specifies how many policy objects to initialise, one for each action.

        Returns: An array containing 4 linear model objects one for each action."""

    # Used to initialise weights to random values. 
    initialisex = np.random.randn(4,400)*10
    initialisey = np.random.randn(4,1)*10

    policy = []

    # Depending on policyType, create a list of the desired linear models.
    if policyType == 'lr':
        for i in range(num_actions):
            policy.append(OLSModel.OLS())     
            policy[i].fit(initialisex,initialisey)                                
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
    """ Input: observation, an array of shape (samples*4,84,84,1)
        Returns: observation as array of shape (samples,84,84,4). """

    observations = np.moveaxis(observation,3,1).reshape((-1,4,84,84))
    observation = np.moveaxis(observations,1,3)
    return observation

def preprocess(observation):    

    """ Input: An image of of shape (210,160,3) with 8 bit colour.

        Returns: An image of shape shape (84,84,1) with 1 bit colour. """
    # Convert to grey scale.
    observation = cv2.cvtColor(cv2.resize(observation,(84,110)), cv2.COLOR_BGR2GRAY)
    
    # Get rid of first 27 rows of image since they are just the score.
    observation = observation[26:110,:]                                                         
    
    # Convert to 1 bit colour.
    ret, observation = cv2.threshold(observation,1,1,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))
    
def standardize(state):

    """ Input: State of shape (samples,16,16,5)
        Returns: Standard deviation and mean of states and flattened state of shape (samples,400) """
    state_temp = np.zeros((state.shape[0],400))
    state = np.moveaxis(state,3,1)
    for i in range(state.shape[0]):
        state_temp[i,:] = state[i,:,:,:].reshape((1,400))
    mean = np.mean(state_temp,axis=0)
    sd = np.std(state_temp,axis=0)
    return mean, sd, state_temp