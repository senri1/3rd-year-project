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

    X,_,_,_ = collectData(train_samples,envName,agent)

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


def preprocess(observation):                                                                    # converts rgb to black and white

    observation = cv2.cvtColor(cv2.resize(observation,(84,110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]                                                         # get rid of first 27 rows of image
    ret, observation = cv2.threshold(observation,1,1,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84))
    

def collectData(samples,envName,agent):                                                                       # collects steps amount data, using random actions

    env = gym.make(envName)
    env.reset()

    steps = samples*4                                                                           # 4 steps is 1 sample
    obs = np.zeros( (samples,84,84,4), dtype = 'uint8' )
    actions = np.zeros( (samples,1), dtype = 'uint8' )
    rewards = np.zeros( (samples,1), dtype = 'uint8' )
    num_episodes = 0

    j = 0
    observation, reward, done, info = env.step(0)

    for i in range(steps):
        current_sample = int(np.floor(i/4))

        if j%4 == 0:
            j=0

        action = agent.getAction(observation)                                                      # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        obs[current_sample ,:,:,j] = preprocess(observation) 
        actions[current_sample,1] = action
        rewards[current_sample,1] = reward

        if done:
            num_episodes = num_episodes + 1
            env.reset()

        j = j+1
        
    return obs,actions,rewards,num_episodes                                                                               # returns samples*84*84*4 tensor


def createPolicy(policyType,num_actions):

    """ This method initialises linear model objects as the policy based on 
    the policy type selected between lr = oridinairy least squares linear regressoin,
    l1 = LASSO, l2 = ridge regression. It also creates a Standard scaler object for
    standardisation """

    policy = []
    if policyType == 'lr':
        for _ in range(num_actions):
            policy.append([lm.LinearRegression(),StandardScaler()])
    if policyType == 'l1':
        for i in range(num_actions):
            policy.append(lm.Lasso(),StandardScaler()])
    if policyType == 'l2':
        for i in range(num_actions):
            policy.append(lm.Ridge(),StandardScaler()])
    return policy        