import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from OLSModel import OLS
import pickle
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def create_CAE():

    """ Creats a convolutional autoencoder.

        Input:  train_samples is the number of samples to train on 1 sample is 4 frames concatenated together 
                to have shape of (1,84,84,4). envName is the name of the environment to get the samples from.
                agent is an object with method getAction that determines the actions taken in the environment 
                to take samples.                 
                
        Returns: Returns encoder, decoder and CAE objects, the history of training and validation losses every
                 epoch and the mean and standard deviation of the states produced by encoder from the training 
                 to be used for standardization. """

    # Get height width and channels and split the training data into train and test. 
    height = 84
    width = 84
    channels = 4

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
    
    return CAE,encoder,decoder

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

def createLinearLayers(layerType ,num_actions):

    linearLayers = []

    # Depending on policyType, create a list of the desired linear models.
    if layerType == 'lr':
        for i in range(num_actions):
            linearLayers.append(OLS(0.0))             

    if layerType == 'l2':
        for i in range(num_actions):
            linearLayers.append(OLS(1.0))

    return linearLayers        


class Linearagent():
    
    def __init__(
        self,
        epsilon=1,
        disc_factor = 0.99,
        num_actions=4,
        linearType='lr'
        ):
        self.CAE, self.encoder, self.decoder = create_CAE()
        self.mean = None
        self.sd = None
        self.disc_factor = disc_factor
        self.epsilon = epsilon
        self.min_epsilon = 0.1
        self.init_epsilon = 1
        self.training_steps = 0
        self.num_actions = num_actions
        self.Linear = createLinearLayers(linearType,num_actions)
        self.LinearTarget = createLinearLayers(linearType,num_actions)

    def train_autoencoder(self, memory, data_size):

        X,_,_,_,_ = memory.get_batch(data_size)
        X_train, X_test = train_test_split(X,test_size=0.1)

        History = self.CAE.fit(X_train,
                    X_train,
                    batch_size=32,
                    epochs=5,
                    verbose=1,
                    validation_data=(X_test, X_test)
                    )       

        # Get the standard deviation and mean of the states produced by the encoder from the training data.    
        state = self.encoder.predict(X_train)
        self.mean, self.sd,_ = getStandard(state)

    def train(self, state_batch, qtargets,steps):
        for i in range(self.num_actions):
            if state_batch[i].size != 0:
                self.Linear[i].fit(state_batch[i], qtargets[i],steps)

    def getQvalues(self,state):
        Q = np.zeros(self.num_actions)
        z = self.encoder.predict(state)
        _,_,z = getStandard(z)
        z = z - self.mean
        z = np.divide(z, self.sd, out=np.zeros_like(z), where=self.sd!=0)
        for i in range(self.num_actions):
            Q[i] = self.Linear[i].predict(z)
        return Q

    def getAction(self,state):

        Qvalues = self.getQvalues(state)
        probability = np.random.random_sample()

        if self.epsilon <= probability:
            action = np.argmax(Qvalues) 
        else:
            action = np.random.randint(0,high=4)
        return action

    def getTrainingData(self,state_batch,qtarget,action_batch):

        _,_,state_batch = getStandard(self.encoder.predict(state_batch))
        state_batch = state_batch - self.mean
        state_batch = np.divide(state_batch, self.sd, out=np.zeros_like(state_batch), where=self.sd!=0)
        state = []
        qtargets = []
        for i in range(self.num_actions):
            mask = np.nonzero(~(action_batch==i))
            s = np.delete(state_batch, mask[0], 0)
            q = np.delete(qtarget, mask[0], 0)
            state.append(s)
            qtargets.append(q)
        return state, qtargets

    def getQtargets(self, next_state_batch, reward_batch, not_done_batch):
        _,_,next_state_batch = getStandard(self.encoder.predict(next_state_batch))
        next_state_batch = next_state_batch - self.mean
        next_state_batch = np.divide(next_state_batch, self.sd, out=np.zeros_like(next_state_batch), where=self.sd!=0)
        idx = np.nonzero(not_done_batch==0)
        next_state_batch[idx[0],:] = 0
        qtarget = np.zeros((next_state_batch.shape[0],self.num_actions))
        for i in range(self.num_actions):
            qtarget[:,i] = self.LinearTarget[i].predict(next_state_batch)
        qtarget = np.max(qtarget,axis=1,keepdims=True)
        qtarget = reward_batch + self.disc_factor * qtarget
        return qtarget

    def updateQTarget(self):
        for i in range(self.num_actions):
            self.LinearTarget[i].setWeights(self.Linear[i].getWeights())

    def decrease_epsilon(self):
        self.training_steps +=1
        eps_decay_steps = 1000000.0
        self.epsilon = max(self.min_epsilon, self.init_epsilon - (self.init_epsilon-self.min_epsilon) * float(self.training_steps)/eps_decay_steps)

    def saveLinear(self,dir):
        for i in range(self.num_actions):
            torch.save(self.Linear[i].getWeights(), dir + 'LinearAction' + str(i) + '.pth')
    
    def saveLinearTarget(self,dir):
        for i in range(self.num_actions):   
            torch.save(self.Linear[i].getWeights(), dir + 'LinearTargetAction' + str(i) + '.pth')

    def loadLinear(self,dir):
        for i in range(self.num_actions):
            state_dict = torch.load(dir + 'LinearAction' + str(i) + '.pth')
            self.Linear[i].setWeights(state_dict)
    
    def loadLinearTarget(self,dir):
        for i in range(self.num_actions):
            state_dict = torch.load(dir + 'LinearTargetAction' + str(i) + '.pth')
            self.LinearTarget[i].setWeights(state_dict)

    def save_encoder(self,dir):
        """ saves the CAE as well as the mean and standard deviation of states obtained from the data it was trained on."""

        self.CAE.save(os.getcwd() + '/' + dir + 'CAE.h5')
        self.encoder.save(os.getcwd() + '/' + dir + 'encoder.h5')
        self.decoder.save(os.getcwd() + '/' + dir + 'decoder.h5')
        np.save(os.getcwd() + '/' + dir + 'mean' ,self.mean)
        np.save(os.getcwd() + '/' + dir + 'sd',self.sd)
    
    def load_encoder(self,j):
        """ This method loads a convolutional autoencoder trained on the environment specified in self.env, as well as
            the mean and standard deviation of the states obtained from the training data used to train the CAE. """

        self.CAE = tf.keras.models.load_model( os.getcwd() + '/' + dir + 'CAE.h5' )
        self.encoder = tf.keras.models.load_model( os.getcwd() + '/' + dir + 'encoder.h5')
        self.decoder = tf.keras.models.load_model( os.getcwd() + '/' + dir + 'decoder.h5' )
        self.mean = np.load(os.getcwd() + '/' + dir + 'mean' + '.npy')
        self.sd = np.load(os.getcwd() + '/' + dir + 'sd' + '.npy')
        return self.encoder

    def save_agent(self,j):
        agent_name = 'agent' + str(j)
        dir = 'saved_agents/' + agent_name + '/'
        
        try:
            os.mkdir(dir)
        except FileExistsError:
            print("Directory " , dir ,  " already exists")

        self.saveLinear(dir)
        self.saveLinearTarget(dir)
        self.save_encoder(dir)
        with open(os.getcwd() + '/' + dir + 'metadata.pckl' , "wb") as f:
            pickle.dump([self.epsilon, self.training_steps], f)

    def load_agent(self,j):
        agent_name = 'agent' + str(j)
        dir = 'saved_agents/' + agent_name + '/'
        self.loadLinear(dir)
        self.loadLinearTarget(dir)
        self.load_encoder(dir)
        with open(os.getcwd() +'/' + dir +'metadata.pckl', "rb") as f:
            metadata = pickle.load(f)
        self.epsilon = metadata[0]
        self.training_steps = metadata[1]
