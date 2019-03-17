import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from OLSModel import OLS

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
        self.CAE, self.encoder, _ = create_CAE()
        self.disc_factor = disc_factor
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.Linear = createLinearLayers(linearType,num_actions)
        self.LinearTarget = createLinearLayers(linearType,num_actions)

    def train_autoencoder(self, memory, data_size):

        X,_,_,_,_ = memory.get_batch(data_size)
        X_train, X_test = train_test_split(X,test_size=0.1)

        History = self.CAE.fit(X_train,
                    X_train,
                    batch_size=32,
                    epochs=10,
                    verbose=1,
                    validation_data=(X_test, X_test)
                    )       

        # Get the standard deviation and mean of the states produced by the encoder from the training data.    
        state = self.encoder.predict(X_train)
        mean,sd,_ = getStandard(state)

    
    def getQvalues(self,state):
        Q = np.zeros(self.num_actions)
        z = self.encoder(state)
        _,_z = getStandard(z)
        for i in range(self.num_actions):
            Q[i] = self.Linear[i].predict(z)
        return Q

    def getAction(self,state):

        if self.encoder == None:
            return np.random.randint(0,high=4)

        else:

            Qvalues = self.getQvalues(state)
            probability = np.random.random_sample()

            if self.epsilon <= probability:
                action = np.argmax(Qvalues) 
            else:
                action = np.random.randint(0,high=4)
            return action

    def getData(self,state_batch,next_state_batch,action_batch,not_done_batch):
        _,_,state_batch = getStandard(state_batch)
        _,_,next_state_batch = getStandard(next_state_batch)
        next_state_batch = next_state_batch * not_done_batch
        state = []
        next_state = []
        for i in range(self.num_actions):
            s = state_batch*(action_batch==i)
            s = s[~(s==0).all(1)]
            ns = next_state_batch*(action_batch==i)
            ns = ns[~(ns==0).all(1)]
            state.append(s)
            next_state.append(ns)
        return state, next_state

    def decrease_epsilon(self,steps):
        min_epsilon = 0.1
        init_epsilon = 1.0
        eps_decay_steps = 1000000.0
        self.epsilon = max(min_epsilon, init_epsilon - (init_epsilon-min_epsilon) * float(steps)/eps_decay_steps)



"""
done = False
initial_state = env.reset()
action = agent.getAction(LazyFrame2Torch(initial_state)) 
state, reward, done, _ = env.step(action)
memory.add(initial_state,action,reward,state,done )

action = agent.getAction(LazyFrame2Torch(state)) 
next_state,reward,done,_ = env.step(action)
memory.add(state,action,reward,next_state,done)
state = next_state

class encoder(torch.nn.Module):
    def __init__(self,num_actions):
        super(encoder,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0).cuda()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0).cuda()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0).cuda()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=0).cuda()
    
    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        return out

class decoder(torch.nn.Module):
    def __init__(self):
        super(decoder,self).__init__()
        self.conv1T = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1,padding=0)
        self.conv2T = nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=3, stride=1,padding=0)
        self.conv3T = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,padding=0)
        self.conv4T = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=8, stride=4,padding=0)
        self.conv5T = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=1, stride=1,padding=0)
    
    def forward(self,y):
        out = F.relu(self.conv1T(y))
        out = F.relu(self.conv2T(out))
        out = F.relu(self.conv3T(out))
        out = F.relu(self.conv4T(out))
        out = F.sigmoid(self.conv5T(out))
        return out


"""