from helperFuncs import*

class CNNagent():
    
    def __init__(
        self,
        epsilon=1,
        disc_factor = 0.99,
        num_actions=4
        ):
        self.CNN = None
        self.CNNupdate = None 
        self.disc_factor = disc_factor
        self.epsilon = epsilon
        self.num_actions = num_actions

    def create_CNN(self):
        
        height = 84
        width = 84
        channels = 4

        CNN_input = tf.keras.layers.Input( shape = (height,width,channels))
        self.CNN = layers.Conv2D( filters = 32, kernel_size = 8, padding = 'valid', strides = 4, activation = 'relu', input_shape = (height,width,channels) ) (CNN_input)
        self.CNN = layers.Conv2D( filters = 64, kernel_size = 4, padding = 'valid', strides = 2, activation = 'relu' ) (self.CNN)
        self.CNN = layers.Conv2D( filters = 64, kernel_size = 3, padding = 'valid', strides = 1, activation = 'relu' ) (self.CNN)
        self.CNN = layers.Flatten() (self.CNN)
        self.CNN = layers.Dense(units = 256, activation = 'relu') (self.CNN)
        self.CNN = layers.Dense(units = 4) (self.CNN)
        self.CNN = tf.keras.Model( CNN_input, self.CNN )

        self.CNN.compile(loss='mse',
                optimizer='adam',
                metrics=['mse'])
        
        return self.CNN
    
    def getQvalues(self,state):
        return self.CNN.predict(state)

    def getAction(self,state):

        if self.CNN == None:
            return np.random.randint(0,high=4)

        else:

            Qvalues = self.getQvalues(state)
            probability = np.random.random_sample()

            if self.epsilon <= probability:
                action = np.argmax(Qvalues)
            else:
                action = np.random.randint(0,high=4)
            return action

    def getState(self,observation):
        numStates = observation.shape[0] - 4
        states = np.zeros( ( numStates ,84,84,4  ) )
        for i in range(numStates):
            states[i:i+1,:,:,:] = Img2Frame( observation[i+1:i+5,:,:,:] ) 
        return states

    def getTrainingData(self,states,actions,rewards):
        Y = np.zeros((actions.shape[0]-1,1))
        for i in range(actions.shape[0]-1):
            Y[i,0] = rewards[i] + self.disc_factor * np.max(self.getQvalues(states[i,:,:,:]))
        return states[0:-1,:,:,:],Y

    def improve_policy(self,X,Y)