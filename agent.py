from VAE2 import*
from sklearn.linear_model import LinearRegression

class myAgent:

    def __init__(
        self,
        epsilon=1,
        disc_factor = 0.9,
        num_actions=4,
        state_size = 400,
        env = 'Breakout-v0',
        encoder=None,
        decoder=None,
        CAE=None,
        policyType = 'lr'
        ):
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.state_size = state_size
        self.env = env
        self.Q = []
        self.encoder = encoder
        self.decoder = decoder
        self.CAE = CAE
        self.CAE_loss = 0
        self.policyType = policyType
        self.myPolicy = createPolicy(self.policyType,self.num_actions)
    
    def create_encoder(self,num_samples):
        """ This methoed creates a convolutional autoencoder trained on the agents current environment
        specified in self.env for samples (1 sample is 4 frames) specified in num_samples and returns the encoder
        and the training and validation loss for each epoch."""

        self.CAE, self.encoder, self.decoder, self.CAE_loss = ConvAE(num_samples,self.env,self)
        return self.encoder, self.CAE_loss


    def load_encoder(self):
        """ This method loads a convolutional autoencoder trained on the environment specified in self.env """

        self.CAE = tf.keras.models.load_model( os.getcwd() + '/CAE_' + self.env + '.h5' )
        self.encoder = tf.keras.models.load_model( os.getcwd() + '/encoder_' + self.env + '.h5')
        self.decoder = tf.keras.models.load_model( os.getcwd() + '/CAE_' + self.env + '.h5' )
        return self.encoder

    def save_encoder(self):
        """ saves the CAE """

        self.CAE.save('CAE_' + self.env + '.h5')
        self.encoder.save('encoder_' + self.env + '.h5')
        self.decoder.save('decoder_' + self.env + '.h5')

    def getQvalues(self,state):
        """ Takes in the state i.e. encoder(observation) and returns an array with length equal to
        the number of actions, where each element specifies the Q value for a given action. """

        Qvalues = np.zeros( (1,self.num_actions) )
        
        for j in range(self.num_actions):
                Qvalues[0,j] = self.myPolicy[j][0].predict(state.reshape((1,-1)))
        
        return Qvalues

    def getState(self,observation,samples):
        """ Takes in the processed image observations (1 observation is 4 frames) received
        from collecting the data in the environment and returns encoder(observation). """
        steps = samples*4 - 4
        states = np.zeros( ( steps ,84,84,4  ) )
        
        for i in range(steps):
            for j in range(4):
                states[i,:,:,j] = observation[i+j]

        
        return states

    def getAction(self,observation):
        """ Input: obervation is an input of shape (1,84,84,4)
            Output: if encoder is none outputs random action for random policy,
            else returns action from epsilon greedy policy."""

        if self.encoder == None:
            return np.random.randint(0,high=4)
        else:
            action = None
            state = self.encoder.predict( observation )
            Qvalues = self.getQvalues(state)
            probability = np.random.random_sample()

            if self.epsilon <= probability:
                action = np.argmax(Qvalues)
            else:
                action = np.random.randint(0,high=4)
            return action
        
    
    def imporve_policy(states,actions,rewards):
        X = np.zeros((states.shape[0],1))
        Y = np.zeros((states.shape[0],1))
        for i in range(states.shape[0]):
            X[i,1] = getQvalues(states[i,:])[actions[i]]
            Y[i,1] = rewards[i] +


samples = 100
agent = myAgent()
encoder = agent.load_encoder

for episodes in range(num_episodes):

    observation,actions,rewards,num_episodes = collectObs(samples,4,'Breakout-v0',agent)
    states = agent.getState(observation,samples)
    imporve_policy(states,actions,rewards)


    



