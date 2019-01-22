from VAE2 import*
from sklearn.linear_model import LinearRegression

class myAgent:

    def __init__(
        self,
        epsilon=1,
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
        self.myPolicy = Linearpolicy(self.policyType,self.num_actions)
    
    def create_encoder(num_samples):
        """ This methoed creates a convolutional autoencoder trained on the agents current environment
        specified in self.env for samples (1 sample is 4 frames) specified in num_samples and returns the encoder
        and the training and validation loss for each epoch."""

        self.CAE, self.encoder, self.decoder, self.CAE_loss = ConvAE(num_samples,self.env,self)
        return self.encoder, self.CAE_loss


    def load_encoder():
        """ This method loads a convolutional autoencoder trained on the environment specified in self.env """

        self.CAE = tf.keras.models.load_model( os.getcwd()+'/CAE_', self.env ,'_.h5' )
        self.encoder = tf.keras.models.load_model( os.getcwd()+'/encoder', self.env ,'_.h5' )
        self.decoder = tf.keras.models.load_model( os.getcwd()+'/decoder_', self.env ,'_.h5' )
        return self.encoder

    def save_encoder(self.CAE,self.encoder,self.decoder):
        """ saves the CAE """

        CAE.save('CAE_', self.env , '_.h5')
        encoder.save('encoder_', self.env , '_.h5')
        decoder.save('decoder_', self.env , '_.h5')

    def getQvalues(state):
        """ Takes in the state i.e. encoder(observation) and returns an array with length equal to
        the number of actions, where each element specifies the Q value for a given action. """

        Qvalues = np.zeros( (1,self.num_actions) )
        
        for j in range(self.num_actions):
                Qvalues[1,j] = self.myPolicy[j,0].predict(self.myPolicy[j,1].transform(state))
        
        return Qvalues

    def getState(observation):
        """ Takes in the processed image observations (1 observation is 4 frames) received
        from collecting the data in the environment and returns encoder(observation). """

        states = np.zeros( (observation.shape[0], self.state_size ) )

        for i in range(observation.shape[0]):
            states[i,:] = self.encoder.predict(observation[i,:,:,:]).flatten()[np.newaxis,:]
        
        return states

    def getAction(observation):
        """ Input: state is observation
            Output: if encoder is none outputs random action for random policy,
            else returns action from epsilon greedy policy."""

        if self.encoder == None:
            return np.random.randint(0,high=4)
        else:
            action = None
            state = self.encoder.predict( observation )
            Qvalues = getQvalues(state)
            probability = np.random.random_sample()

            if self.epsilon <= probability:
                action = np.argmax(Qvalues)
            else:
                action = np.random.randint(0,high=4)
            return action
        
    
    def imporve_policy():

    



num_episodes = 100
agent = myAgent()
encoder = agent.load_encoder

for episodes in range(num_episodes):

    observation,actions,rewards,num_episodes = collectData('episodes',1000,'Breakout-v0',agent)
    states = getState(observation,encoder)
    imporve_policy()


    



