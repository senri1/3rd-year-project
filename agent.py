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
        self.CAE, self.encoder, self.decoder, self.CAE_loss = ConvAE(num_samples,self.env,self)
        return self.encoder, self.CAE_loss


    def load_encoder():
        self.CAE = tf.keras.models.load_model( os.getcwd()+'/CAE_', env ,'_.h5' )
        self.encoder = tf.keras.models.load_model( os.getcwd()+'/encoder', env ,'_.h5' )
        self.decoder = tf.keras.models.load_model( os.getcwd()+'/decoder_', env ,'_.h5' )
        return self.encoder

    def save_encoder(self.CAE,self.encoder,self.decoder):
        CAE.save('CAE_', env , '_.h5')
        encoder.save('encoder_', env , '_.h5')
        decoder.save('decoder_', env , '_.h5')

    def getQvalues(state):

        Qvalues = np.zeros( (1,self.num_actions) )
        
        for j in range(self.num_actions):
                Qvalues[1,j] = self.myPolicy[j,0].predict(self.myPolicy[j,1].transform(state))
        
        return Qvalues

    def getState(observation):

        states = np.zeros( (observation.shape[0], self.state_size ) )

        for i in range(observation.shape[0]):
            states[i,:] = self.encoder.predict(observation[i,:,:,:]).flatten()[np.newaxis,:]
        
        return states

    def getAction(observation):

        action = None
        Qvalues = getQvalues(observation)
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


    



