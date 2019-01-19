from VAE2 import*
from sklearn.linear_model import LinearRegression

class myAgent:

    def __init__(self,epsilon=1,num_actions=None,state_size = 400,env = 'Breakout-v0',encoder=None):
        
        if(num_actions==None):
            self.num_actions = 4
        else:
            self.num_actions = num_actions
            
        self.Q = []

        for i in range(num_actions):
            self.Q.append( LinearRegression() )
    
    def create_encoder(num_samples):
        self.encoder = ConvAE(num_samples,self.env)
        return self.encoder


    def load_encoder():
        self.encoder = tf.keras.models.load_model( os.getcwd()+'/encoder.h5' )
        return self.encoder

    def getQvalues(state):

        Qvalues = np.zeros( (1,self.num_actions) )
        
        for j in range(self.num_actions):
                Qvalues[1,j] = self.Q[j].predict(state)
        
        return Qvalues

    def getState(observation):

        states = np.zeros( (observation.shape[0], self.state_size ) )

        for i in range(observation.shape[0]):
            states[i,:] = self.encoder.predict(observation[i,:,:,:]).flatten()[np.newaxis,:]
        
        return states

    def getAction(observation):
        action = None
        probability = np.random.random_sample()
        if self.epsilon <= probability:
            action = np.argmax()[0]
        else:
            action = np.random.randint(0,high=4)
        return action
    
        #def imporve_policy():

    

#num_episodes = 100
#agent = myAgent()
#encoder = agent.load_encoder

#for episodes in range(num_episodes):

#    observation,actions,rewards,num_episodes = collectData(1000,'Breakout-v0',agent)
#    states = getState(observation,encoder)
    



