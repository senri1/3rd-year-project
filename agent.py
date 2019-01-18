from VAE2 import*

class myAgent:

    def __init__(self,epsilon=1,num_actions=None):
        if(num_actions==None):
            self.num_actions = 4
        else:
            self.num_actions = num_actions
    
    def create_encoder(num_samples):
        return convAE(num_samples)


    def load_encoder():
        encoder = tf.keras.models.load_model( os.getcwd()+'/encoder.h5' )
        return encoder

    def Qval(state, weight):
        return np.matmul(weight,state)

    def getState(observation,encoder):

        states = np.zeros( (observation.shape[0],encoder.predict(observation[0,:,:,:]).flatten()[0]) )

        for i in range(observation.shape[0]):
            states[i,:] = encoder.predict(observation[i,:,:,:]).flatten()[np.newaxis,:]
        
        return states

    def getAction(observation):
        action = None
        probability = np.random.random_sample()
        if epsilon >= probability:
            action = np.argmax(Qval)[0]
        else:
            action = np.random.randint(0,high=4)
        return action

    


num_episodes = 100
agent = myAgent()
encoder = agent.load_encoder

for episodes in range(num_episodes):

    observation,actions,rewards,num_episodes = collectData(1000,'Breakout-v0',agent)
    states = getState(observation,encoder)


