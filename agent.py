from VAE2 import*
from sklearn.linear_model import LinearRegression
import pickle

class myAgent:

    def __init__(
        self,
        epsilon=1,
        disc_factor = 0.99,
        num_actions=4,
        state_size = 400,
        env = 'Breakout-v0',
        encoder=None,
        decoder=None,
        CAE=None,
        policyType = 'lr'
        ):
        self.epsilon = epsilon
        self.disc_factor = disc_factor
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
        self.mean = None
        self.sd = None
    
    def create_encoder(self,num_samples):
        """ This methoed creates a convolutional autoencoder trained on the agents current environment
        specified in self.env for samples (1 sample is 4 frames) specified in num_samples and returns the encoder
        and the training and validation loss for each epoch."""

        self.CAE, self.encoder, self.decoder, self.CAE_loss, self.mean, self.sd = ConvAE(num_samples,self.env,self)
        return self.encoder, self.CAE_loss


    def load_encoder(self):
        """ This method loads a convolutional autoencoder trained on the environment specified in self.env, as well as
            the mean and standard deviation of the states obtained from the training data used to train the CAE. """

        self.CAE = tf.keras.models.load_model( os.getcwd() + '/CAE_' + self.env + '.h5' )
        self.encoder = tf.keras.models.load_model( os.getcwd() + '/encoder_' + self.env + '.h5')
        self.decoder = tf.keras.models.load_model( os.getcwd() + '/CAE_' + self.env + '.h5' )
        self.mean = np.load('mean.npy')
        self.sd = np.load('sd.npy')
        return self.encoder

    def save_encoder(self):
        """ saves the CAE as well as the mean and standard deviation of states obtained from the data it was trained on."""

        self.CAE.save('CAE_' + self.env + '.h5')
        self.encoder.save('encoder_' + self.env + '.h5')
        self.decoder.save('decoder_' + self.env + '.h5')
        np.save('mean',self.mean)
        np.save('sd',self.sd)

    def save_policy(self):
        with open("policies.pckl", "wb") as f:
            for policy in self.myPolicy:
                pickle.dump(policy, f)
    
    def load_policy(self):

        self.myPolicy = []
        with open("policies.pckl", "rb") as f:
            while True:
                try:
                    self.myPolicy.append(pickle.load(f))
                except EOFError:
                    break

    def getQvalues(self,state):

        """ Input: state can be shape (1,5,5,16) or (1,400).
            
            Output: Array with Q values for each action. """

        Qvalues = np.zeros( (1,self.num_actions) )
        for j in range(self.num_actions):

                # Use linear model corresponding to the action to get its Q value
                Qvalues[0,j] = self.myPolicy[j][0].predict(state.reshape((1,-1)))
        
        return Qvalues

    def getAction(self,frames):

        """ Input: frames is an input of shape (1,84,84,4)

            Output: action, an integer that can be 0,1,2,3"""

        # If an encoder has not been made, take random action.
        if self.encoder == None:
            return np.random.randint(0,high=4)
        
        # Return action using agents epsilon greedy policy
        else:
            # Use encoder to get state of shape (1,5,5,16) the convert to shape (1,400) 
            state = self.encoder.predict(frames)
            stateMean0 = standardize(state)

            # Standardize then get list of Q values for each action and generate random probability
            stateMean0 = states - self.mean
            states = np.divide(stateMean0, self.sd, out=np.zeros_like(stateMean0), where=self.sd!=0)
            Qvalues = self.getQvalues(state)
            probability = np.random.random_sample()

            # Take random action with probability epsilon
            if self.epsilon <= probability:
                action = np.argmax(Qvalues)
            else:
                action = np.random.randint(0,high=4)
            return action
    
    def getState(self,observation):
        """ Input: Observation is size (steps,84,84,1)
            Output: states size (steps+4,400) """

        # Don't include first 4 observations, since they were used to initialise an action
        # when the observations were collected.
        numStates = observation.shape[0] - 4

        # Encoder takes in 4 frames of size (_,84,84,4)
        frames = np.zeros( ( numStates ,84,84,4  ) )

        # Convert observations into frames, by sliding along the first axis of observation
        # with size 4 and step 1.
        for i in range(numStates):
                frames[i:i+1,:,:,:] = Img2Frame( observation[i+1:i+5,:,:,:] ) 
        
        # Use encoder to get states from frames.
        # Convert states from shape (numStates,5,5,16) to (numStates,400)
        states = self.encoder.predict(frames)
        _,_,states = standardize(states)

        # Standardize the states using the mean and standard deviation of the states that were
        # from the observations used to train the convolutional autoencoder.
        # Also replace any feature with 0 standard deviation with value 0. 
        statesMean0 = states - self.mean
        states = np.divide(statesMean0, self.sd, out=np.zeros_like(statesMean0), where=self.sd!=0)
        
        return states

    def improve_policy(self,states,actions,rewards):

        """ Input: states is output of encoder(observation) with size (steps-4,400), 
                   actions and rewards are arrays with size (steps-4,1)
            
            Output: improved policy  
        """
       
        # create array to keep track of the number of time each action has occured
        actionCount = np.zeros((1,self.num_actions),dtype=np.uint32)
        
        # X: will contain #ofActions arrays, each array will contain states of size (1,400),
        # the array a state is stored in will depend on the action made at that state. 
        #
        # Y: will contain #ofActions arrays of the target value: r(s) + y*Q(s'), the array
        # a target value is stored in is based on the action made in the state for which
        # we are predicting the target value.
        Y = []
        X = []

        # create arrays for training data for each action, each with size equal to the 
        # number of times the action occured.
        for j in range(self.num_actions):
            X.append( np.zeros( ( np.count_nonzero(actions==j),400 ) ) )
            Y.append( np.zeros( ( np.count_nonzero(actions==j),1 ) ) )

        # populate each array corresponding to the action taken
        for i in range(actions.shape[0]-1):
            idx = actionCount[0,actions[i,0]]
            X[actions[i,0]][idx,:] = states[i,:]
            Y[actions[i,0]][idx,0] = rewards[i] + self.disc_factor * np.max(self.getQvalues(states[i+1]))
            actionCount[0,actions[i,0]] += 1
        
        # For each action fit a linear model with squared error loss: ( w.s - r(s) + y*Q(s') )^2.
        # Don't fit if there is no data, in the case an action was never taken.
        for n in range(self.num_actions):
            if(actionCount[0,n] != 0):           
                self.myPolicy[n][0].fit(X[n],Y[n])
        
        return self.myPolicy
    






    



