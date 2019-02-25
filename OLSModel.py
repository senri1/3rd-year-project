import numpy as np 

class OLS:

    def __init__(self):
        self.weights = None
    
    def fit(self,X,Y):
        inverse = np.linalg.pinv(X)
        self.weights = np.matmul(inverse,Y)

    def predict(self,X):
        return np.matmul(X,self.weights)

    def getWeights(self):
        return self.weights

    def setWeights(self,weight):
        self.weights = weight
    
    def getSquaredError(self,X,Y):
        prediction = self.predict(X)
        error = np.abs(prediction-Y)
        squared_error = np.matmul(np.transpose(error),error)
        return squared_error