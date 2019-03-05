import numpy as np
import torch

class OLS:

    def __init__(self):
        self.dtype = torch.float
        self.device = torch.device("cpu")
        self.weights = torch.randn(400, 1, device=self.device, dtype=self.dtype, requires_grad=True)

    
    def fit(self,X,Y):
        learning_rate = 1e-6
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        for t in range(500):
            y_pred = X.mm(self.weights)
            loss = (y_pred - Y).pow(2).sum()
            loss.backward()
            with torch.no_grad():
                self.weights -= learning_rate * self.weights.grad
                self.weights.grad.zero_()

    def predict(self,X):
        X = torch.from_numpy(X).float()
        return X.mm(self.weights)

    def getWeights(self):
        return self.weights.detach().numpy()

    def setWeights(self,weight):
        self.weights = torch.from_numpy(weight).float()
    
    def getSquaredError(self,X,Y):
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        y_pred = X.mm(self.weights)
        squared_error = (y_pred - Y).pow(2).sum()
        return squared_error