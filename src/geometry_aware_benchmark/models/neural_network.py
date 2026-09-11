from .base import BaseSurrogateModel
from sklearn.neural_network import MLPRegressor

class NeuralNetworkSurrogate(BaseSurrogateModel):
    """
    Feed-forward Neural Network / MLP Surrogate Model.
    
    WARNING: RECONSTRUCTED / NOT RECOVERED FROM ORIGINAL REPOSITORY
    Exact architecture not specified. 
    Using standard MLP with (100, 50) hidden layers.
    """
    def __init__(self, random_state=42):
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=random_state,
            early_stopping=True
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
