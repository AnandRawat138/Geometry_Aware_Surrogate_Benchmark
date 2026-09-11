from abc import ABC, abstractmethod
import pickle
import joblib

class BaseSurrogateModel(ABC):
    """
    Base interface for surrogate models.
    """
    
    @abstractmethod
    def fit(self, X, y):
        pass
        
    @abstractmethod
    def predict(self, X):
        pass
        
    def evaluate(self, X, y):
        """Returns RMSE and R2 on the given data."""
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np
        
        preds = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        return rmse, r2
        
    def save(self, filepath):
        """Saves the model to disk."""
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath):
        """Loads a model from disk."""
        return joblib.load(filepath)
