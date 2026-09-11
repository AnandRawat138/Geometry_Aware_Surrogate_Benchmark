from .base import BaseSurrogateModel
from sklearn.ensemble import RandomForestRegressor

class RandomForestSurrogate(BaseSurrogateModel):
    """
    Random Forest Surrogate Model.
    
    Paper specifies:
    - n_estimators = 200
    - random_state = 42
    """
    def __init__(self, n_estimators=200, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
