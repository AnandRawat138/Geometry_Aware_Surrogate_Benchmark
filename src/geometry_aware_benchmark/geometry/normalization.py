import numpy as np

class FeatureNormalizer:
    """
    Normalizes feature vectors.
    Uses standard zero-mean, unit-variance normalization (Z-score normalization).
    """
    def __init__(self):
        self.means = None
        self.stds = None
        
    def fit(self, X):
        """Fits the normalizer to the training data X (2D numpy array)."""
        X = np.asarray(X)
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        # Prevent division by zero
        self.stds[self.stds == 0] = 1.0
        
    def transform(self, X):
        """Transforms data X."""
        X = np.asarray(X)
        if self.means is None or self.stds is None:
            raise ValueError("Normalizer has not been fitted yet.")
        return (X - self.means) / self.stds
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X_norm):
        if self.means is None or self.stds is None:
            raise ValueError("Normalizer has not been fitted yet.")
        return X_norm * self.stds + self.means
