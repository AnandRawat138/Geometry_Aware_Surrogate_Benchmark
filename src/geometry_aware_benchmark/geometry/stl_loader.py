import numpy as np

class STLLoader:
    """
    Mock STL Loader.
    Because the original repository and CAD files were lost, this class generates
    synthetic geometry points or loads a dummy representation to allow the pipeline
    to function.
    
    WARNING: RECONSTRUCTED / NOT RECOVERED FROM ORIGINAL REPOSITORY
    """
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.vertices = None
        self.normals = None
        
    def load(self):
        """Mock loading of an STL file."""
        if self.filepath and "error" in self.filepath:
            raise ValueError("Simulated load error.")
            
        # Generate some synthetic points to represent a dummy shape
        np.random.seed(42)  # Use fixed seed for reproducible dummy shapes
        self.vertices = np.random.rand(100, 3) * 20.0  # 100 random points up to 20m
        self.normals = np.random.randn(100, 3)
        self.normals /= np.linalg.norm(self.normals, axis=1)[:, np.newaxis]
        return self
