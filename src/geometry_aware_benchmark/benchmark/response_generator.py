import numpy as np
import yaml
import os

class BenchmarkResponseGenerator:
    """
    Physics-inspired benchmark response generator.
    
    Generates deterministic benchmark response:
    R(z) = sum(wi*gi) + alpha*sin(k*theta) + beta*cos(2*k*theta) + gamma*f + delta*exp(-lambda*C)
    
    And noisy response:
    R*(z) = R(z) + epsilon, where epsilon ~ N(0, sigma^2)
    """
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.sigma = self.config['benchmark'].get('noise_sigma', 0.1)
        self.alpha = self.config['benchmark']['alpha']
        self.beta = self.config['benchmark']['beta']
        self.gamma = self.config['benchmark']['gamma']
        self.delta = self.config['benchmark']['delta']
        self.lambda_param = self.config['benchmark']['lambda_param']
        self.k = self.config['benchmark']['k_param']
        self.weights = self.config['benchmark']['weights']
        
    def _load_config(self, config_path):
        if config_path is None:
            # Default to the one in configs/
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            config_path = os.path.join(base_dir, 'configs', 'benchmark_constants.yaml')
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def generate_deterministic(self, descriptors, theta_deg, frequency_ghz):
        """
        Calculates the deterministic benchmark response R(z).
        """
        # sum(wi*gi)
        geom_sum = 0.0
        for key, weight in self.weights.items():
            if key in descriptors:
                geom_sum += weight * descriptors[key]
                
        C = descriptors.get('curvature_agg', 0.0)
        
        # Angle dependencies
        angle_term = self.alpha * np.sin(self.k * theta_deg) + self.beta * np.cos(2 * self.k * theta_deg)
        
        # Frequency term
        freq_term = self.gamma * frequency_ghz
        
        # Curvature term
        curv_term = self.delta * np.exp(-self.lambda_param * C)
        
        return geom_sum + angle_term + freq_term + curv_term
        
    def generate_noisy(self, descriptors, theta_deg, frequency_ghz, seed=None):
        """
        Calculates the noisy benchmark response R*(z).
        """
        R_z = self.generate_deterministic(descriptors, theta_deg, frequency_ghz)
        
        if seed is not None:
            np.random.seed(seed)
            
        noise = np.random.normal(0, self.sigma) if self.sigma > 0 else 0.0
        return R_z + noise
