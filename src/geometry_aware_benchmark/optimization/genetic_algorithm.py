import numpy as np

class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm query generator / optimization driver.
    Optimizes the surrogate-predicted response.
    
    Paper specifies:
    - population = 50
    - generations = 100
    - crossover = 0.80
    - mutation = 0.10
    - random_seed = 42
    """
    def __init__(self, surrogate_model, feature_bounds, config_path=None):
        """
        surrogate_model: Trained surrogate model.
        feature_bounds: List of tuples [(min, max), (min, max)...] for each feature.
        """
        self.surrogate = surrogate_model
        self.bounds = np.array(feature_bounds)
        self.num_features = len(feature_bounds)
        
        # In a real impl we would load from config, using defaults for now
        self.pop_size = 50
        self.generations = 100
        self.pc = 0.80
        self.pm = 0.10
        self.seed = 42
        
        self.history = []
        
    def optimize(self):
        np.random.seed(self.seed)
        
        # Initialize population
        population = np.random.uniform(
            low=self.bounds[:, 0], 
            high=self.bounds[:, 1], 
            size=(self.pop_size, self.num_features)
        )
        
        best_candidate = None
        best_fitness = -np.inf
        
        for gen in range(self.generations):
            # Evaluate fitness (maximize response)
            fitness = self.surrogate.predict(population)
            
            # Tracking
            gen_best_idx = np.argmax(fitness)
            gen_best_fit = fitness[gen_best_idx]
            
            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_candidate = population[gen_best_idx].copy()
                
            self.history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "mean_fitness": np.mean(fitness),
                "best_candidate": best_candidate.copy()
            })
            
            # Selection (Tournament)
            new_population = []
            for _ in range(self.pop_size):
                idx1, idx2 = np.random.randint(0, self.pop_size, 2)
                winner = idx1 if fitness[idx1] > fitness[idx2] else idx2
                new_population.append(population[winner])
            new_population = np.array(new_population)
            
            # Crossover (Single point)
            for i in range(0, self.pop_size - 1, 2):
                if np.random.rand() < self.pc:
                    pt = np.random.randint(1, self.num_features)
                    temp = new_population[i, pt:].copy()
                    new_population[i, pt:] = new_population[i+1, pt:]
                    new_population[i+1, pt:] = temp
                    
            # Mutation
            for i in range(self.pop_size):
                if np.random.rand() < self.pm:
                    mut_idx = np.random.randint(0, self.num_features)
                    new_population[i, mut_idx] = np.random.uniform(
                        self.bounds[mut_idx, 0], self.bounds[mut_idx, 1]
                    )
                    
            population = new_population
            
        return best_candidate, best_fitness
