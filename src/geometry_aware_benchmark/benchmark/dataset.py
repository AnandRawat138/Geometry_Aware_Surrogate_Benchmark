import pandas as pd
import numpy as np
from datetime import datetime
from ..geometry.stl_loader import STLLoader
from ..geometry.descriptors import GeometryDescriptors
from .response_generator import BenchmarkResponseGenerator

class DatasetGenerator:
    """
    Generates synthetic benchmark datasets.
    """
    def __init__(self, config_path=None):
        self.generator = BenchmarkResponseGenerator(config_path)
        
    def generate_dataset(self, num_geometries=50, thetas_deg=None, frequencies_ghz=None, seed=42):
        """
        Generates a full dataset spanning multiple geometries, angles, and frequencies.
        """
        if thetas_deg is None:
            # Default: 0 to 180 degrees with 1 degree resolution
            thetas_deg = np.arange(0, 181, 1)
            
        if frequencies_ghz is None:
            # Default: just 10 GHz
            frequencies_ghz = [10.0]
            
        np.random.seed(seed)
        
        records = []
        sample_id = 0
        
        for geom_idx in range(num_geometries):
            geometry_id = f"geom_{geom_idx:04d}"
            
            # Since we don't have real STLs, we simulate slightly different 
            # dummy geometries by seeding the mock loader differently.
            # In a real scenario, this would loop over actual CAD files.
            stl = STLLoader().load()
            # perturb vertices slightly to create variance
            stl.vertices += np.random.randn(*stl.vertices.shape) * 2.0
            
            descriptors = GeometryDescriptors.extract(stl)
            
            # Add material params (mocked as constant per geometry here)
            material_params = {
                'coating_thickness_m': np.random.uniform(0.01, 0.05),
                'loss_tangent': np.random.uniform(0.1, 0.4)
            }
            
            for freq in frequencies_ghz:
                for theta in thetas_deg:
                    R_val = self.generator.generate_noisy(
                        descriptors, theta, freq, seed=None  # seed already set globally
                    )
                    
                    row = {
                        "geometry_id": geometry_id,
                        "sample_id": sample_id,
                        **descriptors,
                        **material_params,
                        "frequency_ghz": freq,
                        "observation_angle_deg": theta,
                        "benchmark_response": R_val
                    }
                    records.append(row)
                    sample_id += 1
                    
        df = pd.DataFrame(records)
        return df
        
    def save_dataset(self, df, filepath, dataset_type="reconstructed_benchmark"):
        """
        Saves dataset as Parquet with metadata.
        """
        # We can write CSV or Parquet. We'll write both or just CSV if parquet is missing, 
        # but pandas has to_parquet if pyarrow is installed. 
        # To be safe with minimal dependencies, we write to CSV and a metadata JSON.
        df.to_csv(filepath, index=False)
        
        metadata = {
            "dataset_type": dataset_type,
            "seed": 42,
            "frequency_ghz": 10,
            "angular_resolution_deg": 1,
            "source": "paper-derived reconstruction",
            "historical_original_data": False,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(df)
        }
        
        meta_filepath = filepath.replace(".csv", "_metadata.json")
        import json
        with open(meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
