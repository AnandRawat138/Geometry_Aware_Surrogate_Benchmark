import os
import sys
import pytest
import numpy as np
import pandas as pd

# Adjust path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from geometry_aware_benchmark.geometry.stl_loader import STLLoader
from geometry_aware_benchmark.geometry.descriptors import GeometryDescriptors
from geometry_aware_benchmark.geometry.normalization import FeatureNormalizer
from geometry_aware_benchmark.benchmark.response_generator import BenchmarkResponseGenerator
from geometry_aware_benchmark.benchmark.dataset import DatasetGenerator

def test_stl_loader():
    loader = STLLoader()
    loader.load()
    assert loader.vertices is not None
    assert loader.normals is not None
    assert loader.vertices.shape == (100, 3)

def test_geometry_descriptors():
    loader = STLLoader().load()
    desc = GeometryDescriptors.extract(loader)
    
    assert "length" in desc
    assert "width" in desc
    assert "shape_factor" in desc
    assert "curvature_agg" in desc
    assert desc["length"] > 0
    
def test_feature_normalizer():
    norm = FeatureNormalizer()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_norm = norm.fit_transform(X)
    
    assert np.allclose(np.mean(X_norm, axis=0), 0.0)
    assert np.allclose(np.std(X_norm, axis=0), 1.0)
    
def test_response_generator():
    gen = BenchmarkResponseGenerator()
    
    desc = {
        "length": 10.0,
        "width": 5.0,
        "shape_factor": 1.5,
        "surface_normal_var": 0.2,
        "aspect_ratio": 2.0,
        "curvature_agg": 0.5
    }
    
    # Deterministic test
    R1 = gen.generate_deterministic(desc, theta_deg=45.0, frequency_ghz=10.0)
    R2 = gen.generate_deterministic(desc, theta_deg=45.0, frequency_ghz=10.0)
    assert R1 == R2
    
    # Check frequency dependence
    R3 = gen.generate_deterministic(desc, theta_deg=45.0, frequency_ghz=20.0)
    assert R3 != R1
    
    # Noisy test
    R_noisy1 = gen.generate_noisy(desc, theta_deg=45.0, frequency_ghz=10.0, seed=42)
    R_noisy2 = gen.generate_noisy(desc, theta_deg=45.0, frequency_ghz=10.0, seed=42)
    assert R_noisy1 == R_noisy2 # Deterministic due to seed

def test_dataset_generator():
    gen = DatasetGenerator()
    df = gen.generate_dataset(num_geometries=2, thetas_deg=[0, 10], frequencies_ghz=[10.0])
    
    assert len(df) == 4 # 2 geom * 2 thetas * 1 freq
    assert "geometry_id" in df.columns
    assert "benchmark_response" in df.columns
