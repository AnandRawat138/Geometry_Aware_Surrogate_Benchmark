import os
import sys
import json
import numpy as np
import pandas as from_pandas
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from geometry_aware_benchmark.benchmark.dataset import DatasetGenerator
from geometry_aware_benchmark.geometry.normalization import FeatureNormalizer
from geometry_aware_benchmark.models.random_forest import RandomForestSurrogate
from geometry_aware_benchmark.models.gaussian_process import GaussianProcessSurrogate
from geometry_aware_benchmark.models.neural_network import NeuralNetworkSurrogate
from geometry_aware_benchmark.evaluation.metrics import calculate_metrics
from geometry_aware_benchmark.evaluation.extreme_value import evaluate_extreme_value
from geometry_aware_benchmark.optimization.genetic_algorithm import GeneticAlgorithmOptimizer

def main():
    print("Starting Reproduction Pipeline...")
    np.random.seed(42)
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data', 'generated')
    results_dir = os.path.join(base_dir, 'results')
    
    # 1. Dataset Generation
    print("1. Generating Benchmark Dataset...")
    generator = DatasetGenerator()
    df = generator.generate_dataset(num_geometries=20, frequencies_ghz=[10.0]) # Scaled down for speed
    dataset_path = os.path.join(data_dir, 'benchmark_dataset.csv')
    generator.save_dataset(df, dataset_path)
    print(f"Dataset generated with {len(df)} samples.")
    
    # 2. Setup Features
    feature_cols = [
        'length', 'width', 'shape_factor', 'surface_normal_var', 
        'aspect_ratio', 'curvature_agg', 'coating_thickness_m', 
        'loss_tangent', 'frequency_ghz', 'observation_angle_deg'
    ]
    target_col = 'benchmark_response'
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 3. Train / Test Split
    print("3. Splitting and Normalizing Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    df_test = df.iloc[len(X_train):].copy() # Simplistic for keeping track of test df geometry
    
    # Need to keep geometry association for test set for extreme value analysis
    # Let's do it cleanly:
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42)
    X_train = df.iloc[train_idx][feature_cols].values
    y_train = df.iloc[train_idx][target_col].values
    X_test = df.iloc[test_idx][feature_cols].values
    y_test = df.iloc[test_idx][target_col].values
    df_test = df.iloc[test_idx].copy()
    
    normalizer = FeatureNormalizer()
    X_train_norm = normalizer.fit_transform(X_train)
    X_test_norm = normalizer.transform(X_test)
    
    # 4. Train Models
    print("4. Training Surrogate Models...")
    models = {
        'random_forest': RandomForestSurrogate(),
        'gaussian_process': GaussianProcessSurrogate(),
        'neural_network': NeuralNetworkSurrogate()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        # Reduce dataset size for GP if needed to save time
        if name == 'gaussian_process' and len(X_train_norm) > 2000:
            idx = np.random.choice(len(X_train_norm), 2000, replace=False)
            model.fit(X_train_norm[idx], y_train[idx])
        else:
            model.fit(X_train_norm, y_train)
            
        print(f"   Evaluating {name}...")
        y_pred = model.predict(X_test_norm)
        metrics = calculate_metrics(y_test, y_pred)
        
        # Extreme value evaluation
        df_test['benchmark_response'] = y_test
        df_pred = df_test.copy()
        df_pred['benchmark_response'] = y_pred
        ev_metrics, _ = evaluate_extreme_value(df_test, df_pred, ['geometry_id', 'frequency_ghz'])
        
        results[name] = {
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "worst_case_abs_error": ev_metrics["worst_case_abs_error"]
        }
        print(f"      RMSE: {metrics['rmse']:.3f}, R2: {metrics['r2']:.3f}, WC Error: {ev_metrics['worst_case_abs_error']:.3f}")
        
    # 5. Geometry Ablation (Without Geometry Features)
    print("5. Running Geometry Ablation (No Geometry Features)...")
    non_geom_cols = ['coating_thickness_m', 'loss_tangent', 'frequency_ghz', 'observation_angle_deg']
    X_train_ablation = df.iloc[train_idx][non_geom_cols].values
    X_test_ablation = df.iloc[test_idx][non_geom_cols].values
    
    normalizer_ab = FeatureNormalizer()
    X_train_ab_norm = normalizer_ab.fit_transform(X_train_ablation)
    X_test_ab_norm = normalizer_ab.transform(X_test_ablation)
    
    rf_ablation = RandomForestSurrogate()
    rf_ablation.fit(X_train_ab_norm, y_train)
    y_pred_ab = rf_ablation.predict(X_test_ab_norm)
    metrics_ab = calculate_metrics(y_test, y_pred_ab)
    
    df_pred_ab = df_test.copy()
    df_pred_ab['benchmark_response'] = y_pred_ab
    ev_metrics_ab, _ = evaluate_extreme_value(df_test, df_pred_ab, ['geometry_id', 'frequency_ghz'])
    
    results['ablation_no_geometry'] = {
        "rmse": metrics_ab["rmse"],
        "r2": metrics_ab["r2"],
        "worst_case_abs_error": ev_metrics_ab["worst_case_abs_error"]
    }
    
    # 6. Output Metadata and Results
    print("6. Saving run metadata...")
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "seed": 42,
        "n_samples": len(df),
        "results": results
    }
    with open(os.path.join(results_dir, 'run_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    # 7. Verification Against Paper
    print("7. Verifying against paper-reported numbers...")
    with open(os.path.join(base_dir, 'configs', 'paper_reported_results.json'), 'r') as f:
        paper_res = json.load(f)["metrics"]
        
    verification = {}
    for model_name, actual in results.items():
        if model_name in paper_res:
            reported = paper_res[model_name]
            verification[model_name] = {
                "rmse_diff": actual["rmse"] - reported["rmse_db"],
                "r2_diff": actual["r2"] - reported.get("r2_test", actual["r2"]),
                "wc_error_diff": actual["worst_case_abs_error"] - reported["worst_case_error_db"]
            }
            
    with open(os.path.join(results_dir, 'verification_report.json'), 'w') as f:
        json.dump(verification, f, indent=2)
        
    print("Pipeline Complete! Check results/ folder.")
    
if __name__ == "__main__":
    main()
