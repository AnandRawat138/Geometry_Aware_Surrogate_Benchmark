import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Generating Figures...")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data', 'generated')
    fig_dir = os.path.join(base_dir, 'results', 'figures')
    
    os.makedirs(fig_dir, exist_ok=True)
    
    dataset_path = os.path.join(data_dir, 'benchmark_dataset.csv')
    if not os.path.exists(dataset_path):
        print("Dataset not found. Run reproduce_all.py first.")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Figure: Angular Response for a single geometry
    geom_id = df['geometry_id'].iloc[0]
    sub_df = df[(df['geometry_id'] == geom_id) & (df['frequency_ghz'] == 10.0)]
    
    if len(sub_df) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(sub_df['observation_angle_deg'], sub_df['benchmark_response'], label='Reconstructed Response R(z)')
        plt.xlabel("Observation Angle (Degrees)")
        plt.ylabel("Benchmark Response R(z)")
        plt.title(f"Reconstructed Angular Response for {geom_id}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'angular_response.png'))
        plt.savefig(os.path.join(fig_dir, 'angular_response.pdf'))
        plt.close()
        
    # Figure: Learning Curve mockup
    # In a real scenario we'd track train vs validation scores over multiple sample sizes
    sizes = [100, 500, 1000, 2000, 3000]
    rmse_scores = [3.5, 1.8, 1.2, 0.8, 0.5] # Synthetic for display
    
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, rmse_scores, marker='o', linestyle='-', color='r', label='Validation RMSE')
    plt.xlabel("Number of Training Samples")
    plt.ylabel("RMSE (dB)")
    plt.title("Synthetic Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'learning_curve.png'))
    plt.savefig(os.path.join(fig_dir, 'learning_curve.pdf'))
    plt.close()
    
    print("Figures generated in results/figures/")

if __name__ == "__main__":
    main()
