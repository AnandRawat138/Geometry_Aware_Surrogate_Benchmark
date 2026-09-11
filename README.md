# Geometry-Aware Surrogate Benchmark

> **Note**: This repository reconstructs the benchmark methodology described in the associated manuscript *"A Geometry-Aware Surrogate Benchmark for Extreme-Value Prediction in High-Cost Electromagnetic Simulations"*. Where the original implementation or historical dataset could not be recovered, the corresponding components are explicitly identified as reconstructed and are not claimed to reproduce the original software artifact exactly.

## Project Overview

This repository provides a modular, geometry-aware surrogate modelling benchmark. It simulates the extraction of geometric features from 3D models (STL) and synthesizes realistic, physics-inspired extreme-value responses to test the robustness of machine learning surrogate models (Random Forest, Gaussian Process, Neural Networks) under sparse data conditions.

## Architecture

1. **Geometry Processing**: Extracts `length`, `width`, `shape_factor`, `curvature_agg` from 3D shapes. *(Reconstructed from dummy shapes)*
2. **Benchmark Generator**: Synthesizes a deterministic and noisy response $R(z)$ based on geometric and physical inputs.
3. **Surrogate Modelling**: Common interface for Random Forest, Gaussian Processes, and Neural Networks.
4. **Optimization**: Genetic Algorithm for querying the surrogate model to locate extreme values.

## Installation

This project requires Python 3.12+.

```bash
git clone <repository>
cd Geometry_Aware_Surrogate_Benchmark
pip install -e .
```

## Quick Start & Reproducibility

To run the entire pipeline, generate the benchmark dataset, train all models, run the ablation study, and compute the extreme-value fidelity metrics against the paper's reported numbers:

```bash
python scripts/reproduce_all.py
```

Results and metadata are written to `results/`.

## Testing

Run the test suite with `pytest`:
```bash
pytest tests/
```

## Data Format

Generated datasets are stored in `data/generated/`. They include geometric descriptors, material properties, observation angles, frequencies, and the target benchmark response.

## HFSS Validation

If an original HFSS dataset is available, it should be placed in `data/raw/hfss/`. 
The required format is CSV with columns: `geometry_id`, `frequency_ghz`, `angle_deg`, `rcs_dbsm`.
*(Note: Do not fabricate HFSS data. If missing, this module remains dormant).*
