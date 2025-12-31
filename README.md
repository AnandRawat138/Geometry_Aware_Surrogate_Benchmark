📘 Geometry-Aware Surrogate Benchmark for Extreme-Value Prediction

This repository contains the reference implementation and benchmark framework accompanying the paper:

A Geometry-Aware Surrogate Benchmark for Extreme-Value Prediction in High-Cost Electromagnetic Simulations
Anand Rawat, Sanjeev Kumar
(Under review)

📌 Overview

High-fidelity physics-based simulations such as electromagnetic solvers are computationally expensive, limiting their use in large-scale design exploration and worst-case analysis. This repository provides a reproducible software benchmark for evaluating geometry-aware surrogate models under sparse data and extreme-value prediction regimes.

The benchmark is not intended as a high-fidelity electromagnetic solver replacement. Instead, it serves as a methodological evaluation platform for:

data-efficient surrogate modelling,
geometry-aware feature representations,
extreme-value (worst-case) prediction,
reproducible comparison of learning strategies.

A representative radar cross-section (RCS) case study is included solely as an illustrative high-cost simulation example.

🎯 Scope and Design Philosophy
Benchmark-first, not physics-first
Emphasis on trend consistency and ranking, not absolute accuracy

Modular design to support:
multiple surrogate models,
alternative geometry descriptors,
different optimization or query strategies
Explicit support for sparse-data regimes

🧱 Repository Structure
GeometryAwareSurrogateBenchmark/
│
├── geometry/
│   ├── baseline.stl
│   └── example_geometries/
│
├── features/
│   ├── geometry_features.py
│   └── normalization.py
│
├── surrogate_models/
│   ├── random_forest.py
│   ├── gaussian_process.py
│   └── neural_network.py
│
├── benchmark/
│   ├── dataset_generation.py
│   ├── evaluation_metrics.py
│   └── learning_curve.py
│
├── optimization/
│   └── ga_driver.py
│
├── validation/
│   ├── hfss_comparison.csv
│   └── plot_comparison.py
│
├── figures/
│   ├── learning_curve.png
│   ├── feature_importance.png
│   └── hfss_surrogate_comparison.png
│
├── requirements.txt
├── README.md
└── LICENSE


(Directory contents may evolve as the benchmark is extended.)

Install dependencies:
pip install -r requirements.txt

Python ≥ 3.9 is recommended.

🚀 Running the Benchmark
1️⃣ Feature Extraction
python features/geometry_features.py

2️⃣ Train a Surrogate Model
Example (Random Forest):
python surrogate_models/random_forest.py

3️⃣ Benchmark Evaluation
python benchmark/evaluation_metrics.py

4️⃣ Learning Curve Analysis
python benchmark/learning_curve.py

📊 Validation and Physical Anchoring
A limited high-fidelity electromagnetic solver cross-check using ANSYS HFSS is included to anchor the benchmark to real scattering behaviour.
Single geometry
Single frequency (10 GHz)
Coarse angular resolution (10°)

This validation does not claim quantitative solver equivalence and is intended only for qualitative trend and worst-case direction consistency, consistent with the scope of the benchmark.

👥 Intended Users

This benchmark is intended for:
Machine learning researchers evaluating surrogate models for high-cost simulations
Engineers performing early-stage design-space exploration
Researchers studying extreme-value prediction under sparse data
Educators demonstrating surrogate modelling workflows without heavy simulation cost

🔬 Extending the Benchmark
The framework is designed to be extensible. Users may:
Add new surrogate models
Replace geometry descriptors
Incorporate alternative physics domains (e.g., CFD, acoustics)
Introduce uncertainty-aware sampling strategies
Contributions are welcome.

📄 Related Work and Preprint
A preliminary, application-focused version of this work was released as a preprint:

Anand Rawat, Sanjeev Kumar,
Machine Learning Assisted Worst-Case Radar Cross Section Optimisation Using Geometry-Aware Surrogate Modelling,
Research Square, 2025.
https://doi.org/10.21203/rs.3.rs-8435404/v1

The present repository and manuscript substantially extend and reframe that work as a software benchmark with expanded validation and comparative evaluation.

📜 License

This project is released under the MIT License.
See the LICENSE file for details.

📬 Contact

Anand Rawat
Department of Computer Science and Engineering
Pranveer Singh Institute of Technology, Kanpur, India
📧 anandrawat138a@gmail.com

Reviewer-Friendly Notes

This repository is designed for reproducibility, not maximum performance.
Absolute electromagnetic accuracy is out of scope.
The benchmark prioritizes consistency, transparency, and extensibility.
