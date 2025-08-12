# clustering

Research, benchmarking and optimising clustering algorithms for GPU integration.

## Overview

This repository contains a comprehensive implementation and evaluation framework for clustering algorithms, with a focus on GPU-accelerated HDBSCAN. The project compares performance between our custom GPU-enabled HDBSCAN implementation, scikit-learn's HDBSCAN, and DBSCAN across various datasets and parameters.

## Repository Structure

```
clustering/
├── gpu_hdbscan/                 # GPU-accelerated HDBSCAN implementation
│   ├── boruvka/                 # Borůvka's algorithm implementation
│   ├── kd_tree/                 # KD-tree data structure implementation  
│   ├── single_linkage/          # Single linkage clustering components
│   ├── main.cpp                 # Main driver for GPU HDBSCAN
│   └── Makefile                 # Build configuration
├── utils/                       # Utility functions and evaluation tools
│   ├── eval.py                  # Evaluation metrics and analysis
│   └── plot.py                  # Visualization and plotting functions
├── benchmark_integrated.py      # Comprehensive benchmarking suite
├── findEps.py                   # DBSCAN parameter optimization
└── README.md                    # This file
```

## Components

### GPU HDBSCAN Implementation (`gpu_hdbscan/`)

Our custom GPU-accelerated implementation of the HDBSCAN clustering algorithm, organized into modular components:

- **`boruvka/`**: Implementation of Borůvka's minimum spanning tree algorithm for efficient cluster hierarchy construction
- **`kd_tree/`**: KD-tree data structure implementation for fast nearest neighbor searches
- **`single_linkage/`**: Single linkage clustering components used in the hierarchical clustering process
- **`main.cpp`**: Main driver program that coordinates all components
- **`Makefile`**: Build system that compiles all C++ files across directories and produces the `gpu_hdbscan` executable in the `build/` directory

### sk_learn_hdbscan(`sk_learn_hdbscan/`)

Our attempt at converting the Cython code of scikit-learn into Python Code for us to debug at each juncture, the output of the HDBSCAN algorithm with ours, for more comprehensive analysis.

- **`utils/`**: Contains _param_validation.py which validates the data types used in the other python scripts within folder
- **`sk_hdbscan.py`**: Overall wrapper which provides access to HDBSCAN function
- **`recreation.py`**: Implementation of functions used by HDBSCAN

### Utilities (`utils/`)

#### `eval.py`
Evaluation framework providing:
- Clustering quality metrics
- Performance measurement tools
- Statistical analysis functions
- Comparative evaluation between different algorithms

#### `plot.py`
Visualization toolkit for:
- Performance comparison charts
- Clustering result visualizations
- Parameter sensitivity analysis plots
- Comprehensive reporting graphics


### Benchmarking and Evaluation

#### `benchmark_integrated.py`
The main benchmarking script that performs comprehensive performance comparisons between:
- Our GPU-accelerated HDBSCAN
- Scikit-learn's HDBSCAN implementation  
- Scikit-learn's DBSCAN implementation

Features:
- Batched data processing for efficient memory usage
- Integration with evaluation and plotting utilities
- Generates detailed performance metrics and visual comparisons
- Outputs results in both CSV format and comprehensive plots

#### `findEps.py`
Parameter optimization utility for DBSCAN that determines optimal values for:
- `eps` (epsilon): Maximum distance between two samples for them to be considered neighbors
- `min_samples`: Minimum number of samples in a neighborhood for a point to be considered a core point

### Data Analysis and Visualisation

#### `eda.ipynb`
The jupyter notebook which contains code we used to visualise the simulated data
## Building and Running

### Prerequisites
- C++ compiler with GPU support (HIP-capable)
- Python 3.x
- Required Python packages: scikit-learn, numpy, pandas, matplotlib as stated in requirements.txt
- CUDA toolkit (for GPU acceleration)

### Building GPU HDBSCAN
```bash
cd gpu_hdbscan
make
```
This creates a `build/` directory containing the `gpu_hdbscan` executable.

### Running Benchmarks
```bash
# Run comprehensive benchmarking
python benchmark_integrated.py

# Find optimal DBSCAN parameters
python findEps.py
```

## Features

- **GPU Acceleration**: Leverages GPU computing for significant performance improvements in large-scale clustering tasks
- **Comprehensive Benchmarking**: Systematic comparison across multiple algorithms and datasets
- **Modular Design**: Clean separation of concerns with reusable components
- **Visualization Tools**: Rich plotting capabilities for result analysis and presentation
- **Parameter Optimization**: Automated parameter tuning for optimal clustering performance

## Research Focus

This project focuses on:
- Optimizing clustering algorithms for GPU architectures
- Comparative analysis of clustering algorithm performance
- Scalability improvements for large-scale datasets
- Development of efficient hierarchical clustering methods

## Output

The benchmarking suite generates:
- Performance comparison CSV files
- Visual plots showing algorithm comparisons
- Statistical analysis of clustering quality
- Execution time and memory usage metrics