import subprocess
import numpy as np
import time
import psutil
import os
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for headless environments
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import glob

class GPUHDBSCANWrapper:
    def __init__(self, executable_path="./gpu_hdbscan_edited/build/gpu_hdbscan"):
        self.executable_path = executable_path

    def fit_predict_batched(
        self,
        input_csv_path: str,
        dims: int,
        min_samples: int = 5,
        min_cluster_size: int = 5,
        distance_metric: int = 2,
        minkowski_p: float = 2.0,
        quiet_mode: bool = True
    ) -> np.ndarray:
        """
        Run GPU HDBSCAN directly on a pre-batched CSV file.
        """
        exe_path = os.path.abspath(self.executable_path)
        log_path = os.path.join(os.getcwd(), "log_file.txt")

        # Build the command
        cmd = [
            exe_path,
            "--dimensions",      str(dims),
            "--minpts",          str(min_samples),
            "--input",           input_csv_path,
            "--distMetric",      str(distance_metric),
            "--minclustersize",  str(min_cluster_size),
            "--skip-toa",
            "--skip-amp",
        ]
        if quiet_mode:
            cmd.append("--quiet")
        if distance_metric == 4:
            cmd += ["--minkowskiP", str(minkowski_p)]

        # Execute
        result = subprocess.run(cmd, text=True, timeout=300, capture_output=True)

        # Log stdout/stderr
        with open(log_path, "w") as log_file:
            if result.stdout:
                log_file.write(result.stdout)
            if result.stderr:
                log_file.write(result.stderr)

        # On failure, return all-noise
        if result.returncode != 0:
            print(f"GPU HDBSCAN failed (code {result.returncode}):\n{result.stderr}")
            # Count data rows: total lines minus header
            with open(input_csv_path) as f:
                n = sum(1 for _ in f) - 1
            return np.full(n, -1, dtype=int)

        # On success, parse labels
        with open(input_csv_path) as f:
            n = sum(1 for _ in f) - 1
        return self._parse_output(result.stdout, n)
    
    def _parse_output(self, output, n_points):
        """
        Parse the output from your C++ executable to extract cluster labels
        """
        lines = output.split('\n')
        for line in lines:
            if line.startswith("CLUSTER_LABELS:"):
                labels_str = line.replace("CLUSTER_LABELS:", "").strip()
                labels = list(map(int, labels_str.split()))
                return np.array(labels)
        
        # If no labels found, return all noise
        print("Warning: Could not parse cluster labels from output")
        return np.full(n_points, -1)

def load_dbscan_parameters(params_file: str) -> Dict[str, Tuple[int, float]]:
    """
    Load DBSCAN parameters from text file.
    
    Args:
        params_file: Path to the parameters file
        
    Returns:
        Dictionary mapping filename to (min_samples, epsilon) tuple
    """
    params = {}
    
    try:
        with open(params_file, 'r') as f:
            lines = f.readlines()
            
        # Skip header line
        for line in lines[1:]:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) >= 3:
                    filename = parts[0]
                    min_samples = int(parts[1])
                    epsilon = float(parts[2])
                    params[filename] = (min_samples, epsilon)
                    
    except FileNotFoundError:
        print(f"Parameters file not found: {params_file}")
        return {}
    except Exception as e:
        print(f"Error reading parameters file: {e}")
        return {}
    
    print(f"Loaded parameters for {len(params)} files")
    return params

def track_performance(func, *args, **kwargs):
    """Track execution time and memory usage"""
    process = psutil.Process()
    
    # Get initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Time the execution
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    # Get peak memory (approximation)
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = max(0, final_memory - initial_memory)
    
    execution_time = end_time - start_time
    
    return result, execution_time, memory_used

def plot_clusters_2d(X, labels, feature_names=None, title="Clustering Results", 
                     save_path=None):
    """
    Plot 2D clustering results using the first two features
    """
    # Use first two dimensions
    X_2d = X[:, :2]
    
    # Set default feature names
    if feature_names is None:
        feature_names = ['Feature 1', 'Feature 2']
    
    # Get unique clusters (excluding noise if present)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_count = np.sum(labels == -1)
    
    print(f"Found {n_clusters} clusters and {noise_count} noise points")
    
    # Create matplotlib plot
    plt.figure(figsize=(12, 8))
    
    # Create color map
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Plot noise points
            mask = labels == label
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c='black', marker='x', s=50, alpha=0.6, label='Noise')
        else:
            # Plot cluster points
            mask = labels == label
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=[colors[i]], s=60, alpha=0.7, label=f'Cluster {label}')
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(f"{title}\nClusters: {n_clusters}, Noise points: {noise_count}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def evaluate_clustering_quality(X, labels, feature_names=None):
    """
    Evaluate clustering quality using multiple metrics
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Remove noise points for evaluation
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_count = np.sum(labels == -1)
    total_points = len(labels)
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise_points': noise_count,
        'noise_ratio': noise_count / total_points,
        'total_points': total_points
    }
    
    # Calculate metrics only if we have valid clusters
    if n_clusters > 1 and len(labels_clean) > 0:
        try:
            metrics['silhouette_score'] = silhouette_score(X_clean, labels_clean)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_clean, labels_clean)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_clean, labels_clean)
        except Exception as e:
            print(f"Warning: Could not calculate some metrics: {e}")
            metrics['silhouette_score'] = None
            metrics['calinski_harabasz_score'] = None
            metrics['davies_bouldin_score'] = None
    else:
        metrics['silhouette_score'] = None
        metrics['calinski_harabasz_score'] = None
        metrics['davies_bouldin_score'] = None
    
    return metrics

def run_dbscan_vs_gpu_hdbscan_benchmark(
    batch_data_path: str,
    dbscan_params_file: str,
    executable_path: str,
    use_amp: bool = False,
    use_toa: bool = False,
    min_cluster_size: int = 50  # For GPU HDBSCAN
):
    """
    Run DBSCAN vs GPU HDBSCAN benchmark using parameters from file
    
    Args:
        batch_data_path: Path to folder containing batch CSV files
        dbscan_params_file: Path to file containing DBSCAN parameters
        executable_path: Path to GPU HDBSCAN executable
        use_amp: Whether to include amplitude feature
        use_toa: Whether to include TOA feature
        min_cluster_size: Minimum cluster size for GPU HDBSCAN
    """
    # Define output folder
    output_dir = "dbscan_benchmark_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load DBSCAN parameters
    dbscan_params = load_dbscan_parameters(dbscan_params_file)
    if not dbscan_params:
        print("No DBSCAN parameters loaded. Exiting.")
        return
    
    # Initialize GPU HDBSCAN wrapper
    gpu_hdbscan = GPUHDBSCANWrapper(executable_path=executable_path)
    
    # Results storage
    results = []
    
    # Get all CSV files in batch directory
    if not os.path.isdir(batch_data_path):
        print(f"Batch data directory not found: {batch_data_path}")
        return
    
    csv_files = sorted(glob.glob(os.path.join(batch_data_path, "*.csv")))
    if not csv_files:
        print("No CSV files found in batch directory")
        return
    
    # Process each batch file
    for csv_file in csv_files:
        batch_filename = os.path.basename(csv_file)
        batch_name = os.path.splitext(batch_filename)[0]
        
        # Check if we have parameters for this file
        if batch_filename not in dbscan_params:
            print(f"No parameters found for {batch_filename}, skipping...")
            continue
        
        min_samples, epsilon = dbscan_params[batch_filename]
        
        # Count data rows
        with open(csv_file, 'r') as f:
            n_samples = sum(1 for _ in f) - 1
        
        print(f"\n=== Processing {batch_name} ({n_samples} samples) ===")
        print(f"DBSCAN Parameters: min_samples={min_samples}, epsilon={epsilon}")
        
        # Load data
        df = pd.read_csv(csv_file)
        feature_cols = ['PW(microsec)', 'FREQ(MHz)', 'AZ_S0(deg)', 'EL_S0(deg)']
        if use_amp:
            feature_cols.append('Amp_S0(dBm)')
        if use_toa:
            feature_cols.append('TOA(ns)')
        
        X = df[feature_cols].to_numpy()
        
        # Standardize features for DBSCAN
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run DBSCAN
        print("  -> Running DBSCAN...")
        dbscan_model = DBSCAN(eps=epsilon, min_samples=min_samples)
        dbscan_labels, dbscan_time, dbscan_memory = track_performance(
            dbscan_model.fit_predict, X_scaled
        )
        
        # Run GPU HDBSCAN (using min_samples from DBSCAN params)
        print("  -> Running GPU HDBSCAN...")
        dims = X.shape[1]
        gpu_labels, gpu_time, gpu_memory = track_performance(
            gpu_hdbscan.fit_predict_batched, csv_file,
            dims,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            quiet_mode=True
        )
        
        # Evaluate clustering quality for both methods
        print("  -> Evaluating clustering quality...")
        dbscan_metrics = evaluate_clustering_quality(X_scaled, dbscan_labels, feature_cols)
        gpu_metrics = evaluate_clustering_quality(X, gpu_labels, feature_cols)
        
        # Create visualizations
        print("  -> Creating visualizations...")
        
        # DBSCAN visualization
        plot_clusters_2d(
            X_scaled, dbscan_labels,
            feature_names=feature_cols[:2],
            title=f"DBSCAN: {batch_name} (eps={epsilon}, min_samples={min_samples})",
            save_path=os.path.join(output_dir, f"dbscan_{batch_name}.png")
        )
        
        # GPU HDBSCAN visualization
        plot_clusters_2d(
            X, gpu_labels,
            feature_names=feature_cols[:2],
            title=f"GPU HDBSCAN: {batch_name} (min_samples={min_samples})",
            save_path=os.path.join(output_dir, f"gpu_hdbscan_{batch_name}.png")
        )
        
        # Collect results
        result = {
            'Batch': batch_name,
            'Samples': n_samples,
            'DBSCAN_Epsilon': epsilon,
            'MinSamples': min_samples,
            'MinClusterSize': min_cluster_size,
            
            # Performance metrics
            'DBSCAN_Time': dbscan_time,
            'GPU_HDBSCAN_Time': gpu_time,
            'DBSCAN_Memory': dbscan_memory,
            'GPU_HDBSCAN_Memory': gpu_memory,
            'Speedup_Factor': dbscan_time / gpu_time if gpu_time > 0 else 0,
            
            # Clustering results
            'DBSCAN_Clusters': dbscan_metrics['n_clusters'],
            'GPU_HDBSCAN_Clusters': gpu_metrics['n_clusters'],
            'DBSCAN_Noise_Points': dbscan_metrics['n_noise_points'],
            'GPU_HDBSCAN_Noise_Points': gpu_metrics['n_noise_points'],
            'DBSCAN_Noise_Ratio': dbscan_metrics['noise_ratio'],
            'GPU_HDBSCAN_Noise_Ratio': gpu_metrics['noise_ratio'],
            
            # Quality metrics
            'DBSCAN_Silhouette': dbscan_metrics['silhouette_score'],
            'GPU_HDBSCAN_Silhouette': gpu_metrics['silhouette_score'],
            'DBSCAN_Calinski_Harabasz': dbscan_metrics['calinski_harabasz_score'],
            'GPU_HDBSCAN_Calinski_Harabasz': gpu_metrics['calinski_harabasz_score'],
            'DBSCAN_Davies_Bouldin': dbscan_metrics['davies_bouldin_score'],
            'GPU_HDBSCAN_Davies_Bouldin': gpu_metrics['davies_bouldin_score'],
        }
        
        results.append(result)
        
        # Print batch summary
        print(f"  -> DBSCAN: {dbscan_metrics['n_clusters']} clusters, {dbscan_metrics['n_noise_points']} noise points")
        print(f"  -> GPU HDBSCAN: {gpu_metrics['n_clusters']} clusters, {gpu_metrics['n_noise_points']} noise points")
        print(f"  -> Time: DBSCAN {dbscan_time:.3f}s, GPU HDBSCAN {gpu_time:.3f}s")
        print(f"  -> Speedup: {dbscan_time / gpu_time:.2f}x" if gpu_time > 0 else "  -> Speedup: N/A")
    
    # Create results DataFrame and save
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'dbscan_vs_gpu_hdbscan_benchmark.csv'), index=False)
        
        # Print summary
        print("\n" + "="*80)
        print("DBSCAN vs GPU HDBSCAN BENCHMARK SUMMARY")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Create summary plots
        create_comparison_plots(results_df, output_dir)
        
        return results_df
    else:
        print("No results to save.")
        return None

def create_comparison_plots(df, output_dir):
    """Create comparison plots between DBSCAN and GPU HDBSCAN"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    batches = df['Batch'].values
    x = np.arange(len(batches))
    width = 0.35
    
    # Time comparison
    dbscan_times = df['DBSCAN_Time'].values
    gpu_times = df['GPU_HDBSCAN_Time'].values
    
    axes[0, 0].bar(x - width/2, dbscan_times, width, label='DBSCAN', alpha=0.8)
    axes[0, 0].bar(x + width/2, gpu_times, width, label='GPU HDBSCAN', alpha=0.8)
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_title('Execution Time Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(batches, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Memory comparison
    dbscan_memory = df['DBSCAN_Memory'].values
    gpu_memory = df['GPU_HDBSCAN_Memory'].values
    
    axes[0, 1].bar(x - width/2, dbscan_memory, width, label='DBSCAN', alpha=0.8)
    axes[0, 1].bar(x + width/2, gpu_memory, width, label='GPU HDBSCAN', alpha=0.8)
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Memory (MB)')
    axes[0, 1].set_title('Memory Usage Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(batches, rotation=45)
    axes[0, 1].legend()
    
    # Clusters comparison
    dbscan_clusters = df['DBSCAN_Clusters'].values
    gpu_clusters = df['GPU_HDBSCAN_Clusters'].values
    
    axes[1, 0].bar(x - width/2, dbscan_clusters, width, label='DBSCAN', alpha=0.8)
    axes[1, 0].bar(x + width/2, gpu_clusters, width, label='GPU HDBSCAN', alpha=0.8)
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Number of Clusters')
    axes[1, 0].set_title('Number of Clusters Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(batches, rotation=45)
    axes[1, 0].legend()
    
    # Speedup visualization
    speedup_values = df['Speedup_Factor'].values
    
    axes[1, 1].bar(x, speedup_values, width, alpha=0.8, color='green')
    axes[1, 1].set_xlabel('Batch')
    axes[1, 1].set_ylabel('Speedup Factor')
    axes[1, 1].set_title('GPU HDBSCAN Speedup over DBSCAN')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(batches, rotation=45)
    axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dbscan_vs_gpu_hdbscan_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Configuration
    executable_path = "./gpu_hdbscan_edited/build/gpu_hdbscan"
    batch_data_path = "./data/batch_data"
    dbscan_params_file = "dbscan_params.txt"
    
    # Check if executable exists
    if not os.path.exists(executable_path):
        print(f"Executable not found at {executable_path}")
        print("Please build the project first")
        exit(1)
    
    # Check if batch data directory exists
    if not os.path.exists(batch_data_path):
        print(f"Batch data directory not found at {batch_data_path}")
        exit(1)
    
    # Check if parameters file exists
    if not os.path.exists(dbscan_params_file):
        print(f"DBSCAN parameters file not found at {dbscan_params_file}")
        exit(1)
    
    # Run the benchmark
    print("Starting DBSCAN vs GPU HDBSCAN benchmark...")
    results = run_dbscan_vs_gpu_hdbscan_benchmark(
        batch_data_path=batch_data_path,
        dbscan_params_file=dbscan_params_file,
        executable_path=executable_path,
        use_amp=False,
        use_toa=False,
        min_cluster_size=50
    )
    
    if results is not None:
        print(f"\nBenchmark complete! Results saved to 'dbscan_benchmark_outputs/'")
        print("Check the output directory for:")
        print("- dbscan_vs_gpu_hdbscan_benchmark.csv (detailed results)")
        print("- dbscan_vs_gpu_hdbscan_comparison.png (comparison plots)")
        print("- Individual cluster visualizations for each batch")
    else:
        print("Benchmark failed or no results generated.")