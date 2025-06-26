import subprocess
import numpy as np
import time
import psutil
import os
import tempfile
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
import pandas as pd

class GPUHDBSCANWrapper:
    def __init__(self, executable_path="./gpu_hdbscan/build/gpu_hdbscan"):
        self.executable_path = executable_path
        
    def fit_predict(self, X, min_samples=5, min_cluster_size=5, distance_metric=2, minkowski_p=2.0):
        """
        Wrapper to call GPU HDBSCAN executable
        """
        # Create temporary file for input data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write data in the format your C++ code expects
            for point in X:
                f.write(' '.join(map(str, point)) + '\n')
            temp_input = f.name
        
        try:
            # Prepare command arguments
            cmd = [
                self.executable_path,
                "--dimensions", str(X.shape[1]),
                "--minpts", str(min_samples),
                "--input", temp_input,
                "--distMetric", str(distance_metric),
                "--minclustersize", str(min_cluster_size),
                "--quiet"  # Suppress debug output
            ]
            
            if distance_metric == 4:  # Minkowski
                cmd.extend(["--minkowskiP", str(minkowski_p)])
            
            # Run the executable and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"GPU HDBSCAN failed: {result.stderr}")
                return np.full(len(X), -1)  # Return noise labels on failure
            
            # Parse output to extract cluster labels
            # You'll need to modify your C++ code to output labels in a parseable format
            labels = self._parse_output(result.stdout, len(X))
            return labels
            
        finally:
            # Clean up temporary file
            os.unlink(temp_input)
    
    def _parse_output(self, output, n_points):
        """
        Parse the output from your C++ executable to extract cluster labels
        You'll need to modify your C++ code to output labels in a format like:
        CLUSTER_LABELS: 0 0 1 1 -1 2 2 ...
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

def generate_test_data(data_type='blobs', n_samples=1000):
    """Generate synthetic datasets for testing"""
    if data_type == 'blobs':
        X, y = make_blobs(n_samples=n_samples, centers=4, n_features=2, 
                         cluster_std=1.5, random_state=42)
    elif data_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    elif data_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    elif data_type == 'anisotropic':
        X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2, random_state=42)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
    
    X = StandardScaler().fit_transform(X)
    return X, y

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

def run_benchmark():
    """Main benchmark function"""
    # Initialize algorithms
    gpu_hdbscan = GPUHDBSCANWrapper()
    
    # Test parameters
    min_samples = 5
    min_cluster_size = 50
    
    # Results storage
    results = []
    
    # Test datasets
    datasets = ['blobs', 'circles', 'moons', 'anisotropic']
    sample_sizes = [10000, 100000, 500000]
    
    for data_type in datasets:
        for n_samples in sample_sizes:
            print(f"\nTesting {data_type} dataset with {n_samples} samples...")
            
            # Generate data
            X, true_labels = generate_test_data(data_type, n_samples)
            
            # Test GPU HDBSCAN
            print("Running GPU HDBSCAN...")
            try:
                gpu_labels, gpu_time, gpu_memory = track_performance(
                    gpu_hdbscan.fit_predict, X, min_samples, min_cluster_size
                )
                gpu_success = True
            except Exception as e:
                print(f"GPU HDBSCAN failed: {e}")
                gpu_labels, gpu_time, gpu_memory = np.array([]), float('inf'), 0
                gpu_success = False
            
            # Test sklearn HDBSCAN
            print("Running sklearn HDBSCAN...")
            sklearn_hdbscan = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
            sklearn_labels, sklearn_time, sklearn_memory = track_performance(
                sklearn_hdbscan.fit_predict, X
            )
            
            # Store results
            result = {
                'Dataset': data_type,
                'Samples': n_samples,
                'GPU_Time': gpu_time,
                'Sklearn_Time': sklearn_time,
                'GPU_Memory': gpu_memory,
                'Sklearn_Memory': sklearn_memory,
                'GPU_Success': gpu_success,
                'GPU_Clusters': len(set(gpu_labels)) - (1 if -1 in gpu_labels else 0) if gpu_success else 0,
                'Sklearn_Clusters': len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0),
            }
            
            if gpu_success and sklearn_time > 0:
                result['Speedup_vs_Sklearn'] = sklearn_time / gpu_time
            else:
                result['Speedup_vs_Sklearn'] = 0
            
            results.append(result)
            
            print(f"Results for {data_type} ({n_samples} samples):")
            print(f"  GPU: {gpu_time:.4f}s, {gpu_memory:.2f}MB, {result['GPU_Clusters']} clusters")
            print(f"  Sklearn: {sklearn_time:.4f}s, {sklearn_memory:.2f}MB, {result['Sklearn_Clusters']} clusters")
            if gpu_success:
                print(f"  Speedup vs Sklearn: {result['Speedup_vs_Sklearn']:.2f}x")
    
    # Create results DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv('gpu_hdbscan_benchmark_results.csv', index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    
    # Plot results
    # plot_benchmark_results(df)
    
    return df

# def plot_benchmark_results(df):
#     """Plot benchmark results"""
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
#     # Time comparison
#     datasets = df['Dataset'].unique()
#     x = np.arange(len(datasets))
#     width = 0.25
    
#     gpu_times = df.groupby('Dataset')['GPU_Time'].mean()
#     sklearn_times = df.groupby('Dataset')['Sklearn_Time'].mean()
    
#     axes[0,0].bar(x - width, gpu_times, width, label='GPU HDBSCAN', alpha=0.8)
#     axes[0,0].bar(x, sklearn_times, width, label='Sklearn HDBSCAN', alpha=0.8)
#     axes[0,0].set_xlabel('Dataset')
#     axes[0,0].set_ylabel('Time (seconds)')
#     axes[0,0].set_title('Execution Time Comparison')
#     axes[0,0].set_xticks(x)
#     axes[0,0].set_xticklabels(datasets)
#     axes[0,0].legend()
#     axes[0,0].set_yscale('log')
    
#     # Memory comparison
#     gpu_memory = df.groupby('Dataset')['GPU_Memory'].mean()
#     sklearn_memory = df.groupby('Dataset')['Sklearn_Memory'].mean()
    
#     axes[0,1].bar(x - width, gpu_memory, width, label='GPU HDBSCAN', alpha=0.8)
#     axes[0,1].bar(x, sklearn_memory, width, label='Sklearn HDBSCAN', alpha=0.8)
#     axes[0,1].set_xlabel('Dataset')
#     axes[0,1].set_ylabel('Memory (MB)')
#     axes[0,1].set_title('Memory Usage Comparison')
#     axes[0,1].set_xticks(x)
#     axes[0,1].set_xticklabels(datasets)
#     axes[0,1].legend()
    
#     # Speedup visualization
#     speedup_sklearn = df.groupby('Dataset')['Speedup_vs_Sklearn'].mean()
    
#     axes[1,0].bar(x - width/2, speedup_sklearn, width, label='vs Sklearn', alpha=0.8)
#     axes[1,0].set_xlabel('Dataset')
#     axes[1,0].set_ylabel('Speedup Factor')
#     axes[1,0].set_title('GPU HDBSCAN Speedup')
#     axes[1,0].set_xticks(x)
#     axes[1,0].set_xticklabels(datasets)
#     axes[1,0].legend()
#     axes[1,0].axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
#     # Scaling with data size
#     for dataset in datasets:
#         subset = df[df['Dataset'] == dataset]
#         axes[1,1].plot(subset['Samples'], subset['GPU_Time'], 'o-', label=f'GPU {dataset}', alpha=0.8)
#         axes[1,1].plot(subset['Samples'], subset['Sklearn_Time'], 's--', label=f'Sklearn {dataset}', alpha=0.8)
    
#     axes[1,1].set_xlabel('Number of Samples')
#     axes[1,1].set_ylabel('Time (seconds)')
#     axes[1,1].set_title('Scaling with Data Size')
#     axes[1,1].set_yscale('log')
#     axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     plt.tight_layout()
#     plt.savefig('gpu_hdbscan_benchmark.png', dpi=300, bbox_inches='tight')
#     plt.show()

if __name__ == "__main__":
    # Make sure your executable is built
    executable_path = "./gpu_hdbscan/build/gpu_hdbscan"
    if not os.path.exists(executable_path):
        print(f"Executable not found at {executable_path}")
        print("Please run 'make' to build the project first")
        exit(1)
    
    # Run benchmark
    results = run_benchmark()
    print(f"\nBenchmark complete! Results saved to 'gpu_hdbscan_benchmark_results.csv'")
    print("Plots saved to 'gpu_hdbscan_benchmark.png'")