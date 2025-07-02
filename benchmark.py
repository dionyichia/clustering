import subprocess
import numpy as np
import time
import psutil
import os
from typing import List, Dict
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for headless environments
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import glob

class GPUHDBSCANWrapper:
    def __init__(self, executable_path="./gpu_hdbscan_edited/build/gpu_hdbscan", data_path="./data/noisy_pdwInterns.csv"):
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

        Args:
            input_csv_path: Path to a CSV with header + data rows.
            dims: Number of feature columns in the CSV.
            min_samples: HDBSCAN min_samples.
            min_cluster_size: HDBSCAN min_cluster_size.
            distance_metric: (int) which distance metric flag to use.
            minkowski_p: Minkowski p‐value, if distMetric==4.
            quiet_mode: Suppress executable’s stdout/stderr to log file.

        Returns:
            ndarray of cluster labels (length = #rows in CSV minus header).
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
        # Again count rows for parse fallback
        with open(input_csv_path) as f:
            n = sum(1 for _ in f) - 1
        return self._parse_output(result.stdout, n)
    
    def fit_predict(self, X, min_samples=5, min_cluster_size=5, distance_metric=2, minkowski_p=2.0, quiet_mode=True):
        """
        Wrapper to call GPU HDBSCAN executable
        """
        exe_path = os.path.abspath(self.executable_path)
        exe_dir = os.path.dirname(exe_path)
        wking_dir = os.getcwd()

        log_path = os.path.join(wking_dir, "log_file.txt")

        filename = "temp_input" + str(os.getpid()) + ".txt"
        temp_input = os.path.join(wking_dir, filename)

        with open(temp_input, 'w') as f:
            # Write data in the format your C++ code expects
            # Add dummy header
            f.write(','.join([f'feature_{i}' for i in range(X.shape[1])]) + '\n')

            for point in X:
                f.write(','.join(map(str, point)) + '\n')
            temp_input = f.name
        
        try:
            # Prepare command arguments
            if quiet_mode:
                cmd = [
                    exe_path,
                    "--dimensions", str(X.shape[1]),
                    "--minpts", str(min_samples),
                    "--input", filename,
                    "--distMetric", str(distance_metric),
                    "--minclustersize", str(min_cluster_size),
                    "--quiet",  # Suppress debug output
                ]
            else:
                cmd = [
                    exe_path,
                    "--dimensions", str(X.shape[1]),
                    "--minpts", str(min_samples),
                    "--input", filename,
                    "--distMetric", str(distance_metric),
                    "--minclustersize", str(min_cluster_size)
                ]
            
            if distance_metric == 4:  # Minkowski
                cmd.extend(["--minkowskiP", str(minkowski_p)])
            
            # Run the executable and capture output
            result = subprocess.run(cmd, text=True, timeout=300, capture_output=True)

            # Write output to log file
            with open(log_path, "w") as log_file:
                if result.stdout:
                    log_file.write(result.stdout)
                if result.stderr:
                    log_file.write(result.stderr)
        
            if result.returncode != 0:
                print(f"GPU HDBSCAN failed with return code: {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                print(f"Command: {' '.join(cmd)}")
                return np.full(len(X), -1)
                    
            # Parse output to extract cluster labels
            # You'll need to modify your C++ code to output labels in a parseable format
            labels = self._parse_output(result.stdout, len(X))
            return labels
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_input):
                os.unlink(temp_input)
                # os.unlink(log_path)

            pass
    
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
        X, y = make_blobs(n_samples=n_samples, centers=10, n_features=2, 
                         cluster_std=1.5, random_state=42)
    # elif data_type == 'circles':
    #     X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    # elif data_type == 'moons':
    #     X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    elif data_type == 'anisotropic':
        X, y = make_blobs(n_samples=n_samples, centers=10, n_features=2, random_state=42)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
    else:
        print("Invalid data type for generation: ", data_type)
        return None, None
    
    X = StandardScaler().fit_transform(X)
    return X, y

def add_gaussian_noise(data_path, std_array=[1.0, 0.00021, 0.2, 0.2] ):
    """
    Add Gaussian noise to specific columns in a dataset.

    Parameters:
    - data: numpy array or pandas DataFrame
    - snr_db: desired Signal-to-Noise Ratio in dB. If set, overrides std.
    - std: standard deviation of the noise. Used only if snr_db is None.

    Returns:
    - noisy_data file path: data with added Gaussian noise
    """

    df = pd.read_csv(data_path)

    columns_to_noise = ["FREQ(MHz)", "PW(microsec)", "AZ_S0(deg)", "EL_S0(deg)"]
    std_map = dict(zip(columns_to_noise, std_array))

    df_noisy = df.copy()

    for col in columns_to_noise:
        if col in df.columns:
            noise = np.random.normal(loc=0.0, scale=std_map[col], size=df[col].shape)
            df_noisy[col] = df[col] + noise
        else:
            raise ValueError (f"Column {col} not found in CSV.")

    # Construct output file path
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(data_path)
    name, ext = os.path.splitext(base)
    output_path = os.path.join(output_dir, f"noisy_{name}{ext}")

    # Save data
    df_noisy.to_csv(output_path, index=False)

    return output_path

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


def batch_data(data_path: str, batch_interval: float = 2.0, 
               chunk_size: int = 10000, assume_sorted: bool = True) -> List[Dict[str, any]]:
    """
    Memory-efficient version for very large datasets that processes data in chunks.
    
    Args:
        data_path (str): Path to the CSV data file
        batch_interval (float): Time interval in seconds for batching
        chunk_size (int): Number of rows to process at once
        assume_sorted (bool): Whether to assume data is pre-sorted by TOA
    
    Returns:
        List of dictionaries in TestConfig format with keys:
        - name: str
        - data_type: str  
        - data_path: str
        - sample_size: int
        - batch_number: int
        - time_range: tuple
    """
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    if batch_interval <= 0:
        batch_interval = 2.0
    
    # Create batch directory
    batch_dir = os.path.join(os.path.dirname(data_path), 'batch_data')
    os.makedirs(batch_dir, exist_ok=True)
    
    test_configs = []
    current_batch = 1
    current_batch_data = []
    batch_start_time = None
    batch_end_time = None
    
    try:
        # Process file in chunks
        for chunk in pd.read_csv(data_path, chunksize=chunk_size):
            if 'TOA(ns)' not in chunk.columns:
                raise ValueError("Required column 'TOA(ns)' not found")
            
            # Convert TOA (sort only if not pre-sorted)
            chunk['TOA_seconds'] = chunk['TOA(ns)'] / 1e9
            if not assume_sorted:
                chunk = chunk.sort_values('TOA_seconds')
            
            # Set initial batch start time
            if batch_start_time is None:
                batch_start_time = chunk['TOA_seconds'].iloc[0]
            
            for _, row in chunk.iterrows():
                current_time = row['TOA_seconds']
                
                # Check if we need to start a new batch
                if current_time > batch_start_time + batch_interval:
                    # Save current batch if it has data
                    if current_batch_data:
                        batch_end_time = current_batch_data[-1]['TOA(ns)'] / 1e9
                        config = _save_batch_and_create_config(
                            current_batch_data, batch_dir, current_batch, 
                            batch_start_time, batch_end_time
                        )
                        test_configs.append(config)
                        current_batch_data = []
                        current_batch += 1
                    
                    # Update batch start time
                    batch_start_time = current_time
                
                # Add row to current batch (remove helper column)
                row_dict = row.drop('TOA_seconds').to_dict()
                current_batch_data.append(row_dict)
        
        # Save final batch
        if current_batch_data:
            batch_end_time = current_batch_data[-1]['TOA(ns)'] / 1e9
            config = _save_batch_and_create_config(
                current_batch_data, batch_dir, current_batch,
                batch_start_time, batch_end_time
            )
            test_configs.append(config)
        
        print(f"Memory-efficient processing complete: {len(test_configs)} batches created")
        return test_configs
        
    except Exception as e:
        raise RuntimeError(f"Error in memory-efficient processing: {e}")


def _save_batch_and_create_config(batch_data: List[Dict], batch_dir: str, batch_num: int,
                                    start_time: float, end_time: float) -> Dict[str, any]:
    """
    Helper function to save a batch of data and create a TestConfig dictionary.
    
    Returns:
        Dictionary in TestConfig format
    """
    # Save batch file
    batch_df = pd.DataFrame(batch_data)
    batch_filename = f"Data_Batch_{batch_num}.csv"
    batch_file_path = os.path.join(batch_dir, batch_filename)
    batch_df.to_csv(batch_file_path, index=False)
    
    # Create TestConfig dictionary
    config = {
        'name': f"Data_Batch_{batch_num}_{len(batch_data)}samples",
        'data_type': f"Data_Batch_{batch_num}",
        'data_path': batch_file_path,
        'sample_size': len(batch_data),
        'batch_number': batch_num,
        'time_range': (start_time, end_time)
    }
    
    return config

def run_benchmark_with_visualization_batched(
    data_path=None,
    executable_path=None,
    use_amp=False,
    use_toa=False
):
    """Enhanced benchmark function with cluster visualization"""
    # Define output folder
    output_dir = "benchmark_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize algorithm wrapper
    gpu_hdbscan = GPUHDBSCANWrapper(executable_path=executable_path)
    
    # Test parameters
    min_samples     = 3
    min_cluster_size= 50
    quiet_mode      = True
    
    results = []
    
    if data_path is None:
        raise ValueError("Must supply data_path to batch_data folder")
    
    # 1) Make sure batches exist
    batch_dir = os.path.join(os.path.dirname(data_path), "batch_data")
    if not os.path.isdir(batch_dir):
        raise FileNotFoundError(f"batch_data folder not found at {batch_dir}")
    
    # 2) Grab every CSV in there
    csv_paths = sorted(glob.glob(os.path.join(batch_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError("No .csv files found in batch_data/")
    
    # 3) Loop over each batch file
    for csv_file in csv_paths:
        # derive a name & sample count
        batch_name = os.path.splitext(os.path.basename(csv_file))[0]
        # count data rows (minus header)
        with open(csv_file, 'r') as f:
            n_samples = sum(1 for _ in f) - 1
        
        print(f"\n=== Processing batch {batch_name} ({n_samples} rows) ===")
        
        # read only the features you want
        df = pd.read_csv(csv_file)
        feature_cols = ['PW(microsec)', 'FREQ(MHz)', 'AZ_S0(deg)', 'EL_S0(deg)']
        if use_amp: feature_cols.append('Amp_S0(dBm)')
        if use_toa: feature_cols.append('TOA(ns)')
        
        X = df[feature_cols].to_numpy()
        
        # GPU HDBSCAN on the file directly
        print("  -> Running GPU HDBSCAN...")
        dims = X.shape[1]
        
        # track timing & memory around that call if you like:
        gpu_labels, gpu_time, gpu_mem = track_performance(
            gpu_hdbscan.fit_predict_batched, csv_file,
            dims,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            quiet_mode=quiet_mode
        )
        
        # sklearn HDBSCAN for comparison
        print("  -> Running sklearn HDBSCAN...")
        sklearn_model = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
        sklearn_labels, sklearn_time, sklearn_memory = track_performance(
            sklearn_model.fit_predict, X
        )
        
        # collect results
        result = {
            'Batch':        batch_name,
            'Samples':      n_samples,
            'GPU_Time': gpu_time,
            'Sklearn_Time': sklearn_time,
            'GPU_Memory': gpu_mem,
            'Sklearn_Memory': sklearn_memory,
            'GPU_Clusters': len(set(gpu_labels)) - (1 if -1 in gpu_labels else 0),
            'Sklearn_Clusters': len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0),
        }
        results.append(result)
    
    # save summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'batch_benchmark_summary.csv'), index=False)
    print("\n== All batches processed ==")
    print(results_df)
    
    return results_df

def run_benchmark_with_visualization(data_path=None, executable_path=None, use_amp=False, use_toa=False):
    """Enhanced benchmark function with cluster visualization"""
    # Define output folder
    output_dir = "benchmark_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize algorithms
    gpu_hdbscan = GPUHDBSCANWrapper(executable_path=executable_path)
    
    # Test parameters
    min_samples = 3
    min_cluster_size = 50
    quiet_mode = True
    batch_interval = 2.0 # Interval used in batching
    
    # Results storage
    results = []
    
    # Test datasets
    if not data_path:
        print("Data file not found, generating data...")

        data = [
            {
                'name': 'blobs_100k',
                'type': 'blobs',
                'data_path': None,
                'sample_size': 100000,
            },
            {
                'name': 'anisotropic_100k', 
                'type': 'anisotropic',
                'data_path': None,
                'sample_size': 1000000,
            }
        ]

    else:
        print("Data file found")

        # Chunk size can be taken as the maximum number of points in a batch
        data = batch_data(data_path=data_path, batch_interval=batch_interval, chunk_size=200000, assume_sorted=True)
        
    for dataset in data:
        data_type = dataset.get('type')
        n_samples = dataset.get('sample_size')

        print(f"\nTesting {dataset.get('name')} dataset with {n_samples} samples...")
        
        # If no data provided, generate data
        if data_path is None:
            X, true_labels = generate_test_data(data_type, n_samples)
        else:
            df = pd.read_csv(dataset.get('data_path'))  
            feature_cols = ['PW(microsec)', 'FREQ(MHz)', 'AZ_S0(deg)', 'EL_S0(deg)']

            if use_amp:
                feature_cols.append('Amp_S0(dBm)')
            if use_toa:
                feature_cols.append('TOA(ns)')

            X = df[feature_cols].to_numpy()
            true_labels = df['EmitterId'].to_numpy()
        
        feature_names = feature_cols if data_path is not None else ['Feature 1', 'Feature 2']
        
        # Test GPU HDBSCAN
        print("Running GPU HDBSCAN...")
        try:
            gpu_labels, gpu_time, gpu_memory = track_performance(
                gpu_hdbscan.fit_predict, X, min_samples, min_cluster_size, quiet_mode=quiet_mode
            )
            gpu_success = True
            
            # VISUALIZE CLUSTERS - This is the new part!
            if gpu_success and len(np.unique(gpu_labels)) > 1:
                print("Creating cluster visualizations...")
                
                # Basic 2D plot
                plot_clusters_2d(X, gpu_labels, 
                                feature_names=['Feature 1', 'Feature 2'],
                                title=f"GPU HDBSCAN: {data_type} dataset ({n_samples} samples)",
                                save_path=os.path.join(output_dir, f"gpu_clusters_{data_type}_{n_samples}.png"))
                
                # Comprehensive analysis
                analysis_results = comprehensive_cluster_analysis(
                    X, gpu_labels, 
                    feature_names=['Feature 1', 'Feature 2'],
                    save_prefix=os.path.join(output_dir, f"gpu_analysis_{data_type}_{n_samples}")
                )
                
                print(f"Cluster analysis saved with prefix: gpu_analysis_{data_type}_{n_samples}")
            
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
        
        # Visualize sklearn results too for comparison
        if len(np.unique(sklearn_labels)) > 1:
            plot_clusters_2d(X, sklearn_labels, 
                            feature_names=['Feature 1', 'Feature 2'],
                            title=f"Sklearn HDBSCAN: {data_type} dataset ({n_samples} samples)",
                            save_path=os.path.join(output_dir, f"sklearn_clusters_{data_type}_{n_samples}.png"))
        
        # Store results (same as before)
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

    # Create results DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'gpu_hdbscan_benchmark_results.csv'), index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Plot results
    # plot_benchmark_results(results_df)
    
    return df

def plot_benchmark_results(df):
    """Plot benchmark results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    datasets = df['Dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.25
    
    # Time comparison
    gpu_times = df.groupby('Dataset')['GPU_Time'].mean()
    sklearn_times = df.groupby('Dataset')['Sklearn_Time'].mean()
    
    axes[0, 0].bar(x - width, gpu_times, width, label='GPU HDBSCAN', alpha=0.8)
    axes[0, 0].bar(x, sklearn_times, width, label='Sklearn HDBSCAN', alpha=0.8)
    axes[0, 0].set_xlabel('Dataset')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_title('Execution Time Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(datasets)
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Memory comparison
    gpu_memory = df.groupby('Dataset')['GPU_Memory'].mean()
    sklearn_memory = df.groupby('Dataset')['Sklearn_Memory'].mean()
    
    axes[0, 1].bar(x - width, gpu_memory, width, label='GPU HDBSCAN', alpha=0.8)
    axes[0, 1].bar(x, sklearn_memory, width, label='Sklearn HDBSCAN', alpha=0.8)
    axes[0, 1].set_xlabel('Dataset')
    axes[0, 1].set_ylabel('Memory (MB)')
    axes[0, 1].set_title('Memory Usage Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(datasets)
    axes[0, 1].legend()
    
    # Speedup visualization
    speedup_sklearn = df.groupby('Dataset')['Speedup_vs_Sklearn'].mean()
    
    axes[1, 0].bar(x - width / 2, speedup_sklearn, width, label='vs Sklearn', alpha=0.8)
    axes[1, 0].set_xlabel('Dataset')
    axes[1, 0].set_ylabel('Speedup Factor')
    axes[1, 0].set_title('GPU HDBSCAN Speedup')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(datasets)
    axes[1, 0].legend()
    axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    # Scaling with data size
    for dataset in datasets:
        subset = df[df['Dataset'] == dataset]
        axes[1, 1].plot(subset['Samples'], subset['GPU_Time'], 'o-', label=f'GPU {dataset}', alpha=0.8)
        axes[1, 1].plot(subset['Samples'], subset['Sklearn_Time'], 's--', label=f'Sklearn {dataset}', alpha=0.8)
    
    axes[1, 1].set_xlabel('Number of Samples')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_title('Scaling with Data Size')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('gpu_hdbscan_benchmark.png', dpi=300, bbox_inches='tight')

def plot_clusters_2d(X, labels, feature_names=None, title="GPU HDBSCAN Clustering Results", 
                     save_path=None, use_plotly=False):
    """
    Plot 2D clustering results using the first two features (pulse width and frequency)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points (will use first 2 dimensions)
    labels : array-like, shape (n_samples,)
        Cluster labels from GPU HDBSCAN
    feature_names : list, optional
        Names of features for axis labels
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    use_plotly : bool
        Whether to use Plotly for interactive plots
    """
    
    # Use first two dimensions
    X_2d = X[:, :2]
    
    # Set default feature names
    if feature_names is None:
        feature_names = ['Pulse Width', 'Frequency']
    
    # Get unique clusters (excluding noise if present)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_count = np.sum(labels == -1)
    
    print(f"Found {n_clusters} clusters and {noise_count} noise points")
    
    if use_plotly:
        # Create interactive Plotly plot
        df_plot = pd.DataFrame({
            feature_names[0]: X_2d[:, 0],
            feature_names[1]: X_2d[:, 1],
            'Cluster': labels.astype(str)
        })
        
        # Create color map for clusters
        colors = px.colors.qualitative.Set1
        color_map = {}
        for i, label in enumerate(unique_labels):
            if label == -1:
                color_map[str(label)] = 'black'  # Noise points in black
            else:
                color_map[str(label)] = colors[i % len(colors)]
        
        fig = px.scatter(
            df_plot,
            x=feature_names[0],
            y=feature_names[1],
            color='Cluster',
            title=f"{title}<br>Clusters: {n_clusters}, Noise points: {noise_count}",
            color_discrete_map=color_map,
            width=800,
            height=600
        )
        
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.show()
        
        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
            
    else:
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

def visualize_high_dimensional_clusters(X, labels, feature_names=None, 
                                       method='pca', title="High-Dimensional Clustering",
                                       save_path=None):
    """
    Visualize high-dimensional clustering results using dimensionality reduction
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels
    feature_names : list, optional
        Names of features
    method : str
        Dimensionality reduction method ('pca' or 'tsne')
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X_scaled)
        explained_var = reducer.explained_variance_ratio_
        method_title = f"PCA (Explained variance: {explained_var[0]:.2f}, {explained_var[1]:.2f})"
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
        X_reduced = reducer.fit_transform(X_scaled)
        method_title = "t-SNE"
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    # Plot the results
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_count = np.sum(labels == -1)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            mask = labels == label
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                       c='black', marker='x', s=50, alpha=0.6, label='Noise')
        else:
            mask = labels == label
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                       c=[colors[i]], s=60, alpha=0.7, label=f'Cluster {label}')
    
    plt.xlabel(f'{method_title} Component 1')
    plt.ylabel(f'{method_title} Component 2')
    plt.title(f"{title} - {method_title}\nClusters: {n_clusters}, Noise points: {noise_count}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return X_reduced

def evaluate_clustering_quality(X, labels, feature_names=None):
    """
    Evaluate clustering quality using multiple metrics
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels
    feature_names : list, optional
        Names of features
    
    Returns:
    --------
    dict : Dictionary containing evaluation metrics
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
    
    # Print results
    print("\n" + "="*50)
    print("CLUSTERING EVALUATION METRICS")
    print("="*50)
    print(f"Number of clusters: {metrics['n_clusters']}")
    print(f"Number of noise points: {metrics['n_noise_points']}")
    print(f"Noise ratio: {metrics['noise_ratio']:.3f}")
    print(f"Total points: {metrics['total_points']}")
    
    if metrics['silhouette_score'] is not None:
        print(f"\nQuality Metrics:")
        print(f"Silhouette Score: {metrics['silhouette_score']:.3f} (higher is better, range: [-1, 1])")
        print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.3f} (higher is better)")
        print(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f} (lower is better)")
    else:
        print("\nQuality metrics could not be calculated (need at least 2 clusters)")
    
    return metrics

def create_cluster_summary_table(X, labels, feature_names=None):
    """
    Create a summary table of cluster statistics
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels
    feature_names : list, optional
        Names of features
    
    Returns:
    --------
    pd.DataFrame : Summary statistics for each cluster
    """
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    unique_labels = np.unique(labels)
    summary_data = []
    
    for label in unique_labels:
        mask = labels == label
        cluster_data = X[mask]
        
        cluster_info = {
            'Cluster': 'Noise' if label == -1 else f'Cluster {label}',
            'Size': np.sum(mask),
            'Percentage': np.sum(mask) / len(labels) * 100
        }
        
        # Add statistics for each feature
        for i, feature_name in enumerate(feature_names):
            cluster_info[f'{feature_name}_Mean'] = np.mean(cluster_data[:, i])
            cluster_info[f'{feature_name}_Std'] = np.std(cluster_data[:, i])
            cluster_info[f'{feature_name}_Min'] = np.min(cluster_data[:, i])
            cluster_info[f'{feature_name}_Max'] = np.max(cluster_data[:, i])
        
        summary_data.append(cluster_info)
    
    summary_df = pd.DataFrame(summary_data)
    
    # print("\n" + "="*80)
    # print("CLUSTER SUMMARY STATISTICS")
    # print("="*80)
    # print(summary_df.to_string(index=False, float_format='%.3f'))
    
    return summary_df

def comprehensive_cluster_analysis(X, labels, feature_names=None, save_prefix="cluster_analysis"):
    """
    Perform comprehensive cluster analysis including visualization and evaluation
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels from GPU HDBSCAN
    feature_names : list, optional
        Names of features
    save_prefix : str
        Prefix for saved files
    
    Returns:
    --------
    dict : Dictionary containing all analysis results
    """
    
    if feature_names is None:
        if X.shape[1] == 4:
            feature_names = ['Pulse Width', 'Frequency', 'Azimuth', 'Elevation']
        else:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    results = {}
    
    # 1. 2D visualization using first two features
    print("Creating 2D visualization...")
    plot_clusters_2d(X, labels, feature_names[:2], 
                     title="GPU HDBSCAN: Pulse Width vs Frequency",
                     save_path=f"{save_prefix}_2d.png")
    
    # 2. High-dimensional visualization if needed
    if X.shape[1] > 2:
        print("Creating PCA visualization...")
        X_pca = visualize_high_dimensional_clusters(X, labels, feature_names, 
                                                   method='pca',
                                                   title="GPU HDBSCAN: PCA Projection",
                                                   save_path=f"{save_prefix}_pca.png")
        results['X_pca'] = X_pca
        
        # print("Creating t-SNE visualization...")
        # X_tsne = visualize_high_dimensional_clusters(X, labels, feature_names, 
        #                                             method='tsne',
        #                                             title="GPU HDBSCAN: t-SNE Projection",
        #                                             save_path=f"{save_prefix}_tsne.png")
        results['X_tsne'] = None
    
    # 3. Evaluate clustering quality
    print("Evaluating clustering quality...")
    metrics = evaluate_clustering_quality(X, labels, feature_names)
    results['metrics'] = metrics
    
    # 4. Create cluster summary table
    print("Creating cluster summary...")
    summary_df = create_cluster_summary_table(X, labels, feature_names)
    results['summary'] = summary_df
    
    # Save summary to CSV
    summary_df.to_csv(f"{save_prefix}_summary.csv", index=False)
    
    return results

if __name__ == "__main__":
    # Make sure your executable is built
    executable_path = "./gpu_hdbscan_edited/build/gpu_hdbscan"
    if not os.path.exists(executable_path):
        print(f"Executable not found at {executable_path}")
        print("Please run 'make' to build the project first")
        exit(1)

    # Make sure data file path exists
    data_path = "./data/pdwInterns.csv"
    if not os.path.exists(data_path):
        print(f"Date file not found at {data_path}")
        exit(1)
    
    # Run benchmark
    # if data path provided will use data file else will generate 2d data.
    noisy_data_path = "./data/noisy_pdwInterns.csv"
    std_array = [1.0, 0.00021, 0.2, 0.2] 

    if not os.path.exists(noisy_data_path):
        print(f"Noisy Data not found at {noisy_data_path}")
        noisy_data_path = add_gaussian_noise(data_path, std=std_array)
    
    # Testing Batched Functions
    batch_path = "./data/batch_data"
    
    if os.path.exists(batch_path):
        results = run_benchmark_with_visualization_batched(data_path=batch_path,executable_path=executable_path,use_amp=False,use_toa=False) 
    else:
        results = run_benchmark_with_visualization(data_path=noisy_data_path, executable_path=executable_path, use_amp=False, use_toa=False)

    print(f"\nBenchmark complete! Results saved to 'gpu_hdbscan_benchmark_results.csv'")
    print("Plots saved to 'gpu_hdbscan_benchmark.png'")
