import subprocess
import numpy as np
import time
import psutil
import os
from typing import List, Dict,Tuple
from sklearn.cluster import HDBSCAN,DBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
import pandas as pd
import glob
import threading

from utils.eval import *

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
        cluster_method: int = 1, # 1 is EOM, 2 is Leaf
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
            "--clusterMethod",   str(cluster_method),
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

def add_gaussian_noise(data_path, std_map):
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

    columns_to_noise = list(std_map.keys())
    print("Adding noise to columns: ", columns_to_noise)

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

def find_min_points_per_emitter(df, emitter_col='EmitterId'):
    if emitter_col not in df.columns:
        raise ValueError(f"Column {emitter_col} not found in dataset")
    
    emitter_counts = df[emitter_col].value_counts()

    min_points = int(emitter_counts.min())

    print(f"Unique Emitter in Batch: {len(emitter_counts)}")
    print(f"Points per Emitter - Min: {min_points}, Max: {emitter_counts.max()}, Mean: {emitter_counts.mean():.1f}")

    return min_points

def run_benchmark_with_visualization_batched(
    data_path=None,
    executable_path=None,
    use_amp=False,
    use_toa=False,
    use_lat_lng=False,
):
    """Enhanced benchmark function with cluster visualization"""
    # Define output folder
    output_dir = "benchmark_outputs"
    os.makedirs(output_dir, exist_ok=True)
    dbscan_params_file = "dbscan_params.txt"
    dbscan_params = load_dbscan_parameters(dbscan_params_file)
    if not dbscan_params:
        print("No DBSCAN parameters loaded. Exiting.")
        return
    
    # Initialize algorithm wrapper
    gpu_hdbscan = GPUHDBSCANWrapper(executable_path=executable_path)
    
    # Test parameters
    min_samples = 3
    min_cluster_size = 20
    quiet_mode = True

    emitter_col = "EmitterId"

    if use_lat_lng:
        feature_cols = ['PW(microsec)', 'FREQ(MHz)', 'AZ_S0(deg)', 'EL_S0(deg)', 'Latitude(deg)', 'Longitude(deg)']
    else:
        feature_cols = ['PW(microsec)', 'FREQ(MHz)', 'AZ_S0(deg)', 'EL_S0(deg)']
    
    results = []
    evaluation_results = []
    
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
    
    # Find Optimal Parameters for DBSCAN and HDBSCAN
    # 3) Loop over each batch file
    for csv_file in csv_paths:
        batch_filename = os.path.basename(csv_file)
        # derive a name & sample count
        batch_name = os.path.splitext(os.path.basename(csv_file))[0]
        if batch_filename not in dbscan_params:
            print(f"No parameters found for {batch_filename}, skipping...")
            continue
        min_samples, epsilon = dbscan_params[batch_filename]

        # count data rows (minus header)
        with open(csv_file, 'r') as f:
            n_samples = sum(1 for _ in f) - 1
        
        print(f"\n=== Processing batch {batch_name} ({n_samples} rows) ===")
        
        # read only the features you want
        df = pd.read_csv(csv_file)

        # Try finding min points per emitter from all unique emitters in batch
        try: 
            min_cluster_size = max(20, find_min_points_per_emitter(df, emitter_col))
        except Exception as e:
            print(f"Error processing file: {e}")

        if use_amp: feature_cols.append('Amp_S0(dBm)')
        if use_toa: feature_cols.append('TOA(ns)')
        
        X = df[feature_cols].to_numpy()
        
        # Scale the data for DBSCAN
        # technically our HDBSCAN uses MinMaxScaling
        # if i change to MinMaxScaler, will need to recompute optimal epsilon and min_samples
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        true_labels = df['EmitterId'].to_numpy()
        
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

        # DBSCAN for comparison - FIXED VERSION
        print("  -> Running sklearn DBSCAN with optimal epsilon and minPoints...")
        print(f"DBSCAN Parameters: min_samples={min_samples}, epsilon={epsilon}")
        dbscan_model = DBSCAN(eps=epsilon, min_samples=min_samples)
        dbscan_labels, dbscan_time, dbscan_memory = track_performance(
            dbscan_model.fit_predict, X_scaled  # Use scaled data
        )
        
        # Evaluate clustering results with ground truth
        print("  -> Evaluating clustering results...")
        eval_results = evaluate_clustering_with_ground_truth(
            X, gpu_labels, sklearn_labels, dbscan_labels, true_labels,
            batch_name, feature_cols, gpu_time, sklearn_time, dbscan_time, output_dir
        )
        evaluation_results.append(eval_results)
        
        # collect performance results 
        result = {
            'Batch': batch_name,
            'Samples': n_samples,
            'GPU_Time': gpu_time,
            'Sklearn_Time': sklearn_time,
            'DBSCAN_Time': dbscan_time,
            'GPU_Memory': gpu_mem,
            'Sklearn_Memory': sklearn_memory,
            'DBSCAN_Memory': dbscan_memory,
            'GPU_Clusters': eval_results['gpu_metrics']['N_Clusters'],
            'Sklearn_Clusters': eval_results['sklearn_metrics']['N_Clusters'],
            'DBSCAN_Clusters': eval_results['dbscan_metrics']['N_Clusters'],
            'True_Clusters': eval_results['ground_truth_stats']['N_True_Clusters'],
            # Add accuracy metrics
            'GPU_ARI': eval_results['gpu_metrics']['ARI'],
            'GPU_NMI': eval_results['gpu_metrics']['NMI'],
            'GPU_V_Measure': eval_results['gpu_metrics']['V_Measure'],
            'Sklearn_ARI': eval_results['sklearn_metrics']['ARI'],
            'Sklearn_NMI': eval_results['sklearn_metrics']['NMI'],
            'Sklearn_V_Measure': eval_results['sklearn_metrics']['V_Measure'],
            'DBSCAN_ARI': eval_results['dbscan_metrics']['ARI'],
            'DBSCAN_NMI': eval_results['dbscan_metrics']['NMI'],
            'DBSCAN_V_Measure': eval_results['dbscan_metrics']['V_Measure'],
            'Algorithm_Agreement_ARI': eval_results['algorithm_agreement']['ARI'],
        }
        results.append(result)
    
    # save detailed summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'batch_benchmark_summary_with_accuracy.csv'), index=False)
    
    # Create overall summary - UPDATED TO INCLUDE DBSCAN
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY WITH ACCURACY METRICS")
    print("="*80)
    print(results_df[['Batch', 'Samples', 'GPU_Time', 'Sklearn_Time', 'DBSCAN_Time',
                      'GPU_ARI', 'Sklearn_ARI', 'DBSCAN_ARI', 'Algorithm_Agreement_ARI']].to_string(index=False))
    
    # Calculate average performance - UPDATED TO INCLUDE DBSCAN
    avg_gpu_ari = results_df['GPU_ARI'].mean()
    avg_sklearn_ari = results_df['Sklearn_ARI'].mean()
    avg_dbscan_ari = results_df['DBSCAN_ARI'].mean()
    avg_agreement = results_df['Algorithm_Agreement_ARI'].mean()
    
    print(f"\nAVERAGE PERFORMANCE:")
    print(f"GPU HDBSCAN ARI: {avg_gpu_ari:.3f}")
    print(f"Sklearn HDBSCAN ARI: {avg_sklearn_ari:.3f}")
    print(f"DBSCAN ARI: {avg_dbscan_ari:.3f}")
    print(f"Algorithm Agreement: {avg_agreement:.3f}")
    
    return results_df, evaluation_results

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

def timeout_handler(func, args, kwargs, timeout_duration):
    """Run function with timeout"""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if thread.is_alive():
        return None, True, None  # result, timed_out, exception
    else:
        return result[0], False, exception[0]


def track_performance_with_timeout(func, *args, timeout=300, **kwargs):
    """Track execution time and memory usage with timeout"""
    process = psutil.Process()
    
    # Get initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Time the execution with timeout
    start_time = time.time()
    result, timed_out, exception = timeout_handler(func, args, kwargs, timeout)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    if timed_out:
        execution_time = timeout
        result = None
        print(f"Algo execution timed out > {timeout / 60} seconds")
        
    if exception:
        raise exception
    
    # Get peak memory (approximation)
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = max(0, final_memory - initial_memory)
    
    return result, execution_time, memory_used, timed_out


def create_temp_csv(df):
    """Create temporary CSV file for GPU HDBSCAN"""
    temp_filename = f"temp_data_{int(time.time())}.csv"
    df.to_csv(temp_filename, index=False)
    return temp_filename


def predict_completion_time(sample_sizes, times, target_size):
    """Predict completion time using polynomial fitting without scipy"""
    # Filter out None values and convert to numpy arrays
    valid_indices = [i for i, t in enumerate(times) if t is not None]
    if len(valid_indices) < 2:
        return None

    valid_sizes = np.array([sample_sizes[i] for i in valid_indices])
    valid_times = np.array([times[i] for i in valid_indices])

    try:
        # Try quadratic fit
        coeffs = np.polyfit(valid_sizes, valid_times, deg=2)  # a*x^2 + b*x + c
        prediction = np.polyval(coeffs, target_size)

        # Ensure prediction is reasonable
        if prediction > 0 and prediction < 7200:
            return prediction

        # Fallback: linear extrapolation
        if len(valid_times) >= 2:
            slope = (valid_times[-1] - valid_times[-2]) / (valid_sizes[-1] - valid_sizes[-2])
            prediction = valid_times[-1] + slope * (target_size - valid_sizes[-1])
            return max(0, prediction)

    except:
        pass

    return None

def create_benchmark_plot(results_df, output_dir, timeout):
    """Create professional benchmark visualization"""
    
    # Set style
    plt.style.use('ggplot')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Clustering Algorithm Performance Benchmark', fontsize=16, fontweight='bold')
    
    sample_sizes = results_df['Samples'].values
    
    # 1. Execution Time Comparison
    ax1.set_title('Execution Time vs Sample Size', fontweight='bold')
    
    # Prepare data for plotting
    gpu_times = []
    sklearn_times = []
    dbscan_times = []
    
    for _, row in results_df.iterrows():
        gpu_times.append(row['GPU_Time'] if not row['GPU_Timeout'] else None)
        sklearn_times.append(row['Sklearn_Time'] if not row['Sklearn_Timeout'] else None)
        dbscan_times.append(row['DBSCAN_Time'] if not row['DBSCAN_Timeout'] else None)
    
    # Plot actual measurements
    gpu_mask = [t is not None for t in gpu_times]
    sklearn_mask = [t is not None for t in sklearn_times]
    dbscan_mask = [t is not None for t in dbscan_times]
    
    if any(gpu_mask):
        ax1.plot(np.array(sample_sizes)[gpu_mask], np.array(gpu_times)[gpu_mask], 
                'o-', label='GPU HDBSCAN', linewidth=2, markersize=6)
    
    if any(sklearn_mask):
        ax1.plot(np.array(sample_sizes)[sklearn_mask], np.array(sklearn_times)[sklearn_mask], 
                's-', label='Sklearn HDBSCAN', linewidth=2, markersize=6)
    
    if any(dbscan_mask):
        ax1.plot(np.array(sample_sizes)[dbscan_mask], np.array(dbscan_times)[dbscan_mask], 
                '^-', label='DBSCAN', linewidth=2, markersize=6)
    
    # Store last valid points for each algorithm
    last_gpu = None
    last_sklearn = None
    last_dbscan = None

    gpu_color = ax1.plot([], [], 'o-', label='GPU HDBSCAN')[0].get_color()
    sklearn_color = ax1.plot([], [], 's-', label='Sklearn HDBSCAN')[0].get_color()
    dbscan_color = ax1.plot([], [], '^-', label='DBSCAN')[0].get_color()

    for i, (size, gpu_time, sklearn_time, dbscan_time) in enumerate(zip(sample_sizes, gpu_times, sklearn_times, dbscan_times)):
        # GPU HDBSCAN
        if gpu_time is not None:
            last_gpu = (size, gpu_time)
        else:
            prediction = predict_completion_time(sample_sizes[:i], gpu_times[:i], size)
            if prediction and last_gpu:
                ax1.plot([last_gpu[0], size], [last_gpu[1], prediction], linestyle='--', color=gpu_color, linewidth=1.5)
                ax1.plot(size, prediction, 'o', color=gpu_color, markersize=8, alpha=0.7)
                ax1.annotate(f'~{prediction/60:.1f}min', (size, prediction), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Sklearn HDBSCAN
        if sklearn_time is not None:
            last_sklearn = (size, sklearn_time)
        else:
            prediction = predict_completion_time(sample_sizes[:i], sklearn_times[:i], size)
            if prediction and last_sklearn:
                ax1.plot([last_sklearn[0], size], [last_sklearn[1], prediction], linestyle='--', color=sklearn_color, linewidth=1.5)
                ax1.plot(size, prediction, 's', color=sklearn_color, markersize=8, alpha=0.7)
                ax1.annotate(f'~{prediction/60:.1f}min', (size, prediction), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)

        # DBSCAN
        if dbscan_time is not None:
            last_dbscan = (size, dbscan_time)
        else:
            prediction = predict_completion_time(sample_sizes[:i], dbscan_times[:i], size)
            if prediction and last_dbscan:
                ax1.plot([last_dbscan[0], size], [last_dbscan[1], prediction], linestyle='--', color=dbscan_color, linewidth=1.5)
                ax1.plot(size, prediction, '^', color=dbscan_color, markersize=8, alpha=0.7)
                ax1.annotate(f'~{prediction/60:.1f}min', (size, prediction), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add timeout line
    ax1.axhline(y=timeout, color='red', linestyle='--', alpha=0.7, label=f'Timeout ({timeout/60:.0f}min)')
    
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('Execution Time (seconds)')
    # ax1.set_xscale('log') do not log 
    # ax1.set_yscale('log') do not log time
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory Usage Comparison
    ax2.set_title('Memory Usage vs Sample Size', fontweight='bold')
    
    memory_data = []
    for col in ['GPU_Memory', 'Sklearn_Memory', 'DBSCAN_Memory']:
        mask = results_df[col].notna()
        if mask.any():
            ax2.plot(results_df.loc[mask, 'Samples'], results_df.loc[mask, col], 
                    'o-', label=col.replace('_Memory', ''), linewidth=2, markersize=6)
    
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Clustering Quality (Homogeneity)
    ax3.set_title('Clustering Quality (Homogeneity Score)', fontweight='bold')
    
    for col in ['GPU_Homogeneity', 'Sklearn_Homogeneity', 'DBSCAN_Homogeneity']:
        mask = results_df[col].notna()
        if mask.any():
            ax3.plot(results_df.loc[mask, 'Samples'], results_df.loc[mask, col], 
                    'o-', label=col.replace('_Homogeneity', ''), linewidth=2, markersize=6)
    
    ax3.set_xlabel('Number of Samples')
    ax3.set_ylabel('Homogeneity Score')
    ax3.set_xscale('log')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Number of Clusters Found
    ax4.set_title('Number of Clusters Found', fontweight='bold')
    
    for col in ['GPU_Clusters', 'Sklearn_Clusters', 'DBSCAN_Clusters']:
        mask = results_df[col].notna()
        if mask.any():
            ax4.plot(results_df.loc[mask, 'Samples'], results_df.loc[mask, col], 
                    'o-', label=col.replace('_Clusters', ''), linewidth=2, markersize=6)
    
    ax4.set_xlabel('Number of Samples')
    ax4.set_ylabel('Number of Clusters')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_benchmark_comprehensive.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def print_benchmark_summary(results_df):
    """Print a comprehensive benchmark summary"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Performance summary
    print("\nEXECUTION TIME SUMMARY:")
    print("-" * 40)
    
    for _, row in results_df.iterrows():
        print(f"Sample Size: {row['Samples']:,}")
        
        gpu_time = row['GPU_Time'] if not row['GPU_Timeout'] else f">{row['GPU_Time']:.0f}s (timeout)"
        sklearn_time = row['Sklearn_Time'] if not row['Sklearn_Timeout'] else f">{row['Sklearn_Time']:.0f}s (timeout)"
        dbscan_time = row['DBSCAN_Time'] if not row['DBSCAN_Timeout'] else f">{row['DBSCAN_Time']:.0f}s (timeout)"
        
        print(f"  GPU HDBSCAN:    {gpu_time}")
        print(f"  Sklearn HDBSCAN: {sklearn_time}")
        print(f"  DBSCAN:         {dbscan_time}")
        print()
    
    # Calculate averages for non-timed-out results
    print("AVERAGE PERFORMANCE (completed runs only):")
    print("-" * 40)
    
    completed_gpu = results_df[~results_df['GPU_Timeout']]
    completed_sklearn = results_df[~results_df['Sklearn_Timeout']]
    completed_dbscan = results_df[~results_df['DBSCAN_Timeout']]
    
    if len(completed_gpu) > 0:
        avg_gpu_time = completed_gpu['GPU_Time'].mean()
        avg_gpu_homogeneity = completed_gpu['GPU_Homogeneity'].mean()
        print(f"GPU HDBSCAN - Avg Time: {avg_gpu_time:.2f}s, Avg Homogeneity: {avg_gpu_homogeneity:.3f}")
    
    if len(completed_sklearn) > 0:
        avg_sklearn_time = completed_sklearn['Sklearn_Time'].mean()
        avg_sklearn_homogeneity = completed_sklearn['Sklearn_Homogeneity'].mean()
        print(f"Sklearn HDBSCAN - Avg Time: {avg_sklearn_time:.2f}s, Avg Homogeneity: {avg_sklearn_homogeneity:.3f}")
    
    if len(completed_dbscan) > 0:
        avg_dbscan_time = completed_dbscan['DBSCAN_Time'].mean()
        avg_dbscan_homogeneity = completed_dbscan['DBSCAN_Homogeneity'].mean()
        print(f"DBSCAN - Avg Time: {avg_dbscan_time:.2f}s, Avg Homogeneity: {avg_dbscan_homogeneity:.3f}")

def run_speed_to_samples_benchmark(data_path, executable_path="./gpu_hdbscan_edited/build/gpu_hdbscan"):
    """
    Speed Benchmark of algos
    """
    if not os.path.exists(data_path):
        print("Data file not found!\n")
        return FileNotFoundError
    
    df = pd.read_csv(data_path)
    max_num_samples = len(df)
    print("Total num samples in File: ", max_num_samples)

    # Test Params
    output_dir = "speed_benchmark_outputs"
    os.makedirs(output_dir, exist_ok=True)

    num_samples_for_benchmark = [1000, 10000, 50000, 100000, 150000, 200000, 250000, 275000, 300000, 325000, 350000, 375000, 500000, 750000, 1000000]
    num_samples_for_benchmark = [n for n in num_samples_for_benchmark if n <= max_num_samples]

    feature_cols = ['PW(microsec)', 'FREQ(MHz)', 'AZ_S0(deg)', 'EL_S0(deg)', 'Latitude(deg)', 'Longitude(deg)']
    timeout = 5 * 60  # 5 minute timeout 
    emitter_col = "EmitterId"
    min_samples = 5
    epsilon = 0.39
    quiet_mode = True

    # Initialize algorithm wrapper
    gpu_hdbscan = GPUHDBSCANWrapper(executable_path=executable_path)
    
    results = []
    evaluation_results = []
    
    # Get true labels if available
    true_labels = df[emitter_col].values if emitter_col in df.columns else None
    
    # Scale data for DBSCAN
    scaler = StandardScaler()

    for num_samples in num_samples_for_benchmark:
        print(f"\nProcessing {num_samples} samples...")
        
        # sub_df = df[:num_samples]  # Fixed: was using wrong variable name
        sub_df = df.sample(n=num_samples, random_state=42)
        
        # Get true labels for this batch
        batch_true_labels = true_labels[:num_samples] if true_labels is not None else None
        
        # Try finding min points per emitter from all unique emitters in batch
        try: 
            min_cluster_size = max(20, find_min_points_per_emitter(sub_df, emitter_col))
        except Exception as e:
            print(f"Error processing file: {e}")
            min_cluster_size = 20
        
        # GPU HDBSCAN on the file directly
        print("  -> Running GPU HDBSCAN...")
        
        X = sub_df[feature_cols].to_numpy()
        X_scaled = scaler.fit_transform(X)
        dims = X.shape[1]

        # Create temporary CSV file
        temp_csv_file = create_temp_csv(sub_df)
        
        # Track timing & memory around that call
        gpu_labels, gpu_time, gpu_mem, gpu_timeout = track_performance_with_timeout(
            gpu_hdbscan.fit_predict_batched, 
            temp_csv_file,
            dims,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            quiet_mode=quiet_mode,
            timeout=timeout
        )
        
        # Clean up temp file
        if os.path.exists(temp_csv_file):
            os.remove(temp_csv_file)
        
        # sklearn HDBSCAN for comparison
        print("  -> Running sklearn HDBSCAN...")
        # For num_samples greater than show below, sk-learn implementation runs out of memory and fails silently.
        if num_samples > 250000:
            sklearn_labels, sklearn_time, sklearn_memory, sklearn_timeout = None, timeout, 0, True
        else:
            sklearn_model = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
            sklearn_labels, sklearn_time, sklearn_memory, sklearn_timeout = track_performance_with_timeout(
                sklearn_model.fit_predict, X, timeout=timeout
            )

        # DBSCAN for comparison
        print("  -> Running sklearn DBSCAN...")
        print(f"DBSCAN Parameters: min_samples={min_samples}, epsilon={epsilon}")
        # For num_samples greater than show below, sk-learn implementation runs out of memory and fails silently.
        if num_samples > 450000:
            dbscan_labels, dbscan_time, dbscan_memory, dbscan_timeout = None, timeout, 0, True
        else:
            dbscan_model = DBSCAN(eps=epsilon, min_samples=min_samples)
            dbscan_labels, dbscan_time, dbscan_memory, dbscan_timeout = track_performance_with_timeout(
                dbscan_model.fit_predict, X_scaled, timeout=timeout
            )
        
        # Calculate metrics
        gpu_homogeneity = homogeneity_score(batch_true_labels, gpu_labels) if batch_true_labels is not None and gpu_labels is not None and not gpu_timeout else None
        sklearn_homogeneity = homogeneity_score(batch_true_labels, sklearn_labels) if batch_true_labels is not None and sklearn_labels is not None and not sklearn_timeout else None
        dbscan_homogeneity = homogeneity_score(batch_true_labels, dbscan_labels) if batch_true_labels is not None and dbscan_labels is not None and not dbscan_timeout else None
        
        # Collect performance results 
        result = {
            'Samples': num_samples,  # Fixed: was using undefined variable
            'GPU_Time': gpu_time,
            'Sklearn_Time': sklearn_time,
            'DBSCAN_Time': dbscan_time,
            'GPU_Memory': gpu_mem,
            'Sklearn_Memory': sklearn_memory,
            'DBSCAN_Memory': dbscan_memory,
            'GPU_Clusters': len(np.unique(gpu_labels[gpu_labels != -1])) if gpu_labels is not None and not gpu_timeout else None,
            'Sklearn_Clusters': len(np.unique(sklearn_labels[sklearn_labels != -1])) if sklearn_labels is not None and not sklearn_timeout else None,
            'DBSCAN_Clusters': len(np.unique(dbscan_labels[dbscan_labels != -1])) if dbscan_labels is not None and not dbscan_timeout else None,
            'GPU_Homogeneity': gpu_homogeneity,
            'Sklearn_Homogeneity': sklearn_homogeneity,
            'DBSCAN_Homogeneity': dbscan_homogeneity,
            'GPU_Timeout': gpu_timeout,
            'Sklearn_Timeout': sklearn_timeout,
            'DBSCAN_Timeout': dbscan_timeout
        }
        results.append(result)
    
    # Save detailed summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'speed_benchmark_summary_with_accuracy.csv'), index=False)
    
    # Create visualization
    create_benchmark_plot(results_df, output_dir, timeout)
    
    # Print summary
    print_benchmark_summary(results_df)
    
    return results_df, evaluation_results

if __name__ == "__main__":
    # Make sure your executable is built
    executable_path = "./gpu_hdbscan_edited/build/gpu_hdbscan"
    if not os.path.exists(executable_path):
        print(f"Executable not found at {executable_path}")
        print("Please run 'make' to build the project first")
        exit(1)

    # Make sure data file path exists
    data_path = "./data/pdwInterns_with_latlng.csv"
    
    batch_path = "./data/batch_data"
    batch_interval = 2 # TOA Interval in seconds

    use_lat_lng = True

    speed_benchmark = True # Benchmark for speed

    if not os.path.exists(data_path):
        print(f"Date file not found at {data_path}")
        exit(1)

    if speed_benchmark:
        run_speed_to_samples_benchmark(data_path=data_path)

        print(f"\n Speed benchmark complete!'")
        print("Plots saved to 'gpu_hdbscan_speed_benchmark.png'")

    else:
        # if data path provided will use data file else will generate 2d data.
        if use_lat_lng:
            noisy_data_path = "./data/noisy_pdwInterns_with_latlng.csv"

            std_array = [1.0, 0.00021, 0.2, 0.2, 0.1, 0.1] 
            columns_to_noise = ["FREQ(MHz)", "PW(microsec)", "AZ_S0(deg)", "EL_S0(deg)", "Latitude(deg)", "Longitude(deg)"]
            std_map = dict(zip(columns_to_noise, std_array))
        else:
            noisy_data_path = "./data/noisy_pdwInterns.csv"

            std_array = [1.0, 0.00021, 0.2, 0.2] 
            columns_to_noise = ["FREQ(MHz)", "PW(microsec)", "AZ_S0(deg)", "EL_S0(deg)"]
            std_map = dict(zip(columns_to_noise, std_array))

        if not os.path.exists(noisy_data_path):
            print(f"Noisy Data not found at {noisy_data_path}")
            noisy_data_path = add_gaussian_noise(data_path, std_map=std_map)
        
        if not os.path.exists(batch_path):
            # Chunk size can be taken as the maximum number of points in a batch
            data = batch_data(data_path=noisy_data_path, batch_interval=batch_interval, chunk_size=200000, assume_sorted=True)

        results, eval_results  = run_benchmark_with_visualization_batched(data_path=batch_path,executable_path=executable_path,use_amp=False,use_toa=False, use_lat_lng=use_lat_lng) 

        print(f"\nBenchmark complete! Results saved to 'gpu_hdbscan_benchmark_results.csv'")
        print("Plots saved to 'gpu_hdbscan_benchmark.png'")
