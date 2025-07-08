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
            min_cluster_size = max(20,find_min_points_per_emitter(df, emitter_col))
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

    if not os.path.exists(data_path):
        print(f"Date file not found at {data_path}")
        exit(1)
    
    # Run benchmark
    use_lat_lng = True

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
        data = batch_data(data_path=data_path, batch_interval=batch_interval, chunk_size=200000, assume_sorted=True)

    results, eval_results  = run_benchmark_with_visualization_batched(data_path=batch_path,executable_path=executable_path,use_amp=False,use_toa=False, use_lat_lng=use_lat_lng) 

    print(f"\nBenchmark complete! Results saved to 'gpu_hdbscan_benchmark_results.csv'")
    print("Plots saved to 'gpu_hdbscan_benchmark.png'")
