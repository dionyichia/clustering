import subprocess
import numpy as np
import time
import psutil
import os
from typing import List, Dict,Tuple,Set,Optional
from sklearn.cluster import HDBSCAN,DBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
import pandas as pd
import glob
import threading
import gc


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
    initial_memory = process.memory_info().rss / 1024.0 / 1024.0  # MB
    
    # Time the execution
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    # Get peak memory (approximation)
    final_memory = process.memory_info().rss / 1024.0 / 1024.0  # MB
    memory_used = max(0, final_memory - initial_memory)
    
    execution_time = end_time - start_time
    
    return result, execution_time, memory_used

def sort_csv_by_toa(data_path: str, toa_column: str = "TOA(ns)") -> None:
    """
    Loads a CSV, sorts it globally by TOA(ns), and writes it back to the same file.

    Args:
        data_path (str): Path to the input CSV file.
        toa_column (str): Name of the column containing Time of Arrival in nanoseconds.
    """
    try:
        print(f"Reading CSV from {data_path}...")
        df = pd.read_csv(data_path)

        if toa_column not in df.columns:
            raise ValueError(f"Column '{toa_column}' not found in the CSV.")

        print(f"Sorting by '{toa_column}'...")
        df_sorted = df.sort_values(by=toa_column)

        print("Writing sorted data back to CSV...")
        df_sorted.to_csv(data_path, index=False)

        print("✅ CSV successfully sorted and saved.")

    except Exception as e:
        print(f"❌ Error while sorting CSV: {e}")



def batch_data_by_emitters_fixed_samples(
    data_path: str,
    num_emitters: int,
    max_samples: int = 200_000,
    toa_col: str = "TOA(ns)",
    emitter_col: str = "EmitterId",
    output_path: Optional[str] = None
) -> str:
    """
    Extract up to `max_samples` points by evenly sampling from the first
    `num_emitters` emitters (by order of appearance), taking the n = max_samples//num_emitters
    smallest‐TOA points per emitter, then sorting the combined set by TOA.

    Args:
        data_path:    Path to the input CSV.
        num_emitters: How many distinct emitters to include.
        max_samples:  Total maximum points to return (default 200,000).
        toa_col:      Column name for time‐of‐arrival.
        emitter_col:  Column name for emitter IDs.
        output_path:  Where to save the output CSV. If None, saves alongside
                      the input file as batch_{num_emitters}_{max_samples}.csv.

    Returns:
        The path to the newly created CSV.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No such file: {data_path}")

    # Load
    df = pd.read_csv(data_path)
    if emitter_col not in df.columns or toa_col not in df.columns:
        raise ValueError(f"Missing required column(s): {emitter_col}, {toa_col}")

    # Determine which emitters to use (first appearance)
    unique_emitters = df[emitter_col].drop_duplicates().tolist()
    if num_emitters > len(unique_emitters):
        raise ValueError(f"Requested {num_emitters} emitters but only "
                         f"{len(unique_emitters)} available.")
    selected = unique_emitters[:num_emitters]

    # How many samples per emitter?
    per_emitter = max_samples // num_emitters
    if per_emitter < 1:
        raise ValueError(f"max_samples ({max_samples}) too small for "
                         f"{num_emitters} emitters (need at least {num_emitters}).")

    # For each emitter, grab its per_emitter smallest TOA rows
    batches = []
    for eid in selected:
        sub = df[df[emitter_col] == eid]
        # sort just this emitter by TOA and take the first n
        topn = sub.nsmallest(per_emitter, columns=toa_col)
        batches.append(topn)

    # Combine and sort globally by TOA
    result = pd.concat(batches, ignore_index=True)
    result = result.sort_values(toa_col).reset_index(drop=True)

    # Decide output path
    if output_path is None:
        base, _ = os.path.splitext(os.path.basename(data_path))
        output_filename = f"{base}_batch_{num_emitters}emitters_{max_samples}samples.csv"
        output_path = os.path.join(os.path.dirname(data_path), output_filename)

    # Save
    result.to_csv(output_path, index=False)
    return output_path


def batch_data_by_emitters(data_path: str, emitters_per_batch_list: List[int] = None, 
                          assume_sorted: bool = True) -> List[Dict[str, any]]:
    """
    Process data by batching according to specified number of unique emitters for each batch.
    Loads entire dataset into memory for efficient processing.
    
    Args:
        data_path (str): Path to the CSV data file
        emitters_per_batch_list (List[int]): List specifying number of emitters for each batch.
                                           If None, defaults to [10] for a single batch.
        assume_sorted (bool): Whether to assume data is pre-sorted by TOA
    
    Returns:
        List of dictionaries in TestConfig format with keys:
        - name: str
        - data_type: str  
        - data_path: str
        - sample_size: int
        - batch_number: int
        - emitter_count: int
        - emitter_ids: list
    """
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    # Set default if no list provided
    if emitters_per_batch_list is None:
        emitters_per_batch_list = [10]
    
    # Validate the list
    if not emitters_per_batch_list or any(count <= 0 for count in emitters_per_batch_list):
        raise ValueError("All emitter counts must be positive integers")
    
    # Load entire dataset
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise RuntimeError(f"Error loading data file: {e}")
    
    if 'EmitterId' not in df.columns:
        raise ValueError("Required column 'EmitterId' not found")
    
    # Sort by TOA if not pre-sorted and TOA column exists
    if not assume_sorted and 'TOA(ns)' in df.columns:
        df = df.sort_values('TOA(ns)')
    
    # Create batch directory
    batch_dir = os.path.join(os.path.dirname(data_path), 'batch_data')
    os.makedirs(batch_dir, exist_ok=True)
    
    # Get all unique emitters in order of first appearance
    unique_emitters = df['EmitterId'].drop_duplicates().tolist()
    total_emitters = len(unique_emitters)
    
    print(f"Total emitters found: {total_emitters}")
    print(f"Requested batch sizes: {emitters_per_batch_list}")
    
    test_configs = []
    current_batch = 1
    
    try:
        for batch_size in emitters_per_batch_list:
            # Validate batch size
            if batch_size > total_emitters:
                raise ValueError(f"Batch size {batch_size} exceeds total available emitters ({total_emitters})")
            
            # Select first N emitters for this batch (can reuse emitters across batches)
            batch_emitters = unique_emitters[:batch_size]
            
            # Filter data for these emitters
            batch_data = df[df['EmitterId'].isin(batch_emitters)]
            
            if len(batch_data) == 0:
                print(f"No data found for batch {current_batch} emitters")
                continue
            
            # Save batch and create config
            config = _save_emitter_batch_and_create_config(
                batch_data, batch_dir, current_batch, set(batch_emitters)
            )
            test_configs.append(config)
            
            print(f"Batch {current_batch}: {len(batch_emitters)} emitters, {len(batch_data)} samples")
            
            current_batch += 1
        
        print(f"Processing complete: {len(test_configs)} batches created")
        return test_configs
        
    except Exception as e:
        raise RuntimeError(f"Error in emitter-based processing: {e}")


def _save_emitter_batch_and_create_config(batch_data: pd.DataFrame, batch_dir: str, batch_num: int,
                                         emitter_ids: Set[str]) -> Dict[str, any]:
    """
    Helper function to save a batch of data and create a TestConfig dictionary for emitter-based batching.
    
    Returns:
        Dictionary in TestConfig format
    """
    # Save batch file
    batch_filename = f"Data_Batch_{batch_num}.csv"
    batch_file_path = os.path.join(batch_dir, batch_filename)
    batch_data.to_csv(batch_file_path, index=False)
    
    # Create TestConfig dictionary
    config = {
        'name': f"EmitterBatch_{batch_num}_{len(batch_data)}samples_{len(emitter_ids)}emitters",
        'data_type': f"EmitterBatch_{batch_num}",
        'data_path': batch_file_path,
        'sample_size': len(batch_data),
        'batch_number': batch_num,
        'emitter_count': len(emitter_ids),
        'emitter_ids': sorted(list(emitter_ids))  # Convert to sorted list for consistency
    }
    
    return config


def get_total_emitters(data_path: str) -> int:
    """
    Get the total number of unique emitters in the dataset.
    
    Args:
        data_path (str): Path to the CSV data file
    
    Returns:
        int: Total number of unique emitters
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        if 'EmitterId' not in df.columns:
            raise ValueError("Required column 'EmitterId' not found")
        
        return df['EmitterId'].nunique()
    
    except Exception as e:
        raise RuntimeError(f"Error counting unique emitters: {e}")


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

    if not assume_sorted:
        sort_csv_by_toa(data_path=data_path, toa_column='TOA(ns)')
    
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
    """Enhanced benchmark function with cluster visualization and memory management"""
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
    timeout = 5 * 60  # 5 minute timeout 

    emitter_col = "EmitterId"

    if use_lat_lng:
        feature_cols = ['PW(microsec)', 'FREQ(MHz)', 'AZ_S0(deg)', 'EL_S0(deg)', 'Latitude(deg)', 'Longitude(deg)']
    else:
        feature_cols = ['PW(microsec)', 'FREQ(MHz)', 'AZ_S0(deg)', 'EL_S0(deg)']
    
    results = []
    evaluation_results = []
    
    if data_path is None:
        raise ValueError("Must supply data_path to batch_data folder")

    
    # 2) Grab every CSV in there
    csv_paths = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError("No .csv files found in batch_data folder")
    
    # Memory monitoring function
    def log_memory_usage(stage):
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"  Memory usage at {stage}: {memory_mb:.1f} MB")
        return memory_mb
    
    # Find Optimal Parameters for DBSCAN and HDBSCAN
    # 3) Loop over each batch file
    for i, csv_file in enumerate(csv_paths):
        batch_filename = os.path.basename(csv_file)
        batch_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        print(f"\n=== Processing batch {i+1}/{len(csv_paths)}: {batch_name} ===")
        log_memory_usage("start of batch")
        
        if batch_filename not in dbscan_params:
            print(f"No parameters found for {batch_filename}, skipping...")
            continue
        
        min_samples, epsilon = dbscan_params[batch_filename]

        # count data rows (minus header)
        with open(csv_file, 'r') as f:
            n_samples = sum(1 for _ in f) - 1
        
        print(f"  Processing {n_samples} rows")
        
        try:
            # Read data in a try-except block to handle memory issues
            df = pd.read_csv(csv_file)
            # log_memory_usage("after reading CSV")

            # Try finding min points per emitter from all unique emitters in batch
            try: 
                min_cluster_size = max(20, find_min_points_per_emitter(df, emitter_col))
            except Exception as e:
                print(f"Error processing file: {e}")

            if use_amp: feature_cols.append('Amp_S0(dBm)')
            if use_toa: feature_cols.append('TOA(ns)')

            start_time = df['TOA(ns)'].min()
            end_time = df['TOA(ns)'].max()
            
            X = df[feature_cols].to_numpy()
            # log_memory_usage("after creating feature matrix")
            
            # Scale the data for DBSCAN
            def scale_data_isolated(X):
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                # scaler goes out of scope automatically
                return X_scaled
 
            X_scaled = scale_data_isolated(X)
            # log_memory_usage("after scaling")

            true_labels = df['EmitterId'].to_numpy()
            
            # Clear df early to free memory
            del df
            gc.collect()
            # log_memory_usage("after clearing DataFrame")

            print(f"  Params used for run: min_cluster_size = {min_cluster_size}, min_samples = {min_samples}")
            
            # GPU HDBSCAN on the file directly
            print("  -> Running GPU HDBSCAN...")
            dims = X.shape[1]
            
            # gpu_labels, gpu_time, gpu_mem = track_performance(
            #     gpu_hdbscan.fit_predict_batched, csv_file,
            #     dims,
            #     min_samples=min_samples,
            #     min_cluster_size=min_cluster_size,
            #     quiet_mode=quiet_mode
            # )
            gpu_labels, gpu_time, gpu_mem, gpu_timeout = track_performance_with_timeout(
                gpu_hdbscan.fit_predict_batched, 
                csv_file,
                dims,
                min_samples=min_samples,
                min_cluster_size=min_cluster_size,
                quiet_mode=quiet_mode,
                timeout=timeout
            )
            log_memory_usage("after GPU HDBSCAN")
            
            # sklearn HDBSCAN for comparison
            print("  -> Running sklearn HDBSCAN...")
            if n_samples > 250000:
                sklearn_model,sklearn_labels, sklearn_time, sklearn_memory, sklearn_timeout = None,None, timeout, 0, True
            else:
                sklearn_model = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
                sklearn_labels, sklearn_time, sklearn_memory, sklearn_timeout = track_performance_with_timeout(
                    sklearn_model.fit_predict, X, timeout=timeout
                )
            # sklearn_model = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, cluster_selection_method='leaf')
            # sklearn_labels, sklearn_time, sklearn_memory = track_performance(
            #     sklearn_model.fit_predict, X
            # )
            
            # Explicitly delete the model to free memory
            del sklearn_model
            gc.collect()
            log_memory_usage("after sklearn HDBSCAN")

            # DBSCAN for comparison - with memory management
            print("  -> Running sklearn DBSCAN with optimal epsilon and minPoints...")
            print(f"DBSCAN Parameters: min_samples={min_samples}, epsilon={epsilon}")
            if n_samples > 450000:
                dbscan_model,dbscan_labels, dbscan_time, dbscan_memory, dbscan_timeout = None,None, timeout, 0, True
            else:
                dbscan_model = DBSCAN(eps=epsilon, min_samples=min_samples)
                dbscan_labels, dbscan_time, dbscan_memory, dbscan_timeout = track_performance_with_timeout(
                    dbscan_model.fit_predict, X_scaled, timeout=timeout
                ) 
            # # Create DBSCAN model in a separate scope for better memory management
            # dbscan_model = DBSCAN(eps=epsilon, min_samples=min_samples)
            # dbscan_labels, dbscan_time, dbscan_memory = track_performance(
            #     dbscan_model.fit_predict, X_scaled
            # )
            
            # Explicitly delete DBSCAN model and scaled data
            del dbscan_model
            del X_scaled

            gc.collect()
            log_memory_usage("after DBSCAN")
            
            # Evaluate clustering results with ground truth
            print("  -> Evaluating clustering results...")
            eval_results = evaluate_clustering_with_ground_truth(
                X=X, gpu_labels=gpu_labels, sklearn_labels=sklearn_labels, dbscan_labels=dbscan_labels, true_labels=true_labels,
                batch_name=batch_name, start_time=start_time, end_time=end_time, feature_names=feature_cols, 
                gpu_time=gpu_time, sklearn_time=sklearn_time, dbscan_time=dbscan_time, 
                gpu_mem=gpu_mem, sklearn_mem=sklearn_memory, dbscan_mem=dbscan_memory, save_dir=output_dir
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
                'GPU_Clusters': eval_results['gpu_metrics']['N_Clusters'] if gpu_labels is not None and not gpu_timeout else None,
                'Sklearn_Clusters': eval_results['sklearn_metrics']['N_Clusters'] if sklearn_labels is not None and not sklearn_timeout else None,
                'DBSCAN_Clusters': eval_results['dbscan_metrics']['N_Clusters'] if dbscan_labels is not None and not dbscan_timeout else None,
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
                'GPU_Timeout': gpu_timeout,
                'Sklearn_Timeout': sklearn_timeout,
                'DBSCAN_Timeout': dbscan_timeout
            }
            results.append(result)
            
            # Clean up variables at the end of each batch
            del X, gpu_labels, sklearn_labels, dbscan_labels, true_labels
            del eval_results, result
            gc.collect()
            gc.collect()
            
            log_memory_usage("end of batch")
            
        except MemoryError as e:
            print(f"Memory error processing batch {batch_name}: {e}")
            # Force garbage collection and continue to next batch
            gc.collect()
            continue
        except Exception as e:
            print(f"Error processing batch {batch_name}: {e}")
            # Clean up and continue
            gc.collect()
            continue
    
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

# Additional utility function for memory management
def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    gc.collect() 

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
    initial_memory = process.memory_info().rss / 1024.0 / 1024.0  # MB
    
    # Time the execution with timeout
    start_time = time.time()
    result, timed_out, exception = timeout_handler(func, args, kwargs, timeout)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    if timed_out:
        execution_time = timeout
        result = None
        print(f"Algo execution timed out > {timeout / 60} minutes")
        
    if exception:
        raise exception
    
    # Get peak memory (approximation)
    final_memory = process.memory_info().rss / 1024.0 / 1024.0  # MB
    memory_used = max(0, final_memory - initial_memory)
    
    return result, execution_time, memory_used, timed_out

def create_temp_csv(df):
    """Create temporary CSV file for GPU HDBSCAN"""
    temp_filename = f"temp_data_{int(time.time())}.csv"
    df.to_csv(temp_filename, index=False)
    return temp_filename

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


    for num_samples in num_samples_for_benchmark:
        print(f"\nProcessing {num_samples} samples...")
        
        sub_df = df[:num_samples]  # Fixed: was using wrong variable name
        # sub_df = df.sample(n=num_samples, random_state=42)
        
        # Get true labels for this batch
        batch_true_labels = true_labels[:num_samples] if true_labels is not None else None
        
        # Try finding min points per emitter from all unique emitters in batch
        try: 
            min_cluster_size = max(20, find_min_points_per_emitter(sub_df, emitter_col))
        except Exception as e:
            print(f"Error processing file: {e}")
            min_cluster_size = 20

        # Scale data for DBSCAN
        scaler = StandardScaler()
        
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
    create_speed_benchmark_plot(results_df, output_dir, timeout)
    
    # Print summary
    print_speed_benchmark_summary(results_df)
    
    return results_df, evaluation_results

if __name__ == "__main__":
    # Make sure your executable is built
    executable_path = "./gpu_hdbscan_edited/build/gpu_hdbscan"
    if not os.path.exists(executable_path):
        print(f"Executable not found at {executable_path}")
        print("Please run 'make' to build the project first")
        exit(1)

    batch_interval = 2 # TOA Interval in seconds

    speed_benchmark = False # Benchmark for speed

    use_lat_lng = False
    add_jitter = True
    add_jitter_n_noise = False
    batch_by_num_emitters = True

    data_path = "./data/pdwInterns_with_latlng.csv"
    batch_path = "./data/batch_data"

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
            std_array = [1.0, 0.00021, 0.2, 0.2, 0.1, 0.1] 
            columns_to_noise = ["FREQ(MHz)", "PW(microsec)", "AZ_S0(deg)", "EL_S0(deg)", "Latitude(deg)", "Longitude(deg)"]
            std_map = dict(zip(columns_to_noise, std_array))
        else:
            std_array = [1.0, 0.00021, 0.2, 0.2] 
            columns_to_noise = ["FREQ(MHz)", "PW(microsec)", "AZ_S0(deg)", "EL_S0(deg)"]
            std_map = dict(zip(columns_to_noise, std_array))
        
        if add_jitter:
            noisy_data_path = "./data/noisy_pdwInterns_with_latlng.csv"

            if batch_by_num_emitters:
                batch_path = "./data/batch_data_jitter_by_emitters"
            else:
                batch_path = "./data/batch_data_jitter"
            
            if add_jitter_n_noise:
                noisy_data_path = "./data/noisy_pdwInterns_with_latlng_n_random.csv"
                batch_path = "./data/batch_data_noise"

        if not os.path.exists(noisy_data_path):
            print(f"Noisy Data not found at {noisy_data_path}")
            noisy_data_path = add_gaussian_noise(data_path, std_map=std_map)
        
        if not os.path.exists(batch_path):
            if batch_by_num_emitters:
                emitters_per_batch = [10,20,30,40,50,60,70,80,90,100]
                data = batch_data_by_emitters(data_path=noisy_data_path, emitters_per_batch_list = emitters_per_batch, assume_sorted = True)
            else:# Chunk size can be taken as the maximum number of points in a batch
                data = batch_data(data_path=noisy_data_path, batch_interval=batch_interval, chunk_size=200000, assume_sorted=True)

        results, eval_results  = run_benchmark_with_visualization_batched(data_path=batch_path,executable_path=executable_path,use_amp=False,use_toa=False, use_lat_lng=use_lat_lng) 

        print(f"\nBenchmark complete! Results saved to 'gpu_hdbscan_benchmark_results.csv'")
        print("Plots saved to 'gpu_hdbscan_benchmark.png'")
