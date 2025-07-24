import pandas as pd
import glob
import csv
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_knn_graph_cpp(filename):
    cpp_knn = []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            i = int(row[0])
            neighbors = []
            for j in range(1, len(row), 2):
                nbr_idx = int(row[j])
                dist = float(row[j + 1])
                neighbors.append((nbr_idx, dist))
            cpp_knn.append(neighbors)
    return cpp_knn

n_neighbors = 5
feature_cols = ['PW(microsec)', 'FREQ(MHz)', 'AZ_S0(deg)', 'EL_S0(deg)']
data_path = './data/batch_data_jitter_by_emitter_n_time'
csv_paths = sorted(glob.glob(os.path.join(data_path, "*.csv")))
cpp_csv_path = "knn_graph_output.csv"
for csv_file in csv_paths:
    if os.path.basename(csv_file) == "Data_Batch_6_60_emitters_200000_samples.csv":
        print(f"Checking: {os.path.basename(csv_file)}")
        
        # 1. Load CSV
        df = pd.read_csv(csv_file)
        X = df[feature_cols].to_numpy()

        # 2. Load your C++ KNN output
        cpp_knn = load_knn_graph_cpp(cpp_csv_path)  # change if per-file output

        # 3. Fit sklearn NearestNeighbors model
        sk_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree", metric="euclidean")
        sk_model.fit(X)
        sk_dists, sk_inds = sk_model.kneighbors(X, return_distance=True)

        # 4. Compare
        for i in range(len(X)):
            cpp_nbrs = cpp_knn[i]
            sk_nbrs = list(zip(sk_inds[i], sk_dists[i]))

            for (cpp_idx, cpp_d), (sk_idx, sk_d) in zip(cpp_nbrs, sk_nbrs):
                if cpp_idx != sk_idx or not np.isclose(cpp_d, sk_d, atol=1e-6):
                    print(f"Mismatch in {os.path.basename(csv_file)} at point {i}:")
                    print(f"  CPP   : idx={cpp_idx}, dist={cpp_d}")
                    print(f"  SKL   : idx={sk_idx}, dist={sk_d}")
