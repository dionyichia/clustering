import pandas as pd
import glob
import csv
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


def mst_from_data_matrix_python(raw_data, core_distances, alpha=1.0, dist_fn='euclidean'):
    """
    Construct the MST from mutual-reachability distances.
    
    Parameters
    ----------
    raw_data : ndarray of shape (n_samples, n_features)
        Input array of data samples.
    core_distances : ndarray of shape (n_samples,)
        Core-distance for each sample.
    alpha : float
        Scaling factor for distance.
    dist_fn : str or callable
        Distance function (e.g., 'euclidean' or a custom function).
    
    Returns
    -------
    mst_edges : list of tuples
        Each tuple is (source, target, distance) representing an MST edge.
    """
    n_samples, n_features = raw_data.shape

    # Setup arrays
    in_tree = np.zeros(n_samples, dtype=bool)
    min_reachability = np.full(n_samples, np.inf)
    current_sources = np.full(n_samples, -1, dtype=int)

    mst_edges = []
    current_node = 0

    for _ in range(n_samples - 1):
        in_tree[current_node] = True
        current_node_core_dist = core_distances[current_node]

        new_reachability = np.inf
        source_node = current_node
        new_node = -1

        for j in range(n_samples):
            if in_tree[j]:
                continue

            # Compute pairwise distance
            pair_dist = np.linalg.norm(raw_data[current_node] - raw_data[j]) if dist_fn == 'euclidean' else dist_fn(raw_data[current_node], raw_data[j])
            pair_dist /= alpha

            next_node_core_dist = core_distances[j]
            mutual_reachability = max(current_node_core_dist, next_node_core_dist, pair_dist)

            if mutual_reachability < min_reachability[j]:
                min_reachability[j] = mutual_reachability
                current_sources[j] = current_node
                if mutual_reachability < new_reachability:
                    new_reachability = mutual_reachability
                    source_node = current_node
                    new_node = j
            elif min_reachability[j] < new_reachability:
                new_reachability = min_reachability[j]
                source_node = current_sources[j]
                new_node = j

        assert new_node != -1, "No new node was added — graph may be disconnected or something is wrong."
        mst_edges.append((source_node, new_node, new_reachability))
        current_node = new_node

    return mst_edges


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

def compareMRD(cpp_knn,mrd_graph,n_points):
    for i in range(n_points):
        cpp_nbrs = cpp_knn[i]
        sk_nbrs = mrd_graph[i]

    for (cpp_idx, cpp_d), (sk_idx, sk_d) in zip(cpp_nbrs, sk_nbrs):
        if cpp_idx != sk_idx or not np.isclose(cpp_d, sk_d, atol=1e-6):
            print(f"Mismatch in {os.path.basename(csv_file)} at point {i}:")
            print(f"  CPP   : idx={cpp_idx}, dist={cpp_d}")
            print(f"  SKL   : idx={sk_idx}, dist={sk_d}")

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
        neighbors_distances, neighbors_indices = sk_model.kneighbors(X, return_distance=True)

        # Core distances = distance to the kth neighbor
        core_distances = neighbors_distances[:, -1]

        # # === COMPARE MRD ===
        # # Build MRD graph
        # n_samples = X.shape[0]
        # mrd_graph = [[] for _ in range(n_samples)]

        # for i in range(n_samples):
        #     for j_idx, j in enumerate(neighbors_indices[i]):
        #         d_ij = neighbors_distances[i][j_idx]
        #         c_i = core_distances[i]
        #         c_j = core_distances[j]
        #         mrd = max(c_i, c_j, d_ij)
        #         mrd_graph[i].append((j, mrd))


        # compareMRD(cpp_knn,mrd_graph,len(X))

        mst = mst_from_data_matrix_python(X,core_distances)
class UnionFind:
    def __init__(self, N):
        self.parent = np.full(2 * N - 1, -1, dtype=np.intp, order='C')
        self.next_label = N
        self.size = np.hstack((np.ones(N, dtype=np.intp),
                               np.zeros(N - 1, dtype=np.intp)))

    def union(self, m, n):
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.next_label += 1

    def fast_find(self, n):
        p = n
        # find the highest node in the linkage graph so far
        while self.parent[n] != -1:
            n = self.parent[n]
        # provide a shortcut up to the highest node
        while self.parent[p] != n:
            p, self.parent[p] = self.parent[p], n
        return n

def make_single_linkage(mst):
    """Construct a single-linkage tree from an MST.
    
    Parameters
    ----------
    mst : ndarray of shape (n_samples - 1,)
        The MST representation of the mutual-reachability graph. The MST is
        represented as a collection of edges. Each edge should have attributes:
        - current_node
        - next_node  
        - distance
    
    Returns
    -------
    single_linkage : ndarray of shape (n_samples - 1,)
        The single-linkage tree (dendrogram) built from the MST. Each
        row represents:
        - left node/cluster
        - right node/cluster
        - distance
        - new cluster size
    """
    # Note mst.shape[0] is one fewer than the number of samples
    n_samples = mst.shape[0] + 1
    U = UnionFind(n_samples)
    
    # Create structured array for single linkage output
    # Assuming HIERARCHY_dtype structure based on usage
    dtype = [('left_node', np.intp), 
             ('right_node', np.intp), 
             ('value', np.float64), 
             ('cluster_size', np.intp)]
    
    single_linkage = np.zeros(n_samples - 1, dtype=dtype)
    
    for i in range(n_samples - 1):
        current_node = mst[i]['current_node']  # or mst[i].current_node if using structured array
        next_node = mst[i]['next_node']        # or mst[i].next_node
        distance = mst[i]['distance']          # or mst[i].distance
        
        current_node_cluster = U.fast_find(current_node)
        next_node_cluster = U.fast_find(next_node)
        
        single_linkage[i]['left_node'] = current_node_cluster
        single_linkage[i]['right_node'] = next_node_cluster
        single_linkage[i]['value'] = distance
        single_linkage[i]['cluster_size'] = U.size[current_node_cluster] + U.size[next_node_cluster]
        
        U.union(current_node_cluster, next_node_cluster)
    
    return single_linkage

def _process_mst(min_spanning_tree):
    """
    Builds a single-linkage tree (SLT) from the provided minimum spanning tree
    (MST). The MST is first sorted then processed by a custom Cython routine.

    Parameters
    ----------
    min_spanning_tree : ndarray of shape (n_samples - 1,), dtype=MST_edge_dtype
        The MST representation of the mutual-reachability graph. The MST is
        represented as a collection of edges.

    Returns
    -------
    single_linkage : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype
        The single-linkage tree tree (dendrogram) built from the MST.
    """
    # Sort edges of the min_spanning_tree by weight
    row_order = np.argsort(min_spanning_tree["distance"])
    min_spanning_tree = min_spanning_tree[row_order]
    # Convert edge list into standard hierarchical clustering format
    return make_single_linkage(min_spanning_tree)
