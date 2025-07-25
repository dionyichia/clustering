import pandas as pd
import glob
import csv
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from collections import deque

def mst_from_data_matrix(raw_data, core_distances, alpha=1.0, dist_fn='euclidean'):
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

def mrd(n_neighbors,X):
    # Fit sklearn NearestNeighbors model
    sk_model = NearestNeighbors(n_neighbors,algorithm="kd_tree", metric="euclidean")
    sk_model.fit(X)
    neighbors_distances, neighbors_indices = sk_model.kneighbors(X, return_distance=True)

    # Core distances = distance to the kth neighbor
    core_distances = neighbors_distances[:, -1]

    # Build MRD graph
    n_samples = X.shape[0]
    mrd_graph = [[] for _ in range(n_samples)]

    for i in range(n_samples):
        for j_idx, j in enumerate(neighbors_indices[i]):
            d_ij = neighbors_distances[i][j_idx]
            c_i = core_distances[i]
            c_j = core_distances[j]
            mrd = max(c_i, c_j, d_ij)
            mrd_graph[i].append((j, mrd))

    return mrd_graph, core_distances

def load_mrd_graph_cpp(filename):
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
cpp_csv_path = "mrd_graph_output.csv"

for csv_file in csv_paths:
    if os.path.basename(csv_file) == "Data_Batch_6_60_emitters_200000_samples.csv":
        print(f"Checking: {os.path.basename(csv_file)}")
        
        # 1. Load CSV
        df = pd.read_csv(csv_file)
        X = df[feature_cols].to_numpy()

        # 2. Load your C++ KNN output
        cpp_mrd = load_mrd_graph_cpp(cpp_csv_path)  # change if per-file output

        # 3. Load Python MRD output
        py_mrd,core_distances = mrd(n_neighbors,X)

        # compareMRD(cpp_mrd,py_mrd,len(X))

        mst = mst_from_data_matrix(X,core_distances)

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

# Assuming these are the dtype structures based on typical hierarchical clustering
# You may need to adjust these based on your actual HIERARCHY_dtype and CONDENSED_dtype
HIERARCHY_dtype = np.dtype([
    ('left_node', np.intp),
    ('right_node', np.intp), 
    ('value', np.float64),
    ('cluster_size', np.intp)
])

CONDENSED_dtype = np.dtype([
    ('parent', np.intp),
    ('child', np.intp),
    ('lambda_val', np.float64),
    ('child_size', np.intp)
])

def bfs_from_hierarchy(hierarchy, root):
    """
    Breadth-first search traversal of hierarchy tree starting from root.
    Returns list of nodes in BFS order.
    """
    if root < len(hierarchy) + 1:  # leaf node
        return [root]
    
    n_samples = len(hierarchy) + 1
    queue = deque([root])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        if node >= n_samples:  # internal node
            children = hierarchy[node - n_samples]
            queue.append(children['left_node'])
            queue.append(children['right_node'])
    
    return result

def condense_tree(hierarchy, min_cluster_size=10):
    """
    Condense a tree according to a minimum cluster size. This is akin
    to the runt pruning procedure of Stuetzle. The result is a much simpler
    tree that is easier to visualize. We include extra information on the
    lambda value at which individual points depart clusters for later
    analysis and computation.

    Parameters
    ----------
    hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype
        A single linkage hierarchy in scipy.cluster.hierarchy format.

    min_cluster_size : int, optional (default 10)
        The minimum size of clusters to consider. Clusters smaller than this
        are pruned from the tree.

    Returns
    -------
    condensed_tree : ndarray of shape (n_samples,), dtype=CONDENSED_dtype
        Effectively an edgelist encoding a parent/child pair, along with a
        value and the corresponding cluster_size in each row providing a tree
        structure.
    """
    
    root = 2 * len(hierarchy)
    n_samples = len(hierarchy) + 1
    next_label = n_samples + 1
    node_list = bfs_from_hierarchy(hierarchy, root)
    
    # Initialize relabel array
    relabel = np.empty(root + 1, dtype=np.intp)
    relabel[root] = n_samples
    result_list = []
    ignore = np.zeros(len(node_list), dtype=bool)
    
    for i, node in enumerate(node_list):
        if ignore[i] or node < n_samples:
            continue
            
        children = hierarchy[node - n_samples]
        left = children['left_node']
        right = children['right_node'] 
        distance = children['value']
        
        if distance > 0.0:
            lambda_value = 1.0 / distance
        else:
            lambda_value = np.inf
            
        # Get cluster sizes
        if left >= n_samples:
            left_count = hierarchy[left - n_samples]['cluster_size']
        else:
            left_count = 1
            
        if right >= n_samples:
            right_count = hierarchy[right - n_samples]['cluster_size']
        else:
            right_count = 1
            
        # Process based on cluster sizes
        if left_count >= min_cluster_size and right_count >= min_cluster_size:
            # Both children are large enough clusters
            relabel[left] = next_label
            next_label += 1
            result_list.append((relabel[node], relabel[left], lambda_value, left_count))
            
            relabel[right] = next_label  
            next_label += 1
            result_list.append((relabel[node], relabel[right], lambda_value, right_count))
            
        elif left_count < min_cluster_size and right_count < min_cluster_size:
            # Both children are too small - add all leaf nodes
            for sub_node in bfs_from_hierarchy(hierarchy, left):
                if sub_node < n_samples:
                    result_list.append((relabel[node], sub_node, lambda_value, 1))
                # Mark nodes in left subtree as ignored
                for j, check_node in enumerate(node_list):
                    if check_node == sub_node:
                        ignore[j] = True
                        
            for sub_node in bfs_from_hierarchy(hierarchy, right):
                if sub_node < n_samples:
                    result_list.append((relabel[node], sub_node, lambda_value, 1))
                # Mark nodes in right subtree as ignored  
                for j, check_node in enumerate(node_list):
                    if check_node == sub_node:
                        ignore[j] = True
                        
        elif left_count < min_cluster_size:
            # Left child too small, inherit right child's label
            relabel[right] = relabel[node]
            for sub_node in bfs_from_hierarchy(hierarchy, left):
                if sub_node < n_samples:
                    result_list.append((relabel[node], sub_node, lambda_value, 1))
                # Mark nodes in left subtree as ignored
                for j, check_node in enumerate(node_list):
                    if check_node == sub_node:
                        ignore[j] = True
                        
        else:
            # Right child too small, inherit left child's label  
            relabel[left] = relabel[node]
            for sub_node in bfs_from_hierarchy(hierarchy, right):
                if sub_node < n_samples:
                    result_list.append((relabel[node], sub_node, lambda_value, 1))
                # Mark nodes in right subtree as ignored
                for j, check_node in enumerate(node_list):
                    if check_node == sub_node:
                        ignore[j] = True
    
    return np.array(result_list, dtype=CONDENSED_dtype)