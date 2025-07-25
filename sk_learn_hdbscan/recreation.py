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

# SAMPLE USAGE FOR CONDENSE_TREE
"""
    # Create sample hierarchy data
    # This would typically come from scipy.cluster.hierarchy.linkage
    sample_hierarchy = np.array([
        (0, 1, 0.1, 2),  # merge nodes 0,1 at distance 0.1, cluster size 2
        (2, 4, 0.2, 3),  # merge node 2 with cluster 4 at distance 0.2, size 3  
        (3, 5, 0.5, 4),  # final merge at distance 0.5, total size 4
    ], dtype=HIERARCHY_dtype)
    
    condensed = condense_tree(sample_hierarchy, min_cluster_size=2)
    print("Condensed tree:")
    print(condensed)
"""
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


import numpy as np
from collections import deque

def _get_clusters(
    condensed_tree,
    stability,
    cluster_selection_method='eom',
    allow_single_cluster=False,
    cluster_selection_epsilon=0.0,
    max_cluster_size=None
):
    """Given a tree and stability dict, produce the cluster labels
    (and probabilities) for a flat clustering based on the chosen
    cluster selection method.

    Parameters
    ----------
    condensed_tree : ndarray of shape (n_samples,), dtype=structured array
        Effectively an edgelist encoding a parent/child pair, along with a
        value and the corresponding cluster_size in each row providing a tree
        structure.

    stability : dict
        A dictionary mapping cluster_ids to stability values

    cluster_selection_method : string, optional (default 'eom')
        The method of selecting clusters. The default is the
        Excess of Mass algorithm specified by 'eom'. The alternate
        option is 'leaf'.

    allow_single_cluster : boolean, optional (default False)
        Whether to allow a single cluster to be selected by the
        Excess of Mass algorithm.

    cluster_selection_epsilon: float, optional (default 0.0)
        A distance threshold for cluster splits.

    max_cluster_size: int, default=None
        The maximum size for clusters located by the EOM clusterer. Can
        be overridden by the cluster_selection_epsilon parameter in
        rare cases.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        An integer array of cluster labels, with -1 denoting noise.

    probabilities : ndarray (n_samples,)
        The cluster membership strength of each sample.

    stabilities : ndarray (n_clusters,)
        The cluster coherence strengths of each cluster.
    """
    
    # Assume clusters are ordered by numeric id equivalent to
    # a topological sort of the tree
    if allow_single_cluster:
        node_list = sorted(stability.keys(), reverse=True)
    else:
        node_list = sorted(stability.keys(), reverse=True)[:-1]  # exclude root

    cluster_tree = condensed_tree[condensed_tree['cluster_size'] > 1]
    is_cluster = {cluster: True for cluster in node_list}
    n_samples = np.max(condensed_tree[condensed_tree['cluster_size'] == 1]['child']) + 1

    if max_cluster_size is None:
        max_cluster_size = n_samples + 1  # Set to a value that will never be triggered
    
    cluster_sizes = {
        child: cluster_size for child, cluster_size
        in zip(cluster_tree['child'], cluster_tree['cluster_size'])
    }
    
    if allow_single_cluster:
        # Compute cluster size for the root node
        cluster_sizes[node_list[-1]] = np.sum(
            cluster_tree[cluster_tree['parent'] == node_list[-1]]['cluster_size'])

    if cluster_selection_method == 'eom':
        for node in node_list:
            child_selection = (cluster_tree['parent'] == node)
            subtree_stability = np.sum([
                stability[child] for
                child in cluster_tree['child'][child_selection]])
            
            if subtree_stability > stability[node] or cluster_sizes[node] > max_cluster_size:
                is_cluster[node] = False
                stability[node] = subtree_stability
            else:
                for sub_node in bfs_from_cluster_tree(cluster_tree, node):
                    if sub_node != node:
                        is_cluster[sub_node] = False

        if cluster_selection_epsilon != 0.0 and len(cluster_tree) > 0:
            eom_clusters = [c for c in is_cluster if is_cluster[c]]
            selected_clusters = []
            # first check if eom_clusters only has root node, which skips epsilon check.
            if (len(eom_clusters) == 1 and eom_clusters[0] == cluster_tree['parent'].min()):
                if allow_single_cluster:
                    selected_clusters = eom_clusters
            else:
                selected_clusters = epsilon_search(
                    set(eom_clusters),
                    cluster_tree,
                    cluster_selection_epsilon,
                    allow_single_cluster
                )
            for c in is_cluster:
                if c in selected_clusters:
                    is_cluster[c] = True
                else:
                    is_cluster[c] = False

    elif cluster_selection_method == 'leaf':
        leaves = set(get_cluster_tree_leaves(cluster_tree))
        if len(leaves) == 0:
            for c in is_cluster:
                is_cluster[c] = False
            is_cluster[condensed_tree['parent'].min()] = True

        if cluster_selection_epsilon != 0.0:
            selected_clusters = epsilon_search(
                leaves,
                cluster_tree,
                cluster_selection_epsilon,
                allow_single_cluster
            )
        else:
            selected_clusters = leaves

        for c in is_cluster:
            if c in selected_clusters:
                is_cluster[c] = True
            else:
                is_cluster[c] = False

    clusters = set([c for c in is_cluster if is_cluster[c]])
    cluster_map = {c: n for n, c in enumerate(sorted(list(clusters)))}
    reverse_cluster_map = {n: c for c, n in cluster_map.items()}

    labels = _do_labelling(
        condensed_tree,
        clusters,
        cluster_map,
        allow_single_cluster,
        cluster_selection_epsilon
    )
    probs = get_probabilities(condensed_tree, reverse_cluster_map, labels)

    return (labels, probs)


def bfs_from_cluster_tree(cluster_tree, bfs_root):
    """Breadth-first search from a given root node in the cluster tree."""
    result = []
    process_queue = np.array([bfs_root], dtype=int)
    children = cluster_tree['child']
    parents = cluster_tree['parent']
    
    while len(process_queue) > 0:
        result.extend(process_queue.tolist())
        process_queue = children[np.isin(parents, process_queue)]
    
    return result


def get_cluster_tree_leaves(cluster_tree):
    """Get the leaf nodes of the cluster tree."""
    if len(cluster_tree) == 0:
        return []
    root = cluster_tree['parent'].min()
    return recurse_leaf_dfs(cluster_tree, root)


def epsilon_search(leaves, cluster_tree, cluster_selection_epsilon, allow_single_cluster):
    """Search for clusters within epsilon distance threshold."""
    selected_clusters = []
    processed = []
    children = cluster_tree['child']
    distances = cluster_tree['value']
    
    for leaf in leaves:
        leaf_nodes = children == leaf
        eps = 1 / distances[leaf_nodes][0]
        if eps < cluster_selection_epsilon:
            if leaf not in processed:
                epsilon_child = traverse_upwards(
                    cluster_tree,
                    cluster_selection_epsilon,
                    leaf,
                    allow_single_cluster
                )
                selected_clusters.append(epsilon_child)
                for sub_node in bfs_from_cluster_tree(cluster_tree, epsilon_child):
                    if sub_node != epsilon_child:
                        processed.append(sub_node)
        else:
            selected_clusters.append(leaf)
    
    return set(selected_clusters)

import numpy as np

def max_lambdas(condensed_tree):
    """Calculate maximum lambda values for each cluster."""
    # Find the largest parent ID to size the deaths array
    largest_parent = condensed_tree['parent'].max()
    deaths = np.zeros(largest_parent + 1, dtype=np.float64)
    
    # Initialize with first element
    current_parent = condensed_tree[0]['parent']
    max_lambda = condensed_tree[0]['value']
    
    # Iterate through the condensed tree starting from index 1
    for idx in range(1, len(condensed_tree)):
        parent = condensed_tree[idx]['parent']
        lambda_val = condensed_tree[idx]['value']
        
        if parent == current_parent:
            # Same parent, update max lambda
            max_lambda = max(max_lambda, lambda_val)
        else:
            # New parent, store the max lambda for previous parent
            deaths[current_parent] = max_lambda
            current_parent = parent
            max_lambda = lambda_val
    
    # Store the max lambda for the last parent
    deaths[current_parent] = max_lambda
    
    return deaths


def recurse_leaf_dfs(cluster_tree, current_node):
    """Recursively find leaf nodes using depth-first search."""
    # Find children of the current node
    children = cluster_tree[cluster_tree['parent'] == current_node]['child']
    
    if len(children) == 0:
        # No children, this is a leaf node
        return [current_node]
    else:
        # Has children, recursively get leaves from each child
        leaves = []
        for child in children:
            leaves.extend(recurse_leaf_dfs(cluster_tree, child))
        return leaves


def traverse_upwards(cluster_tree, cluster_selection_epsilon, leaf, allow_single_cluster):
    """Traverse upwards in the tree to find appropriate cluster within epsilon."""
    # Find the root of the tree
    root = cluster_tree['parent'].min()
    
    # Find the parent of the current leaf
    parent_mask = cluster_tree['child'] == leaf
    if not np.any(parent_mask):
        return leaf  # No parent found, return the leaf itself
    
    parent = cluster_tree[parent_mask]['parent'][0]
    
    # If parent is root, handle based on allow_single_cluster flag
    if parent == root:
        if allow_single_cluster:
            return parent
        else:
            return leaf  # return node closest to root
    
    # Calculate parent epsilon (1 / lambda value)
    parent_mask = cluster_tree['child'] == parent
    if not np.any(parent_mask):
        return leaf  # No parent info found
    
    parent_eps = 1.0 / cluster_tree[parent_mask]['value'][0]
    
    # Check if parent epsilon is within the threshold
    if parent_eps > cluster_selection_epsilon:
        return parent
    else:
        # Recursively traverse upwards
        return traverse_upwards(
            cluster_tree,
            cluster_selection_epsilon,
            parent,
            allow_single_cluster
        )


def get_probabilities(condensed_tree, cluster_map, labels):
    """Calculate cluster membership probabilities."""
    # Extract arrays from the structured array
    child_array = condensed_tree['child']
    parent_array = condensed_tree['parent']
    lambda_array = condensed_tree['value']
    
    # Initialize result array
    result = np.zeros(labels.shape[0], dtype=np.float64)
    
    # Get death lambdas for all clusters
    deaths = max_lambdas(condensed_tree)
    
    # Find the root cluster
    root_cluster = np.min(parent_array)
    
    # Process each entry in the condensed tree
    for n in range(len(condensed_tree)):
        point = child_array[n]
        
        # Skip if point is a cluster (>= root_cluster)
        if point >= root_cluster:
            continue
        
        # Get the cluster number for this point
        cluster_num = labels[point]
        
        # Skip noise points (cluster -1)
        if cluster_num == -1:
            continue
        
        # Get the actual cluster ID from the cluster map
        cluster = cluster_map[cluster_num]
        
        # Get the maximum lambda for this cluster
        max_lambda = deaths[cluster]
        
        # Calculate probability
        if max_lambda == 0.0 or np.isinf(lambda_array[n]):
            result[point] = 1.0
        else:
            lambda_val = min(lambda_array[n], max_lambda)
            result[point] = lambda_val / max_lambda
    
    return result