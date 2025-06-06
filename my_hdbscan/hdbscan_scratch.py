import heapq
from math import inf  
from hdbscan_plot import *

def euclid_dist(pointA, pointB):
    """Calculate Euclidean distance between two points"""
    return sum((a - b)**2 for a, b in zip(pointA, pointB))**0.5

def compute_dist_to_neighbours(data_pts, current_pt):
    """Compute distances from current point to all other points"""
    # BUG FIX: This function should actually calculate distances, not just return a list
    distances = []
    for point in data_pts:
        if point != current_pt:
            dist = euclid_dist(point, current_pt)
            distances.append((point, dist))
        else:
            distances.append((point, float('inf')))  # Fixed 'inf' syntax
    return distances

def find_nearest_neighbour_dist(dist_to_other_pts, k=1):
    """Find the k-th nearest neighbor based on distances"""
    # BUG FIX: The original implementation only found the single nearest neighbor
    # and didn't handle the k parameter needed for core distance calculation
    sorted_points = sorted(dist_to_other_pts, key=lambda x: x[1])
    if k < len(sorted_points):
        return sorted_points[k][1]
    return None

def compute_MRD(data_pts, core_dist):
    """Compute Mutual Reachability Distance matrix"""
    # BUG FIX: Complete rewrite as the original function had multiple issues
    n = len(data_pts)
    mrd = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                mrd[i][j] = 0
                continue
            
            # Calculate Euclidean distance
            dist = euclid_dist(data_pts[i], data_pts[j])
            
            # Mutual reachability distance is max of core distances and actual distance
            mrd[i][j] = max(core_dist[i], core_dist[j], dist)
    
    return mrd

class UF:
    """Union-Find data structure with component tracking"""
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size
        self.components = {i: [i] for i in range(size)}  # Track elements in each component

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u == root_v:
            return False

        # Union by size
        if self.size[root_u] < self.size[root_v]:
            root_u, root_v = root_v, root_u

        self.parent[root_v] = root_u
        self.size[root_u] += self.size[root_v]
        self.components[root_u].extend(self.components[root_v])
        del self.components[root_v]
        return True

    def get_size(self, u):
        return self.size[self.find(u)]

    def get_elements(self, u):
        return self.components[self.find(u)]

def construct_tree_kruskal(mutual_reachability_dist):
    """Construct MST using Kruskal's algorithm"""
    n = len(mutual_reachability_dist)
    heap = []
    
    # Build a min heap of all upper triangle distances
    for i in range(n):
        for j in range(i + 1, n):
            weight = mutual_reachability_dist[i][j]
            heapq.heappush(heap, (weight, i, j))

    parent = UF(n)
    mst = []

    while heap and len(mst) < n - 1:
        weight, u, v = heapq.heappop(heap)
        if parent.union(u, v):
            mst.append((u, v, weight))
    
    return mst

def get_density_cluster_plot(mst, min_cluster_size):
    """Create the condensed cluster tree"""
    # BUG FIX: Complete rewrite of this function as it had multiple issues
    n = max(max(u, v) for u, v, _ in mst) + 1  # Get number of nodes
    parent = UF(n)
    condensed_tree = {}
    cluster_id_counter = n  # Start cluster IDs after node IDs
    lambda_max = -1
    
    # Sort MST edges by weight (ascending)
    sorted_edges = sorted(mst, key=lambda x: x[2])
    
    # Track cluster births and lambdas
    cluster_births = {}  # When clusters are created
    
    for (u, v, weight) in sorted_edges:
        lambda_val = 1.0 / weight if weight > 0 else -1
        lambda_max = max(lambda_max, lambda_val)
        
        root_u = parent.find(u)
        root_v = parent.find(v)
        
        # Skip if already in same component
        if root_u == root_v:
            continue
        
        # Check if components are large enough to be clusters
        size_u = parent.get_size(root_u)
        size_v = parent.get_size(root_v)
        
        # If clusters are dying, record their death, If size >= min_cluster_size, they would already have been a cluster, if its < then no cluster, so no need record death
        if size_u >= min_cluster_size and root_u in cluster_births:
            cluster_id = cluster_births[root_u]
            condensed_tree[cluster_id]['lambda_death'] = lambda_val

        if size_v >= min_cluster_size and root_v in cluster_births:
            cluster_id = cluster_births[root_v]
            condensed_tree[cluster_id]['lambda_death'] = lambda_val
        
        # Merge the components
        parent.union(u, v)
        new_root = parent.find(u)
        new_size = parent.get_size(new_root)
        
        # Create a new cluster if large enough
        if new_size >= min_cluster_size:
            cluster_id_counter += 1
            
            condensed_tree[cluster_id_counter] = {
                'id': cluster_id_counter,
                'size': new_size,
                'points': parent.get_elements(new_root)[:],
                'lambda_birth': lambda_val,
                'lambda_death': float('inf'),  # Will be set later if the cluster dies
                'children': [],
                'parent': None
            }
            
            if root_u in cluster_births:
                # Add smaller root_u cluster as child of new big cluster
                condensed_tree[cluster_id_counter]['children'].append(cluster_births[root_u])
                # add new big cluster to be parent of smaller dead root_u cluster
                condensed_tree[cluster_births[root_u]]['parent'] = cluster_id_counter
                del cluster_births[root_u]
                    
            if root_v in cluster_births:
                condensed_tree[cluster_id_counter]['children'].append(cluster_births[root_v])
                condensed_tree[cluster_births[root_v]]['parent'] = cluster_id_counter
                del cluster_births[root_v]
                
            cluster_births[new_root] = cluster_id_counter
                
    return condensed_tree, lambda_max

def extract_clusters(condensed_tree, lambda_max):
    """Extract stable clusters using HDBSCAN's EOM pruning approach"""

    # Compute stability of each cluster
    for cluster in condensed_tree.values():
        birth = cluster['lambda_birth']
        death = cluster['lambda_death'] if cluster['lambda_death'] != float('inf') else lambda_max
        size = cluster['size']
        cluster['stability'] = (birth - death) * size
        cluster['selected'] = False
        
    # Step 1: Identify all leaf clusters (no children)
    leaf_clusters = [
        c_id for c_id, cluster in condensed_tree.items() if 'children' not in cluster or not cluster['children']
    ]

    # Step 2: Walk from leaves upward and apply EOM pruning
    visited = set()
    def post_order_prune(cluster_id):
        cluster = condensed_tree[cluster_id]
        visited.add(cluster_id)

        if 'children' not in cluster or not cluster['children']:
            # Leaf cluster: select it
            cluster['selected'] = True
            return cluster['stability']

        child_stability_sum = 0
        for child_id in cluster['children']:
            if child_id not in visited and child_id in condensed_tree:
                child_stability_sum += post_order_prune(child_id)

        if cluster['stability'] >= child_stability_sum:
            # Parent wins: deselect children, select parent
            cluster['selected'] = True
            for child_id in cluster['children']:
                condensed_tree[child_id]['selected'] = False
        else:
            cluster['selected'] = False  # Children win

        return max(cluster['stability'], child_stability_sum)

    # Start from all roots (clusters with no parent)
    roots = [
        c_id for c_id, cluster in condensed_tree.items()
        if 'parent' not in cluster or cluster['parent'] is None
    ]
    for root in roots:
        post_order_prune(root)

    # Step 3: Assign points from selected clusters only
    point_assignments = {}
    point_stabilities = {}

    for cluster in condensed_tree.values():
        if cluster['selected']:
            for point in cluster['points']:
                stability = cluster['stability']
                if stability > point_stabilities.get(point, 0):
                    point_stabilities[point] = stability
                    point_assignments[point] = cluster['id']

    return point_assignments


def add_noise_points(data_pts, labelled_points):
    """Add unlabelled points as noise (cluster None)"""
    # BUG FIX: This function should modify labelled_points and return it
    result = dict(labelled_points)  # Make a copy
    
    for i, point in enumerate(data_pts):
        if i not in result:
            result[i] = None  # Mark as noise
            
    return result

def my_hdbscan(data_pts, min_samples, min_cluster_size):
    """
    HDBSCAN clustering algorithm implementation
    
    Input:
        - data_pts: dataset of n points
        - min_samples: minimum number of points to define a dense region
        - min_cluster_size: minimum size of a valid cluster

    Output:
        - Cluster labels for each point in data_pts
    """
    n = len(data_pts)
    
    # Step 1: Compute Core Distances
    core_dist = [0] * n
    
    for i in range(n):
        distance_to_other_points = compute_dist_to_neighbours(data_pts, data_pts[i])
        core_dist[i] = find_nearest_neighbour_dist(distance_to_other_points, min_samples)
    
            
    plot_core_distances(data_pts, core_dist)
    
    # Step 2: Compute Mutual Reachability Distances
    mutual_reachability_dist = compute_MRD(data_pts, core_dist)
    plot_mutual_reachability(data_pts, mutual_reachability_dist)
    
    # Step 3 & 4: Construct MST from Mutual Reachability Graph
    mst = construct_tree_kruskal(mutual_reachability_dist)
    plot_mst(data_pts, mst)
    
    # Step 5: Build Cluster Hierarchy
    condensed_tree, lambda_max = get_density_cluster_plot(mst, min_cluster_size)

    # Step 6: Extract Clusters from Condensed Tree
    cluster_assignments = extract_clusters(condensed_tree, lambda_max)
    plot_clusters(data_pts, cluster_assignments)
    
    # Add noise points
    all_labelled_points = add_noise_points(data_pts, cluster_assignments)
    plot_clusters(data_pts, all_labelled_points)
    
    return all_labelled_points

from sklearn.cluster import HDBSCAN  

if __name__ == "__main__":
    # Generate different types of synthetic datasets and test clustering
    
    # Set parameters
    min_samples = 3
    min_cluster_size = 50
    
    # Performance comparison results storage
    performance_results = []
    
    # Test on different datasets
    # for data_type in ['blobs']:
    # for data_type in ['blobs', 'circles', 'moons', 'anisotropic']:
    for data_type in ['anisotropic']:
        print(f"\nTesting on {data_type} dataset...")
        
        # Generate data
        X, true_labels = generate_custom_data(data_type=data_type, n_samples=500)
        
        # Convert numpy array to list of tuples for our implementation
        data_pts = [tuple(x) for x in X]
        
        # Run custom implementation with performance tracking
        print("Running custom HDBSCAN implementation...")
        custom_result, custom_time, custom_memory = track_performance(
            my_hdbscan, data_pts, min_samples, min_cluster_size
        )
        
        # Convert result dict to array for visualization
        custom_labels = np.array([custom_result.get(i, None) for i in range(len(X))])
        
        # Run scikit-learn implementation with performance tracking
        print("Running scikit-learn HDBSCAN implementation...")
        sk_hdbscan = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
        sklearn_result, sklearn_time, sklearn_memory = track_performance(
            sk_hdbscan.fit_predict, X
        )
        
        # Store performance metrics
        performance_results.append({
            'Dataset': data_type,
            'Custom Time (s)': custom_time,
            'Sklearn Time (s)': sklearn_time,
            'Time Ratio': sklearn_time / custom_time if custom_time > 0 else float('inf'),
            'Custom Memory (MB)': custom_memory,
            'Sklearn Memory (MB)': sklearn_memory,
            'Memory Ratio': sklearn_memory / custom_memory if custom_memory > 0 else float('inf')
        })
        
        # Plot the clustering results
        fig = plot_clusters_comparsion(
            X, custom_labels, sklearn_result, 
            title=f'HDBSCAN Clustering Comparison - {data_type.capitalize()} Dataset'
        )
        
        # Save the figure
        fig.savefig(f'hdbscan_comparison_{data_type}.png')
        plt.close(fig)
        
        print(f"Performance on {data_type}:")
        print(f"  Custom implementation: {custom_time:.4f}s, {custom_memory:.2f}MB")
        print(f"  Sklearn implementation: {sklearn_time:.4f}s, {sklearn_memory:.2f}MB")
    
    # Display performance comparison table
    print("\nPerformance Comparison Summary:")
    df_performance = pd.DataFrame(performance_results)
    print(df_performance.to_string(index=False))
    
    # Save performance results
    df_performance.to_csv('hdbscan_performance_comparison.csv', index=False)
    
    print("\nTesting complete! Results saved as PNG files and CSV.")