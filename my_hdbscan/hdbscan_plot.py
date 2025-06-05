# ======== DATA GENERATION AND TESTING FUNCTIONS ========================================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons # For generating sample da
from sklearn.cluster import HDBSCAN
import os
import time
import psutil
import pandas as pd
import networkx as nx

def plot_core_distances(data_pts, core_dist):
    """
    Visualize core distances for each point in the dataset
    
    Parameters:
    -----------
    data_pts : list of tuples
        List of data points (x, y)
    core_dist : list
        List of core distances for each point
    """
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    x_coords = [pt[0] for pt in data_pts]
    y_coords = [pt[1] for pt in data_pts]
    plt.scatter(x_coords, y_coords, c='blue', s=30)
    
    # Plot core distance circles
    for i, pt in enumerate(data_pts):
        circle = plt.Circle((pt[0], pt[1]), core_dist[i], color='r', 
                           fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_patch(circle)
    
    plt.title("Core Distances", fontsize=14)
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.show()

def plot_mutual_reachability(data_pts, mrd_matrix):
    """
    Visualize mutual reachability graph
    
    Parameters:
    -----------
    data_pts : list of tuples
        List of data points (x, y)
    mrd_matrix : list of lists
        Mutual reachability distance matrix (n x n)
    """
    G = nx.Graph()
    
    # Add nodes
    for i, pt in enumerate(data_pts):
        G.add_node(i, pos=pt)
    
    # Add edges - only include edges for distances less than infinity
    for i in range(len(data_pts)):
        for j in range(i+1, len(data_pts)):  # Only upper triangle
            if mrd_matrix[i][j] < float('inf'):
                G.add_edge(i, j, weight=mrd_matrix[i][j])
    
    # Get positions and edge weights
    pos = nx.get_node_attributes(G, 'pos')
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Normalize weights for better visualization
    if weights:
        max_weight = max(weights)
        normalized_weights = [w/max_weight for w in weights]
    else:
        normalized_weights = []
    
    plt.figure(figsize=(10, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=50)
    
    # Draw edges with color based on weight
    nx.draw_networkx_edges(G, pos, width=2, edge_color=normalized_weights, edge_cmap=plt.cm.viridis)
    
    # Draw labels only if there are few points
    if len(data_pts) <= 20:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Mutual Reachability Graph", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_mst(data_pts, mst):
    """
    Visualize the minimum spanning tree
    
    Parameters:
    -----------
    data_pts : list of tuples
        List of data points (x, y)
    mst : list of tuples
        List of edges in the MST, each in the format (u, v, weight)
    """
    G = nx.Graph()
    
    # Add nodes
    for i, pt in enumerate(data_pts):
        G.add_node(i, pos=pt)
    
    # Add edges from MST
    for u, v, weight in mst:
        G.add_edge(u, v, weight=weight)
    
    # Get positions and edge weights
    pos = nx.get_node_attributes(G, 'pos')
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    plt.figure(figsize=(10, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=80)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, edge_color='orange')
    
    # Draw labels only if there are few points
    if len(data_pts) <= 20:
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Draw edge weights
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title("Minimum Spanning Tree", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_clusters(data_pts, cluster_assignments):
    plt.figure(figsize=(10, 8))
    
    labels = [cluster_assignments.get(i, None) for i in range(len(data_pts))]
    labels = [label if label is not None else -1 for label in labels]
    
    unique_labels = sorted(set(labels))
    label_to_color = {label: idx for idx, label in enumerate(unique_labels)}
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))  # enough distinct colors
    
    for i, label in enumerate(labels):
        if label == -1:
            color = 'black'
            label_name = 'Noise'
        else:
            color = cmap(label_to_color[label])
            label_name = f'Cluster {label}'
        
        # Only label the first occurrence of each cluster
        if label_name not in plt.gca().get_legend_handles_labels()[1]:
            plt.scatter(data_pts[i][0], data_pts[i][1], c=[color], s=50, label=label_name)
        else:
            plt.scatter(data_pts[i][0], data_pts[i][1], c=[color], s=50)

    plt.legend(loc='best')
    plt.title("Final Cluster Assignments", fontsize=14)
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.show()

def generate_custom_data(data_type='blobs', n_samples=300, noise=None, random_state=42):
    """
    Generate synthetic datasets for clustering testing
    
    Parameters:
    -----------
    data_type : str, default='blobs'
        Type of dataset to generate: 'blobs', 'circles', 'moons', or 'anisotropic'
    n_samples : int, default=300
        Number of samples to generate
    noise : float, default=None
        Noise level for the dataset (if applicable)
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X : ndarray of shape (n_samples, 2)
        Generated data points
    y_true : ndarray of shape (n_samples,)
        True cluster labels (ground truth)
    """
    if data_type == 'blobs':
        # Generate isotropic Gaussian blobs
        centers = [[0, 0], [1, 5], [5, 0]]
        cluster_std = [0.5, 0.5, 0.5] if noise is None else [noise, noise, noise]
        X, y_true = make_blobs(n_samples=n_samples, centers=centers, 
                              cluster_std=cluster_std, random_state=random_state)
    
    elif data_type == 'circles':
        # Generate concentric circles
        noise_val = 0.05 if noise is None else noise
        X, y_true = make_circles(n_samples=n_samples, noise=noise_val, 
                                factor=0.5, random_state=random_state)
    
    elif data_type == 'moons':
        # Generate two interleaving half circles
        noise_val = 0.05 if noise is None else noise
        X, y_true = make_moons(n_samples=n_samples, noise=noise_val, 
                              random_state=random_state)
    
    elif data_type == 'anisotropic':
        # Generate anisotropic blobs (different variances)
        centers = [[0, 0], [1, 5], [5, 0]]
        X, y_true = make_blobs(n_samples=n_samples, centers=centers, 
                              random_state=random_state)
        
        # Transform the third blob to be elongated
        transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
        X[y_true == 2] = np.dot(X[y_true == 2], transformation)
        
        if noise is not None:
            # Add some noise
            X += np.random.normal(0, noise, X.shape)
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Choose from 'blobs', 'circles', 'moons', or 'anisotropic'")
    
    return X, y_true


def track_performance(func, *args, **kwargs):
    """
    Track execution time and memory usage of a function
    
    Parameters:
    -----------
    func : callable
        Function to be executed and tracked
    *args, **kwargs : 
        Arguments to pass to the function
        
    Returns:
    --------
    result : any
        Result of the function execution
    execution_time : float
        Execution time in seconds
    memory_used : float
        Peak memory usage in MB
    """
    # Get the process ID
    pid = os.getpid()
    process = psutil.Process(pid)
    
    # Record initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # in MB
    
    # Start timing
    start_time = time.time()
    
    # Execute the function
    result = func(*args, **kwargs)
    
    # End timing
    end_time = time.time()
    
    # Calculate final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # in MB
    
    execution_time = end_time - start_time
    memory_used = final_memory - initial_memory
    
    return result, execution_time, memory_used


def plot_clusters_comparsion(X, labels_custom, labels_sklearn=None, title='Cluster Comparison'):
    """
    Plot clustering results for comparison
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, 2)
        2D data points
    labels_custom : array-like of shape (n_samples,)
        Cluster labels from custom implementation
    labels_sklearn : array-like of shape (n_samples,), optional
        Cluster labels from sklearn implementation
    title : str, default='Cluster Comparison'
        Plot title
    """
    fig, axes = plt.subplots(1, 2 if labels_sklearn is not None else 1, figsize=(12, 5))
    
    # Convert None labels to -1 for consistency with scikit-learn
    labels_custom_plot = np.array([l if l is not None else -1 for l in labels_custom])
    
    # Plot custom implementation results
    if labels_sklearn is not None:
        ax1 = axes[0]
    else:
        ax1 = axes
        
    # Get unique cluster labels
    unique_labels = np.unique(labels_custom_plot)
    
    # Plot each cluster with a different color
    for label in unique_labels:
        mask = labels_custom_plot == label
        if label == -1:
            # Plot noise points in black
            ax1.scatter(X[mask, 0], X[mask, 1], c='black', s=10, alpha=0.5, label='Noise')
        else:
            # Plot cluster points
            ax1.scatter(X[mask, 0], X[mask, 1], s=30, label=f'Cluster {label}')
    
    ax1.set_title(f'Custom HDBSCAN: {len(unique_labels) - (1 if -1 in unique_labels else 0)} clusters')
    ax1.legend(loc='upper right')
    
    # Plot sklearn implementation results (if provided)
    if labels_sklearn is not None:
        ax2 = axes[1]
        unique_labels_sk = np.unique(labels_sklearn)
        
        for label in unique_labels_sk:
            mask = labels_sklearn == label
            if label == -1:
                # Plot noise points in black
                ax2.scatter(X[mask, 0], X[mask, 1], c='black', s=10, alpha=0.5, label='Noise')
            else:
                # Plot cluster points
                ax2.scatter(X[mask, 0], X[mask, 1], s=30, label=f'Cluster {label}')
        
        ax2.set_title(f'Scikit-learn HDBSCAN: {len(unique_labels_sk) - (1 if -1 in unique_labels_sk else 0)} clusters')
        ax2.legend(loc='upper right')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig