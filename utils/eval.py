import numpy as np
from sklearn.metrics import (
    adjusted_rand_score, 
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
import pandas as pd
from typing import List, Dict, Any,Tuple

from utils.plot import *

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

def evaluate_clustering_with_ground_truth(
    X: np.ndarray,
    gpu_labels: np.ndarray,
    sklearn_labels: np.ndarray,
    dbscan_labels: np.ndarray,  # NEW PARAMETER
    true_labels: np.ndarray,
    batch_name: str,
    feature_names: list,
    gpu_time,
    sklearn_time,
    dbscan_time,
    save_dir: str = "benchmark_outputs"

) -> Dict[str, Any]:
    """
    Evaluate clustering results against ground truth and create comprehensive visualization.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature data
    gpu_labels : array-like, shape (n_samples,)
        GPU HDBSCAN cluster labels
    sklearn_labels : array-like, shape (n_samples,)
        Sklearn HDBSCAN cluster labels
    dbscan_labels : array-like, shape (n_samples,)
        DBSCAN cluster labels
    true_labels : array-like, shape (n_samples,)
        Ground truth emitter IDs
    batch_name : str
        Name of the batch for labeling
    feature_names : list
        Names of the features
    save_dir : str
        Directory to save plots
    
    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate key metrics for both algorithms against ground truth
    gpu_metrics = {
        'ARI': adjusted_rand_score(true_labels, gpu_labels),
        'NMI': normalized_mutual_info_score(true_labels, gpu_labels),
        'Homogeneity': homogeneity_score(true_labels, gpu_labels),
        'Completeness': completeness_score(true_labels, gpu_labels),
        'V_Measure': v_measure_score(true_labels, gpu_labels),
        'N_Clusters': len(set(gpu_labels)) - (1 if -1 in gpu_labels else 0),
        'N_Noise': np.sum(gpu_labels == -1),
        'time': gpu_time
    }
    
    sklearn_metrics = {
        'ARI': adjusted_rand_score(true_labels, sklearn_labels),
        'NMI': normalized_mutual_info_score(true_labels, sklearn_labels),
        'Homogeneity': homogeneity_score(true_labels, sklearn_labels),
        'Completeness': completeness_score(true_labels, sklearn_labels),
        'V_Measure': v_measure_score(true_labels, sklearn_labels),
        'N_Clusters': len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0),
        'N_Noise': np.sum(sklearn_labels == -1),
        'time': sklearn_time
    }
    
    # NEW: DBSCAN metrics
    dbscan_metrics = {
        'ARI': adjusted_rand_score(true_labels, dbscan_labels),
        'NMI': normalized_mutual_info_score(true_labels, dbscan_labels),
        'Homogeneity': homogeneity_score(true_labels, dbscan_labels),
        'Completeness': completeness_score(true_labels, dbscan_labels),
        'V_Measure': v_measure_score(true_labels, dbscan_labels),
        'N_Clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
        'N_Noise': np.sum(dbscan_labels == -1),
        'time': dbscan_time
    }
    
    # Calculate agreement between the algorithms - UPDATED
    algo_agreement = {
        'ARI': adjusted_rand_score(gpu_labels, sklearn_labels),
        'NMI': normalized_mutual_info_score(gpu_labels, sklearn_labels),
        'V_Measure': v_measure_score(gpu_labels, sklearn_labels),
        # Additional pairwise comparisons
        'GPU_vs_DBSCAN_ARI': adjusted_rand_score(gpu_labels, dbscan_labels),
        'Sklearn_vs_DBSCAN_ARI': adjusted_rand_score(sklearn_labels, dbscan_labels)
    }
    
    # Ground truth statistics
    gt_stats = {
        'N_True_Clusters': len(set(true_labels)),
        'N_Samples': len(true_labels)
    }
    
    # Create comprehensive visualization - UPDATED TO INCLUDE DBSCAN
    create_comprehensive_clustering_plot(
        X, gpu_labels, sklearn_labels, dbscan_labels, true_labels,
        gpu_metrics, sklearn_metrics, dbscan_metrics, algo_agreement, gt_stats,
        batch_name, feature_names, save_dir
    )
    
    return {
        'gpu_metrics': gpu_metrics,
        'sklearn_metrics': sklearn_metrics,
        'dbscan_metrics': dbscan_metrics,  # NEW
        'algorithm_agreement': algo_agreement,
        'ground_truth_stats': gt_stats,
        'batch_name': batch_name
    }


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
    
    # Print results
    print("\n" + "="*50)
    print("CLUSTERING EVALUATION METRICS")
    print("="*50)
    print(f"Number of clusters: {metrics['n_clusters']}")
    print(f"Number of noise points: {metrics['n_noise_points']}")
    print(f"Noise ratio: {metrics['noise_ratio']:.3f}")
    print(f"Total points: {metrics['total_points']}")
    
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