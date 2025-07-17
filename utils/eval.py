from sklearn.metrics import (
    adjusted_rand_score, 
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from typing import Dict, Any

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
    print("Creating 2D visualization of first 2 features...")
    plot_clusters_2d(X, labels, feature_names[:2], 
                     title="GPU HDBSCAN: Pulse Width vs Frequency",
                     save_path=f"{save_prefix}_2d.png")
    
    # 2. High-dimensional visualization if needed
    if X.shape[1] > 2:
        print("Creating PCA visualization for all features...")
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
    summary_df = per_cluster_statistics_summary(X, labels, feature_names)
    results['summary'] = summary_df
    
    # Save summary to CSV
    summary_df.to_csv(f"{save_prefix}_summary.csv", index=False)
    
    return results

def evaluate_cluster_purity(predicted_labels: np.ndarray, true_labels: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate cluster purity by counting pure and impure clusters.
    
    Parameters:
    -----------
    predicted_labels : array-like, shape (n_samples,)
        Cluster labels from the algorithm being evaluated
    true_labels : array-like, shape (n_samples,)
        Ground truth emitter IDs
        
    Returns:
    --------
    dict : Dictionary containing purity metrics
        - n_pure_clusters: Number of clusters with points from single emitter
        - n_impure_clusters: Number of clusters with points from multiple emitters
        - n_noise_clusters: Number of clusters made entirely of noise points
        - total_clusters: Total number of clusters (excluding noise label -1)
        - purity_ratio: Ratio of pure clusters to total clusters
        - cluster_details: List of details for each cluster
    """
    
    # Get unique cluster labels (excluding noise label -1)
    unique_clusters = set(predicted_labels)
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
    
    pure_clusters = 0
    impure_clusters = 0
    noise_clusters = 0
    # cluster_details = []
    
    for cluster_id in unique_clusters:
        # Get indices of points in this cluster
        cluster_mask = predicted_labels == cluster_id
        cluster_true_labels = true_labels[cluster_mask]
        
        # Get unique emitters in this cluster
        unique_emitters = set(cluster_true_labels)
        
        # Check if cluster contains only noise points (assuming -1 represents noise in ground truth too)
        if len(unique_emitters) == 1 and -1 in unique_emitters:
            noise_clusters += 1
            # cluster_type = "noise"
        elif len(unique_emitters) == 1:
            pure_clusters += 1
            # cluster_type = "pure"
        else:
            impure_clusters += 1
            # cluster_type = "impure"
        
        # Store cluster details
        # cluster_details.append({
        #     'cluster_id': cluster_id,
        #     'cluster_type': cluster_type,
        #     'n_points': np.sum(cluster_mask),
        #     'n_unique_emitters': len(unique_emitters),
        #     'emitters': list(unique_emitters)
        # })
    
    total_clusters = len(unique_clusters)
    purity_ratio = pure_clusters / total_clusters if total_clusters > 0 else 0
    
    return {
        'n_pure_clusters': pure_clusters,
        'n_impure_clusters': impure_clusters,
        'n_noise_clusters': noise_clusters,
        'total_clusters': total_clusters,
        'purity_ratio': purity_ratio,
    }


def evaluate_cluster_quality_detailed(predicted_labels: np.ndarray, true_labels: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate cluster quality with detailed metrics including perfectly separated,
    broken-up, missing, incorrectly merged, and false clusters.
    
    Parameters:
    -----------
    predicted_labels : array-like, shape (n_samples,)
        Cluster labels from the algorithm being evaluated
    true_labels : array-like, shape (n_samples,)
        Ground truth cluster labels
        
    Returns:
    --------
    dict : Dictionary containing detailed cluster quality metrics
    """
    
    # Convert to numpy arrays
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    
    # Get unique labels (excluding noise label -1)
    unique_predicted = set(predicted_labels)
    unique_true = set(true_labels)
    
    # Remove noise labels for analysis
    if -1 in unique_predicted:
        unique_predicted.remove(-1)
    if -1 in unique_true:
        unique_true.remove(-1)
    
    # Initialize metrics
    perfectly_separated = []
    broken_up_clusters = []
    missing_clusters = []
    incorrectly_merged = []
    false_clusters = []
    
    # Track which predicted clusters have been accounted for
    accounted_predicted_clusters = set()
    
    # First pass: Analyze predicted clusters for merging and false clusters
    # Check all predicted cluster ids
    for pred_cluster_id in unique_predicted:
        # make a boolean array to track cluster membership
        # True if point label == cluster id
        pred_mask = predicted_labels == pred_cluster_id
        # find corresponding ground truth points to see if cluster is incorrectly merged/noise
        pred_true_labels = true_labels[pred_mask]
        
        # Get unique ground truth labels in this predicted cluster
        unique_true_in_pred = set(pred_true_labels)

        # Remove noise if present
        if -1 in unique_true_in_pred:
            unique_true_in_pred.remove(-1)
        
        # Check if cluster contains only noise points
        if len(unique_true_in_pred) == 0:
            false_clusters.append({
                'predicted_cluster_id': pred_cluster_id,
                'n_points': np.sum(pred_mask),
                'reason': 'contains_only_noise'
            })
            accounted_predicted_clusters.add(pred_cluster_id)

        # Check if cluster contains points from multiple ground truth clusters
        elif len(unique_true_in_pred) > 1:
            incorrectly_merged.append({
                'predicted_cluster_id': pred_cluster_id,
                'n_points': np.sum(pred_mask),
                'ground_truth_labels': list(unique_true_in_pred),
                'n_ground_truth_clusters': len(unique_true_in_pred)
            })
            accounted_predicted_clusters.add(pred_cluster_id)
    
    # Second pass: Analyze each ground truth cluster to determine its fate
    for true_cluster_id in unique_true:
        # Same as above but mask is made using ground truth cluster id
        true_mask = true_labels == true_cluster_id
        # Find how points of ground truth cluster id are split in predicted labels
        true_pred_labels = predicted_labels[true_mask]
        
        # Get unique predicted cluster labels for this ground truth cluster
        # e.g ground truth cluster 1 could be split amongst predicted cluster id 1,2,3
        # unique_pred_in_true = [1,2,3]
        unique_pred_in_true = set(true_pred_labels)

        # Remove noise if present
        if -1 in unique_pred_in_true:
            unique_pred_in_true.remove(-1)
        
        total_true_points = np.sum(true_mask)
        points_in_noise = np.sum(true_pred_labels == -1)
        
        # Check if ground truth cluster is missing (no predicted cluster contains its points)
        if len(unique_pred_in_true) == 0:
            missing_clusters.append({
                'ground_truth_cluster_id': true_cluster_id,
                'n_points': total_true_points,
                'all_points_in_noise': True
            })
        # Check if ground truth cluster is broken up (points spread across multiple predicted clusters)
        elif len(unique_pred_in_true) > 1:
            # For a cluster to be "broken up", ALL predicted clusters containing its points
            # must contain ONLY points from this ground truth cluster
            is_truly_broken_up = True
            distribution = {}
            
            # for each predicted cluster id containing points from ground truth cluster 
            for pred_id in unique_pred_in_true:
                # use existing boolean mask with boolean mask of this predicted cluster id
                # find number of points from ground truth cluster in this predicted cluster id
                overlap_mask = true_mask & (predicted_labels == pred_id)
                # distribution is FOR DEBUGGING, shows how points are split across predicted cluster ids
                distribution[pred_id] = np.sum(overlap_mask)
                
                # Check if this predicted cluster contains points from other ground truth clusters
                # find all points clustered in this cluster
                pred_mask = predicted_labels == pred_id
                pred_true_labels_for_this_pred = true_labels[pred_mask]
                # find the number of cluster ids in this cluster
                unique_true_in_this_pred = set(pred_true_labels_for_this_pred)
                if -1 in unique_true_in_this_pred:
                    unique_true_in_this_pred.remove(-1)
                
                # If this predicted cluster contains points from multiple ground truth clusters,
                # then this ground truth cluster is not "broken up" - it's part of merged clusters
                if len(unique_true_in_this_pred) > 1:
                    is_truly_broken_up = False
                    break
            
            if is_truly_broken_up:
                broken_up_clusters.append({
                    'ground_truth_cluster_id': true_cluster_id,
                    'n_points': total_true_points,
                    'predicted_clusters': list(unique_pred_in_true),
                    'n_predicted_clusters': len(unique_pred_in_true),
                    'distribution': distribution,
                    'points_in_noise': points_in_noise
                })
                # Mark all predicted clusters involved in this broken up cluster as accounted for
                accounted_predicted_clusters.update(unique_pred_in_true)
            else:
                # This is the edge case: ground truth cluster is split across multiple predicted clusters,
                # but some predicted clusters also contain points from other ground truth clusters.
                # We need to account for the predicted clusters that contain ONLY points from this ground truth cluster
                for pred_id in unique_pred_in_true:
                    if pred_id not in accounted_predicted_clusters:
                        # Check if this predicted cluster contains points from only this ground truth cluster
                        pred_mask = predicted_labels == pred_id
                        pred_true_labels_for_this_pred = true_labels[pred_mask]
                        unique_true_in_this_pred = set(pred_true_labels_for_this_pred)
                        if -1 in unique_true_in_this_pred:
                            unique_true_in_this_pred.remove(-1)
                        
                        if len(unique_true_in_this_pred) == 1 and list(unique_true_in_this_pred)[0] == true_cluster_id:
                            # This predicted cluster contains only points from the current ground truth cluster
                            # Add it as a broken up cluster (individual fragment)
                            overlap_mask = true_mask & pred_mask
                            broken_up_clusters.append({
                                'ground_truth_cluster_id': true_cluster_id,
                                'n_points': np.sum(overlap_mask),
                                'predicted_clusters': [pred_id],
                                'n_predicted_clusters': 1,
                                'distribution': {pred_id: np.sum(overlap_mask)},
                                'points_in_noise': 0,
                                'note': 'fragment_of_complex_split'
                            })
                            accounted_predicted_clusters.add(pred_id)
                            
        # Check if ground truth cluster is perfectly separated (exactly one predicted cluster contains all its points)
        elif len(unique_pred_in_true) == 1:
            pred_cluster_id = list(unique_pred_in_true)[0]
            
            # Check if this predicted cluster contains points from only one ground truth cluster
            pred_mask = predicted_labels == pred_cluster_id
            pred_true_labels = true_labels[pred_mask]
            unique_true_in_pred = set(pred_true_labels)
            if -1 in unique_true_in_pred:
                unique_true_in_pred.remove(-1)
            
            # Perfect separation: predicted cluster contains points from exactly one ground truth cluster
            # The ground truth cluster's non-noise points are all in this predicted cluster
            true_points_in_pred = np.sum(true_mask & pred_mask)
            non_noise_true_points = total_true_points - points_in_noise
            
            if (len(unique_true_in_pred) == 1 and 
                list(unique_true_in_pred)[0] == true_cluster_id and
                true_points_in_pred == non_noise_true_points):
                # Perfect separation - all non-noise points from ground truth cluster are in this predicted cluster
                coverage = true_points_in_pred / total_true_points if total_true_points > 0 else 0
                perfectly_separated.append({
                    'predicted_cluster_id': pred_cluster_id,
                    'ground_truth_cluster_id': true_cluster_id,
                    'n_points': true_points_in_pred,
                    'total_ground_truth_points': total_true_points,
                    'points_in_noise': points_in_noise,
                    'coverage': coverage
                })
                accounted_predicted_clusters.add(pred_cluster_id)
            else:
                # This ground truth cluster's points are all in one predicted cluster,
                # but that predicted cluster also contains points from other ground truth clusters
                # This means the predicted cluster is "incorrectly merged" (handled in first pass)
                # and from the ground truth perspective, this cluster is "absorbed" into a merged cluster
                pass
    
    # Calculate summary statistics
    n_predicted_clusters = len(unique_predicted)
    n_true_clusters = len(unique_true)
    
    # Verify all predicted clusters are accounted for
    unaccounted_clusters = unique_predicted - accounted_predicted_clusters
    if unaccounted_clusters:
        print(f"Warning: {len(unaccounted_clusters)} predicted clusters not accounted for: {unaccounted_clusters}")
        
        # Debug the unaccounted clusters
        print("\nDebugging unaccounted clusters:")
        for pred_cluster_id in unaccounted_clusters:
            pred_mask = predicted_labels == pred_cluster_id
            pred_true_labels = true_labels[pred_mask]
            
            # Get unique ground truth labels in this predicted cluster
            unique_true_in_pred = set(pred_true_labels)
            if -1 in unique_true_in_pred:
                unique_true_in_pred.remove(-1)
            
            n_points = np.sum(pred_mask)
            n_noise_points = np.sum(pred_true_labels == -1)
            
            print(f"  Predicted cluster {pred_cluster_id}:")
            print(f"    - Total points: {n_points}")
            print(f"    - Noise points: {n_noise_points}")
            print(f"    - Ground truth labels present: {unique_true_in_pred}")
            print(f"    - Number of ground truth clusters: {len(unique_true_in_pred)}")
            
            if len(unique_true_in_pred) == 0:
                print(f"    - WHY UNACCOUNTED: Should have been classified as 'false cluster' but wasn't")
            elif len(unique_true_in_pred) > 1:
                print(f"    - WHY UNACCOUNTED: Should have been classified as 'incorrectly merged' but wasn't")
            elif len(unique_true_in_pred) == 1:
                true_cluster_id = list(unique_true_in_pred)[0]
                print(f"    - WHY UNACCOUNTED: Contains only ground truth cluster {true_cluster_id}")
                
                # Check what happened to this ground truth cluster
                true_mask = true_labels == true_cluster_id
                true_pred_labels = predicted_labels[true_mask]
                unique_pred_in_true = set(true_pred_labels)
                if -1 in unique_pred_in_true:
                    unique_pred_in_true.remove(-1)
                
                print(f"    - Ground truth cluster {true_cluster_id} is spread across predicted clusters: {unique_pred_in_true}")
                
                if len(unique_pred_in_true) == 1:
                    print(f"    - This should have been 'perfectly separated' - checking why it wasn't...")
                    total_true_points = np.sum(true_mask)
                    true_points_in_pred = np.sum(true_mask & pred_mask)
                    print(f"    - Total points in ground truth cluster: {total_true_points}")
                    print(f"    - Points from ground truth cluster in this predicted cluster: {true_points_in_pred}")
                    
                    if true_points_in_pred == total_true_points:
                        print(f"    - ERROR: This should have been perfectly separated!")
                    else:
                        print(f"    - Ground truth cluster has some points elsewhere (noise?)")
                        
                elif len(unique_pred_in_true) > 1:
                    print(f"    - Ground truth cluster is split - checking if it was processed as broken up...")
                    # This cluster should have been caught in the edge case handling
                    print(f"    - This should have been caught in the edge case handling")
    
    return {
        'summary': {
            'n_predicted_clusters': n_predicted_clusters,
            'n_true_clusters': n_true_clusters,
            'n_perfectly_separated': len(perfectly_separated),
            'n_broken_up': len(broken_up_clusters),
            'n_missing': len(missing_clusters),
            'n_incorrectly_merged': len(incorrectly_merged),
            'n_false_clusters': len(false_clusters),
            'n_accounted_predicted_clusters': len(accounted_predicted_clusters),
            'n_unaccounted_predicted_clusters': len(unaccounted_clusters)
        },
        'detailed': {
            'perfectly_separated': perfectly_separated,
            'broken_up_clusters': broken_up_clusters,
            'missing_clusters': missing_clusters,
            'incorrectly_merged': incorrectly_merged,
            'false_clusters': false_clusters,
        }
    }

def evaluate_clustering_with_ground_truth(
    X: np.ndarray,
    gpu_labels: np.ndarray,
    sklearn_labels: np.ndarray,
    dbscan_labels: np.ndarray,  # NEW PARAMETER
    true_labels: np.ndarray,
    batch_name: str,
    start_time,
    end_time, 
    feature_names: list,
    gpu_time,
    sklearn_time,
    dbscan_time,
    gpu_mem,
    sklearn_mem,
    dbscan_mem,
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
    if gpu_labels is not None:
        gpu_metrics = {
            'ARI': adjusted_rand_score(true_labels, gpu_labels),
            'NMI': normalized_mutual_info_score(true_labels, gpu_labels),
            'Homogeneity': homogeneity_score(true_labels, gpu_labels),
            'Completeness': completeness_score(true_labels, gpu_labels),
            'V_Measure': v_measure_score(true_labels, gpu_labels),
            'N_Clusters': len(set(gpu_labels)) - (1 if -1 in gpu_labels else 0),
            'N_Noise': np.sum(gpu_labels == -1),
            # 'purity': evaluate_cluster_purity(gpu_labels, true_labels),
            'detailed_quality': evaluate_cluster_quality_detailed(gpu_labels, true_labels),
            'time': gpu_time,
            'mem': gpu_mem
        }
    else:
        gpu_metrics = {
            'ARI': None,
            'NMI': None,
            'Homogeneity': None,
            'Completeness': None,
            'V_Measure': None,
            'N_Clusters': None,
            'N_Noise': None,
            # 'purity': evaluate_cluster_purity(gpu_labels, true_labels),
            'detailed_quality': None,
            'time': gpu_time,
            'mem': gpu_mem
        }

    if sklearn_labels is not None:
        sklearn_metrics = {
            'ARI': adjusted_rand_score(true_labels, sklearn_labels),
            'NMI': normalized_mutual_info_score(true_labels, sklearn_labels),
            'Homogeneity': homogeneity_score(true_labels, sklearn_labels),
            'Completeness': completeness_score(true_labels, sklearn_labels),
            'V_Measure': v_measure_score(true_labels, sklearn_labels),
            'N_Clusters': len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0),
            'N_Noise': np.sum(sklearn_labels == -1),
            # 'purity': evaluate_cluster_purity(sklearn_labels, true_labels),
            'detailed_quality': evaluate_cluster_quality_detailed(sklearn_labels, true_labels),
            'time': sklearn_time,
            'mem': sklearn_mem
        }
    else:
        sklearn_metrics = {
            'ARI': None,
            'NMI': None,
            'Homogeneity': None,
            'Completeness': None,
            'V_Measure': None,
            'N_Clusters': None,
            'N_Noise': None,
            # 'purity': evaluate_cluster_purity(gpu_labels, true_labels),
            'detailed_quality': None,
            'time': sklearn_time,
            'mem': sklearn_mem
        }

    
    if dbscan_labels is not None:
        dbscan_metrics = {
            'ARI': adjusted_rand_score(true_labels, dbscan_labels),
            'NMI': normalized_mutual_info_score(true_labels, dbscan_labels),
            'Homogeneity': homogeneity_score(true_labels, dbscan_labels),
            'Completeness': completeness_score(true_labels, dbscan_labels),
            'V_Measure': v_measure_score(true_labels, dbscan_labels),
            'N_Clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'N_Noise': np.sum(dbscan_labels == -1),
            # 'purity': evaluate_cluster_purity(dbscan_labels, true_labels),
            'detailed_quality': evaluate_cluster_quality_detailed(dbscan_labels, true_labels),
            'time': dbscan_time,
            'mem':dbscan_mem
        }
    else:
        dbscan_metrics = {
            'ARI': None,
            'NMI': None,
            'Homogeneity': None,
            'Completeness': None,
            'V_Measure': None,
            'N_Clusters': None,
            'N_Noise': None,
            # 'purity': evaluate_cluster_purity(gpu_labels, true_labels),
            'detailed_quality': None,
            'time': dbscan_time,
            'mem': dbscan_mem
        }
    
    algo_agreement = {
        'ARI': adjusted_rand_score(gpu_labels, sklearn_labels) if gpu_labels is not None and sklearn_labels is not None else None,
        'NMI': normalized_mutual_info_score(gpu_labels, sklearn_labels) if gpu_labels is not None and sklearn_labels is not None else None,
        'V_Measure': v_measure_score(gpu_labels, sklearn_labels) if gpu_labels is not None and sklearn_labels is not None else None,
        # Additional pairwise comparisons
        'GPU_vs_DBSCAN_ARI': adjusted_rand_score(gpu_labels, dbscan_labels) if gpu_labels is not None and dbscan_labels is not None else None,
        'Sklearn_vs_DBSCAN_ARI': adjusted_rand_score(sklearn_labels, dbscan_labels) if sklearn_labels is not None and dbscan_labels is not None else None
    }
    
    # Ground truth statistics
    gt_stats = {
        'N_True_Clusters': len(set(true_labels)),
        'N_Samples': len(true_labels),
        'start_time': start_time,
        'end_time': end_time
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

def per_cluster_statistics_summary(X, labels, feature_names=None):
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

def print_speed_benchmark_summary(results_df):
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