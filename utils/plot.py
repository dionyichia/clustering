import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for headless environments
import matplotlib.pyplot as plt
import os 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def create_comprehensive_clustering_plot(
    X, gpu_labels, sklearn_labels, dbscan_labels, true_labels,  # NEW PARAMETER
    gpu_metrics, sklearn_metrics, dbscan_metrics, algo_agreement, gt_stats,  # NEW PARAMETER
    batch_name, feature_names, save_dir
):
    """Create a comprehensive 4-panel plot showing clustering results and metrics"""
    
    # Use PCA for visualization if more than 2 features
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_plot = pca.fit_transform(X)
        x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)"
        y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)"
    else:
        X_plot = X
        x_label = feature_names[0] if len(feature_names) > 0 else "Feature 1"
        y_label = feature_names[1] if len(feature_names) > 1 else "Feature 2"
    
    # Create figure with subplots - UPDATED TO 2x3 LAYOUT
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Clustering Analysis: {batch_name}', fontsize=16, fontweight='bold')
    
    # Color maps
    cmap_discrete = plt.cm.tab20
    
    # Plot 1: Ground Truth
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(X_plot[:, 0], X_plot[:, 1], c=true_labels, 
                          cmap=cmap_discrete, s=30, alpha=0.7)
    ax1.set_title('Ground Truth (EmitterId)', fontweight='bold')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: GPU HDBSCAN
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(X_plot[:, 0], X_plot[:, 1], c=gpu_labels, 
                          cmap=cmap_discrete, s=30, alpha=0.7)
    ax2.set_title('GPU HDBSCAN Results', fontweight='bold')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sklearn HDBSCAN
    ax3 = axes[0, 2]
    scatter3 = ax3.scatter(X_plot[:, 0], X_plot[:, 1], c=sklearn_labels, 
                          cmap=cmap_discrete, s=30, alpha=0.7)
    ax3.set_title('Sklearn HDBSCAN Results', fontweight='bold')
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: DBSCAN - NEW PLOT
    ax4 = axes[1, 0]
    scatter4 = ax4.scatter(X_plot[:, 0], X_plot[:, 1], c=dbscan_labels, 
                          cmap=cmap_discrete, s=30, alpha=0.7)
    ax4.set_title('DBSCAN Results', fontweight='bold')
    ax4.set_xlabel(x_label)
    ax4.set_ylabel(y_label)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Metrics Summary - UPDATED
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    # Create metrics table - UPDATED TO INCLUDE DBSCAN
    metrics_text = f"""
    Dataset Statistics:
    • Samples: {gt_stats['N_Samples']:,}
    • True Clusters: {gt_stats['N_True_Clusters']}
    
    GPU HDBSCAN vs Ground Truth:
    • ARI: {gpu_metrics['ARI']:.3f}
    • Time: {gpu_metrics['time']:.3f}
    • V-Measure: {gpu_metrics['V_Measure']:.3f}
    • Clusters Found: {gpu_metrics['N_Clusters']}
    • Noise Points: {gpu_metrics['N_Noise']}
    • Homogeneity: {gpu_metrics['Homogeneity']}
    
    Sklearn HDBSCAN vs Ground Truth:
    • ARI: {sklearn_metrics['ARI']:.3f}
    • Time: {sklearn_metrics['time']:.3f}
    • V-Measure: {sklearn_metrics['V_Measure']:.3f}
    • Clusters Found: {sklearn_metrics['N_Clusters']}
    • Noise Points: {sklearn_metrics['N_Noise']}
    • Homogeneity Points: {sklearn_metrics['Homogeneity']}
    
    DBSCAN vs Ground Truth:
    • ARI: {dbscan_metrics['ARI']:.3f}
    • Time: {dbscan_metrics['time']:.3f}
    • V-Measure: {dbscan_metrics['V_Measure']:.3f}
    • Clusters Found: {dbscan_metrics['N_Clusters']}
    • Noise Points: {dbscan_metrics['N_Noise']}
    • Homogeneity Points: {dbscan_metrics['Homogeneity']}
    """
    
    ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Plot 6: Algorithm Agreement - NEW PLOT
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    agreement_text = f"""
    Algorithm Agreement:
    
    GPU vs Sklearn HDBSCAN:
    • ARI: {algo_agreement['ARI']:.3f}
    • NMI: {algo_agreement['NMI']:.3f}
    • V-Measure: {algo_agreement['V_Measure']:.3f}
    
    GPU HDBSCAN vs DBSCAN:
    • ARI: {algo_agreement['GPU_vs_DBSCAN_ARI']:.3f}
    
    Sklearn HDBSCAN vs DBSCAN:
    • ARI: {algo_agreement['Sklearn_vs_DBSCAN_ARI']:.3f}
    
    Metric Interpretation:
    • ARI/NMI/V-Measure: 0.0-1.0 (higher = better)
    • > 0.5: Good
    • > 0.7: Strong
    • > 0.9: Excellent
    """
    
    ax6.text(0.05, 0.95, agreement_text, transform=ax6.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, f'{batch_name}_clustering_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comprehensive plot saved to: {save_path}")


def plot_benchmark_results(df):
    """Plot benchmark results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    datasets = df['Dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.25
    
    # Time comparison
    gpu_times = df.groupby('Dataset')['GPU_Time'].mean()
    sklearn_times = df.groupby('Dataset')['Sklearn_Time'].mean()
    
    axes[0, 0].bar(x - width, gpu_times, width, label='GPU HDBSCAN', alpha=0.8)
    axes[0, 0].bar(x, sklearn_times, width, label='Sklearn HDBSCAN', alpha=0.8)
    axes[0, 0].set_xlabel('Dataset')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_title('Execution Time Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(datasets)
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Memory comparison
    gpu_memory = df.groupby('Dataset')['GPU_Memory'].mean()
    sklearn_memory = df.groupby('Dataset')['Sklearn_Memory'].mean()
    
    axes[0, 1].bar(x - width, gpu_memory, width, label='GPU HDBSCAN', alpha=0.8)
    axes[0, 1].bar(x, sklearn_memory, width, label='Sklearn HDBSCAN', alpha=0.8)
    axes[0, 1].set_xlabel('Dataset')
    axes[0, 1].set_ylabel('Memory (MB)')
    axes[0, 1].set_title('Memory Usage Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(datasets)
    axes[0, 1].legend()
    
    # Speedup visualization
    speedup_sklearn = df.groupby('Dataset')['Speedup_vs_Sklearn'].mean()
    
    axes[1, 0].bar(x - width / 2, speedup_sklearn, width, label='vs Sklearn', alpha=0.8)
    axes[1, 0].set_xlabel('Dataset')
    axes[1, 0].set_ylabel('Speedup Factor')
    axes[1, 0].set_title('GPU HDBSCAN Speedup')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(datasets)
    axes[1, 0].legend()
    axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    # Scaling with data size
    for dataset in datasets:
        subset = df[df['Dataset'] == dataset]
        axes[1, 1].plot(subset['Samples'], subset['GPU_Time'], 'o-', label=f'GPU {dataset}', alpha=0.8)
        axes[1, 1].plot(subset['Samples'], subset['Sklearn_Time'], 's--', label=f'Sklearn {dataset}', alpha=0.8)
    
    axes[1, 1].set_xlabel('Number of Samples')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_title('Scaling with Data Size')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('gpu_hdbscan_benchmark.png', dpi=300, bbox_inches='tight')

def plot_clusters_2d(X, labels, feature_names=None, title="GPU HDBSCAN Clustering Results", 
                     save_path=None, use_plotly=False):
    """
    Plot 2D clustering results using the first two features (pulse width and frequency)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points (will use first 2 dimensions)
    labels : array-like, shape (n_samples,)
        Cluster labels from GPU HDBSCAN
    feature_names : list, optional
        Names of features for axis labels
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    use_plotly : bool
        Whether to use Plotly for interactive plots
    """
    
    # Use first two dimensions
    X_2d = X[:, :2]
    
    # Set default feature names
    if feature_names is None:
        feature_names = ['Pulse Width', 'Frequency']
    
    # Get unique clusters (excluding noise if present)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_count = np.sum(labels == -1)
    
    print(f"Found {n_clusters} clusters and {noise_count} noise points")
    
    if use_plotly:
        # Create interactive Plotly plot
        df_plot = pd.DataFrame({
            feature_names[0]: X_2d[:, 0],
            feature_names[1]: X_2d[:, 1],
            'Cluster': labels.astype(str)
        })
        
        # Create color map for clusters
        colors = px.colors.qualitative.Set1
        color_map = {}
        for i, label in enumerate(unique_labels):
            if label == -1:
                color_map[str(label)] = 'black'  # Noise points in black
            else:
                color_map[str(label)] = colors[i % len(colors)]
        
        fig = px.scatter(
            df_plot,
            x=feature_names[0],
            y=feature_names[1],
            color='Cluster',
            title=f"{title}<br>Clusters: {n_clusters}, Noise points: {noise_count}",
            color_discrete_map=color_map,
            width=800,
            height=600
        )
        
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.show()
        
        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
            
    else:
        # Create matplotlib plot
        plt.figure(figsize=(12, 8))
        
        # Create color map
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Plot noise points
                mask = labels == label
                plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                           c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                # Plot cluster points
                mask = labels == label
                plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                           c=[colors[i]], s=60, alpha=0.7, label=f'Cluster {label}')
        
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f"{title}\nClusters: {n_clusters}, Noise points: {noise_count}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def visualize_high_dimensional_clusters(X, labels, feature_names=None, 
                                       method='pca', title="High-Dimensional Clustering",
                                       save_path=None):
    """
    Visualize high-dimensional clustering results using dimensionality reduction
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels
    feature_names : list, optional
        Names of features
    method : str
        Dimensionality reduction method ('pca' or 'tsne')
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X_scaled)
        explained_var = reducer.explained_variance_ratio_
        method_title = f"PCA (Explained variance: {explained_var[0]:.2f}, {explained_var[1]:.2f})"
    # elif method == 'tsne':
    #     reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
    #     X_reduced = reducer.fit_transform(X_scaled)
    #     method_title = "t-SNE"
    else:
        raise ValueError("Method must be 'pca'")
    
    # Plot the results
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_count = np.sum(labels == -1)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            mask = labels == label
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                       c='black', marker='x', s=50, alpha=0.6, label='Noise')
        else:
            mask = labels == label
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                       c=[colors[i]], s=60, alpha=0.7, label=f'Cluster {label}')
    
    plt.xlabel(f'{method_title} Component 1')
    plt.ylabel(f'{method_title} Component 2')
    plt.title(f"{title} - {method_title}\nClusters: {n_clusters}, Noise points: {noise_count}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return X_reduced