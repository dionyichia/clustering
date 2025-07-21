import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for headless environments
import matplotlib.pyplot as plt
import os 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def create_comprehensive_clustering_plot(
    X, gpu_labels, sklearn_labels, dbscan_labels, true_labels,
    gpu_metrics, sklearn_metrics, dbscan_metrics, algo_agreement, gt_stats,
    batch_name, feature_names, save_dir
):
    """Create a comprehensive 4-panel plot showing clustering results and detailed metrics"""
    
    if X is None or true_labels is None:
        print(f"Skipping plot for {batch_name}: Missing critical data (X or true_labels)")
        return

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

    # Create figure with 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(26, 18))
    fig.suptitle(f'Clustering Analysis: {batch_name}', fontsize=20, fontweight='bold')

    # Batch Statistics Header
    # fig.text(0.5, 0.34, 'Batch Statistics', ha='center', fontsize=16, fontweight='bold')
    # fig.text(0.23, 0.30, 'Dataset Overview', ha='center', fontsize=13, style='italic')
    # fig.text(0.53, 0.30, 'Predicted Emitter Analysis', ha='center', fontsize=13, style='italic')
    # fig.text(0.83, 0.30, 'Predicted Cluster Analysis', ha='center', fontsize=13, style='italic')

    # Color maps
    cmap_discrete = plt.cm.tab20

    # First row: scatter plots
    _plot_clustering_results(
        axes[:2, :], X_plot, x_label, y_label, cmap_discrete,
        true_labels, gpu_labels, sklearn_labels, dbscan_labels
    )

    # # Row 3, Col 1: Batch stats
    # axes[1, 0].axis('off')
    # if gt_stats:
    #     start = gt_stats['start_time'] / 1e9
    #     end = gt_stats['end_time'] / 1e9
    #     interval = end - start

    #     dataset_info = (
    #         f"• {gt_stats['N_Samples']:,} samples\n"
    #         f"• {gt_stats['N_True_Clusters']} true clusters\n"
    #         f"• {interval:.3f}s interval"
    #     )

    #     axes[2, 0].text(0.5, 0.5, dataset_info, transform=axes[2, 0].transAxes,
    #                    ha='center', va='center', fontsize=12, weight='bold')

    # Row 2, Col 1: Emitter metrics
    axes[1, 1].axis('off')
    _plot_emitter_perspective_metrics(axes[1, 1], gpu_metrics, sklearn_metrics, dbscan_metrics, gt_stats)

    # Row 2, Col 2: Cluster metrics
    axes[1, 2].axis('off')
    _plot_cluster_perspective_metrics(axes[1, 2], gpu_metrics, sklearn_metrics, dbscan_metrics)

    # Save
    save_path = os.path.join(save_dir, f'{batch_name}_clustering_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comprehensive plot saved to: {save_path}")


def _plot_clustering_results(axes, X_plot, x_label, y_label, cmap_discrete,
                           true_labels, gpu_labels, sklearn_labels, dbscan_labels):
    """Plot the 4 clustering result scatter plots"""
    
    # Plot 1: Ground Truth
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(X_plot[:, 0], X_plot[:, 1], c=true_labels, 
                          cmap=cmap_discrete, s=30, alpha=0.7)
    ax1.set_title('Ground Truth (EmitterId)', fontweight='bold', fontsize=14)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: GPU HDBSCAN
    ax2 = axes[0, 1]
    if gpu_labels is not None:
        scatter2 = ax2.scatter(X_plot[:, 0], X_plot[:, 1], c=gpu_labels, 
                              cmap=cmap_discrete, s=30, alpha=0.7)
        ax2.set_title('GPU HDBSCAN Results', fontweight='bold', fontsize=14)
    else:
        ax2.text(0.5, 0.5, 'GPU HDBSCAN\nSkipped/Timeout', 
                transform=ax2.transAxes, ha='center', va='center', 
                fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        ax2.set_title('GPU HDBSCAN Results (Skipped)', fontweight='bold', fontsize=14)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sklearn HDBSCAN
    ax3 = axes[0, 2]
    if sklearn_labels is not None:
        scatter3 = ax3.scatter(X_plot[:, 0], X_plot[:, 1], c=sklearn_labels, 
                              cmap=cmap_discrete, s=30, alpha=0.7)
        ax3.set_title('Sklearn HDBSCAN Results', fontweight='bold', fontsize=14)
    else:
        ax3.text(0.5, 0.5, 'Sklearn HDBSCAN\nSkipped/Timeout', 
                transform=ax3.transAxes, ha='center', va='center', 
                fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        ax3.set_title('Sklearn HDBSCAN Results (Skipped)', fontweight='bold', fontsize=14)
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: DBSCAN
    ax4 = axes[1, 0]
    if dbscan_labels is not None:
        scatter4 = ax4.scatter(X_plot[:, 0], X_plot[:, 1], c=dbscan_labels, 
                              cmap=cmap_discrete, s=30, alpha=0.7)
        ax4.set_title('DBSCAN Results', fontweight='bold', fontsize=14)
    else:
        ax4.text(0.5, 0.5, 'DBSCAN\nSkipped/Timeout', 
                transform=ax4.transAxes, ha='center', va='center', 
                fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        ax4.set_title('DBSCAN Results (Skipped)', fontweight='bold', fontsize=14)
    ax4.set_xlabel(x_label)
    ax4.set_ylabel(y_label)
    ax4.grid(True, alpha=0.3)


def _plot_emitter_perspective_metrics(ax, gpu_metrics, sklearn_metrics, dbscan_metrics, gt_stats):
    """Display emitter perspective metrics as formatted text"""

    # ax.axis('off')
    # ax.set_title('Predicted Emitter Analysis', fontweight='bold', fontsize=14, pad=20)
    
    # Helper function to safely get nested values
    def safe_get_nested(d, keys, default="N/A"):
        if d is None:
            return default
        try:
            current = d
            for key in keys:
                current = current[key]
            return str(current) if current is not None else default
        except (KeyError, TypeError):
            return default
    
    # Helper function to format values
    def safe_format(value, format_str="{:.3f}", default="N/A"):
        if value is None:
            return default
        try:
            return format_str.format(value)
        except (ValueError, TypeError):
            return str(value) if value is not None else default
        
    if gt_stats:
        start = gt_stats['start_time'] / 1e9
        end = gt_stats['end_time'] / 1e9
        interval = end - start
        
    # Create metrics text
    metrics_text = f"""Emitter Analysis

Dataset Statistics:
• Batch Interval ({interval:.3f}s): {start:.3f}s - {end:.3f}s
• Samples: {gt_stats['N_Samples']:,}
• True Emitters: {gt_stats['N_True_Clusters']}

GPU HDBSCAN: 
• Perfect: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_perfectly_separated_emitters'])}
• Broken Up: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_broken_up_emitters'])}
• Missing: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_missing_emitters'])}
• Merged: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_incorrectly_merged_emitters'])}

Sklearn HDBSCAN: 
• Perfect: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_perfectly_separated_emitters'])}
• Broken Up: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_broken_up_emitters'])}
• Missing: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_missing_emitters'])}
• Merged: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_incorrectly_merged_emitters'])}

DBSCAN:
• Perfect: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_perfectly_separated_emitters'])}
• Broken Up: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_broken_up_emitters'])}
• Missing: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_missing_emitters'])}
• Merged: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_incorrectly_merged_emitters'])}"""
    
    # Display the metrics text
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=10, fontfamily='monospace')
    
    # # Add explanation at the bottom
    # explanation = (
    #     "Perfect: Emitters with all points in exactly one cluster\n"
    #     "Broken Up: Emitters split across multiple pure clusters\n"
    #     "Missing: Emitters with all points classified as noise\n"
    #     "Merged: Emitters mixed with points from other emitters"
    # )
    
    # ax.text(0.5, 0.08, explanation, transform=ax.transAxes, ha='left', va='bottom',
    #         fontsize=8, style='italic')


def _plot_cluster_perspective_metrics(ax, gpu_metrics, sklearn_metrics, dbscan_metrics):
    """Display cluster perspective metrics as formatted text"""
    
    # ax.axis('off')
    # ax.set_title('Predicted Cluster Analysis', fontweight='bold', fontsize=14, pad=20)
    
    # Helper function to safely get nested values
    def safe_get_nested(d, keys, default="N/A"):
        if d is None:
            return default
        try:
            current = d
            for key in keys:
                current = current[key]
            return str(current) if current is not None else default
        except (KeyError, TypeError):
            return default
    
    # Helper function to format values
    def safe_format(value, format_str="{:.3f}", default="N/A"):
        if value is None:
            return default
        try:
            return format_str.format(value)
        except (ValueError, TypeError):
            return str(value) if value is not None else default
    
    # Create metrics text
    metrics_text = f"""Cluster Analysis

GPU HDBSCAN
• Total Clusters: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_predicted_clusters'])}
• Pure: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_pure_clusters'])}
• Impure: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_impure_clusters'])}
• Broken Up: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_broken_up_clusters'])}
• False: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_false_clusters'])}

• Time: {safe_format(gpu_metrics['time'] if gpu_metrics else None)}
• Memory: {safe_format(gpu_metrics['mem'] if gpu_metrics else None)}
• Homogeneity: {safe_format(gpu_metrics['Homogeneity'] if gpu_metrics else None)}
• V-Measure: {safe_format(gpu_metrics['V_Measure'] if gpu_metrics else None)}

Sklearn HDBSCAN:
• Total Clusters: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_predicted_clusters'])}
• Pure: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_pure_clusters'])}
• Impure: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_impure_clusters'])}
• Broken Up: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_broken_up_clusters'])}
• False: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_false_clusters'])}

• Time: {safe_format(sklearn_metrics['time'] if sklearn_metrics else None)}
• Memory: {safe_format(sklearn_metrics['mem'] if sklearn_metrics else None)}
• Homogeneity: {safe_format(sklearn_metrics['Homogeneity'] if sklearn_metrics else None)}
• V-Measure: {safe_format(sklearn_metrics['V_Measure'] if sklearn_metrics else None)}

DBSCAN:
• Total Clusters: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_predicted_clusters'])}
• Pure: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_pure_clusters'])}
• Impure: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_impure_clusters'])}
• Broken Up: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_broken_up_clusters'])}
• False: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_false_clusters'])}

• Time: {safe_format(dbscan_metrics['time'] if dbscan_metrics else None)}
• Memory: {safe_format(dbscan_metrics['mem'] if dbscan_metrics else None)}
• Homogeneity: {safe_format(dbscan_metrics['Homogeneity'] if dbscan_metrics else None)}
• V-Measure: {safe_format(dbscan_metrics['V_Measure'] if dbscan_metrics else None)}"""
    
    # Display the metrics text
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=10, fontfamily='monospace')
    
    # # Add explanation at the bottom
    # explanation = (
    #     "Pure: Clusters with points from exactly one emitter\n"
    #     "Broken Up: Pure clusters that are siblings (same emitter split)\n"
    #     "Impure: Clusters with points from multiple emitters\n"
    #     "False: Clusters containing only noise points"
    # )
    
    # ax.text(0.05, 0.01, explanation, transform=ax.transAxes, ha='left', va='bottom',
    #         fontsize=8, style='italic')
    

# def create_comprehensive_clustering_plot(
#     X, gpu_labels, sklearn_labels, dbscan_labels, true_labels,
#     gpu_metrics, sklearn_metrics, dbscan_metrics, algo_agreement, gt_stats,
#     batch_name, feature_names, save_dir
# ):
#     """Create a comprehensive 4-panel plot showing clustering results and metrics"""
    
#     # Early return if critical data is missing
#     if X is None or true_labels is None:
#         print(f"Skipping plot for {batch_name}: Missing critical data (X or true_labels)")
#         return
    
#     # Use PCA for visualization if more than 2 features
#     if X.shape[1] > 2:
#         pca = PCA(n_components=2)
#         X_plot = pca.fit_transform(X)
#         x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)"
#         y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)"
#     else:
#         X_plot = X
#         x_label = feature_names[0] if len(feature_names) > 0 else "Feature 1"
#         y_label = feature_names[1] if len(feature_names) > 1 else "Feature 2"
    
#     # Create figure with subplots - UPDATED TO 2x3 LAYOUT
#     fig, axes = plt.subplots(2, 3, figsize=(20, 12))
#     fig.suptitle(f'Clustering Analysis: {batch_name}', fontsize=16, fontweight='bold')
    
#     # Color maps
#     cmap_discrete = plt.cm.tab20
    
#     # Plot 1: Ground Truth
#     ax1 = axes[0, 0]
#     scatter1 = ax1.scatter(X_plot[:, 0], X_plot[:, 1], c=true_labels, 
#                           cmap=cmap_discrete, s=30, alpha=0.7)
#     ax1.set_title('Ground Truth (EmitterId)', fontweight='bold')
#     ax1.set_xlabel(x_label)
#     ax1.set_ylabel(y_label)
#     ax1.grid(True, alpha=0.3)
    
#     # Plot 2: GPU HDBSCAN
#     ax2 = axes[0, 1]
#     if gpu_labels is not None:
#         scatter2 = ax2.scatter(X_plot[:, 0], X_plot[:, 1], c=gpu_labels, 
#                               cmap=cmap_discrete, s=30, alpha=0.7)
#         ax2.set_title('GPU HDBSCAN Results', fontweight='bold')
#     else:
#         ax2.text(0.5, 0.5, 'GPU HDBSCAN\nSkipped/Timeout', 
#                 transform=ax2.transAxes, ha='center', va='center', 
#                 fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
#         ax2.set_title('GPU HDBSCAN Results (Skipped)', fontweight='bold')
#     ax2.set_xlabel(x_label)
#     ax2.set_ylabel(y_label)
#     ax2.grid(True, alpha=0.3)
    
#     # Plot 3: Sklearn HDBSCAN
#     ax3 = axes[0, 2]
#     if sklearn_labels is not None:
#         scatter3 = ax3.scatter(X_plot[:, 0], X_plot[:, 1], c=sklearn_labels, 
#                               cmap=cmap_discrete, s=30, alpha=0.7)
#         ax3.set_title('Sklearn HDBSCAN Results', fontweight='bold')
#     else:
#         ax3.text(0.5, 0.5, 'Sklearn HDBSCAN\nSkipped/Timeout', 
#                 transform=ax3.transAxes, ha='center', va='center', 
#                 fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
#         ax3.set_title('Sklearn HDBSCAN Results (Skipped)', fontweight='bold')
#     ax3.set_xlabel(x_label)
#     ax3.set_ylabel(y_label)
#     ax3.grid(True, alpha=0.3)
    
#     # Plot 4: DBSCAN
#     ax4 = axes[1, 0]
#     if dbscan_labels is not None:
#         scatter4 = ax4.scatter(X_plot[:, 0], X_plot[:, 1], c=dbscan_labels, 
#                               cmap=cmap_discrete, s=30, alpha=0.7)
#         ax4.set_title('DBSCAN Results', fontweight='bold')
#     else:
#         ax4.text(0.5, 0.5, 'DBSCAN\nSkipped/Timeout', 
#                 transform=ax4.transAxes, ha='center', va='center', 
#                 fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
#         ax4.set_title('DBSCAN Results (Skipped)', fontweight='bold')
#     ax4.set_xlabel(x_label)
#     ax4.set_ylabel(y_label)
#     ax4.grid(True, alpha=0.3)
    
#     # Plot 5: Metrics Summary - UPDATED
#     ax5 = axes[1, 1]
#     ax5.axis('off')

#     start = gt_stats['start_time'] / 1e9
#     end = gt_stats['end_time'] / 1e9
#     interval = end - start

#     # Helper function to safely format metric values
#     def safe_format(value, format_str="{:.3f}", default="N/A"):
#         if value is None:
#             return default
#         try:
#             return format_str.format(value)
#         except:
#             return default

#     # Helper function to safely access nested dict values
#     def safe_get_nested(d, keys, default="N/A"):
#         if d is None:
#             return default
#         try:
#             current = d
#             for key in keys:
#                 current = current[key]
#             return current if current is not None else default
#         except (KeyError, TypeError):
#             return default

#     # Create metrics table with guards
#     metrics_text = f"""
#     Dataset Statistics:
#     • Batch Interval ({interval:.3f}s): {start:.3f}s - {end:.3f}s
#     • Samples: {gt_stats['N_Samples']:,}
#     • True Clusters: {gt_stats['N_True_Clusters']}
    
#     GPU HDBSCAN vs Ground Truth:
#     • Time: {safe_format(gpu_metrics['time'] if gpu_metrics else None)}
#     • Memory: {safe_format(gpu_metrics['mem'] if gpu_metrics else None)}
#     • Clusters Found: {safe_format(gpu_metrics['N_Clusters'] if gpu_metrics else None, "{}", "N/A")}
#     • Noise Points: {safe_format(gpu_metrics['N_Noise'] if gpu_metrics else None, "{}", "N/A")}
#     • Perfectly Separated: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_perfectly_separated'])}
#     • Broken Up Clusters: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_broken_up'])}
#     • Missing Clusters: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_missing'])}
#     • Incorrectly Merged: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_incorrectly_merged'])}
#     • False Clusters: {safe_get_nested(gpu_metrics, ['detailed_quality', 'summary', 'n_false_clusters'])}
#     • Homogeneity: {safe_format(gpu_metrics['Homogeneity'] if gpu_metrics else None)}
#     • V-Measure: {safe_format(gpu_metrics['V_Measure'] if gpu_metrics else None)}

#     Sklearn HDBSCAN vs Ground Truth:
#     • Time: {safe_format(sklearn_metrics['time'] if sklearn_metrics else None)}
#     • Memory: {safe_format(sklearn_metrics['mem'] if sklearn_metrics else None)}
#     • Clusters Found: {safe_format(sklearn_metrics['N_Clusters'] if sklearn_metrics else None, "{}", "N/A")}
#     • Noise Points: {safe_format(sklearn_metrics['N_Noise'] if sklearn_metrics else None, "{}", "N/A")}
#     • Perfectly Separated: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_perfectly_separated'])}
#     • Broken Up Clusters: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_broken_up'])}
#     • Missing Clusters: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_missing'])}
#     • Incorrectly Merged: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_incorrectly_merged'])}
#     • False Clusters: {safe_get_nested(sklearn_metrics, ['detailed_quality', 'summary', 'n_false_clusters'])}
#     • Homogeneity: {safe_format(sklearn_metrics['Homogeneity'] if sklearn_metrics else None)}
#     • V-Measure: {safe_format(sklearn_metrics['V_Measure'] if sklearn_metrics else None)}
    
#     DBSCAN vs Ground Truth:
#     • Time: {safe_format(dbscan_metrics['time'] if dbscan_metrics else None)}
#     • Memory: {safe_format(dbscan_metrics['mem'] if dbscan_metrics else None)}
#     • Clusters Found: {safe_format(dbscan_metrics['N_Clusters'] if dbscan_metrics else None, "{}", "N/A")}
#     • Noise Points: {safe_format(dbscan_metrics['N_Noise'] if dbscan_metrics else None, "{}", "N/A")}
#     • Perfectly Separated: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_perfectly_separated'])}
#     • Broken Up Clusters: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_broken_up'])}
#     • Missing Clusters: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_missing'])}
#     • Incorrectly Merged: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_incorrectly_merged'])}
#     • False Clusters: {safe_get_nested(dbscan_metrics, ['detailed_quality', 'summary', 'n_false_clusters'])}
#     • Homogeneity: {safe_format(dbscan_metrics['Homogeneity'] if dbscan_metrics else None)}
#     • V-Measure: {safe_format(dbscan_metrics['V_Measure'] if dbscan_metrics else None)}

#     """
    
#     ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, 
#              fontsize=9, verticalalignment='top', fontfamily='monospace')
    
#     # Plot 6: Algorithm Agreement - NEW PLOT
#     ax6 = axes[1, 2]
#     ax6.axis('off')
    
#     agreement_text = f"""
#     Algorithm Agreement:
    
#     GPU vs Sklearn HDBSCAN:
#     • ARI: {safe_format(algo_agreement['ARI'] if algo_agreement else None)}
#     • NMI: {safe_format(algo_agreement['NMI'] if algo_agreement else None)}
#     • V-Measure: {safe_format(algo_agreement['V_Measure'] if algo_agreement else None)}
    
#     GPU HDBSCAN vs DBSCAN:
#     • ARI: {safe_format(algo_agreement['GPU_vs_DBSCAN_ARI'] if algo_agreement else None)}
    
#     Sklearn HDBSCAN vs DBSCAN:
#     • ARI: {safe_format(algo_agreement['Sklearn_vs_DBSCAN_ARI'] if algo_agreement else None)}
    
#     """
    
#     ax6.text(0.05, 0.95, agreement_text, transform=ax6.transAxes, 
#              fontsize=9, verticalalignment='top', fontfamily='monospace',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
#     plt.tight_layout()
    
#     # Save the plot
#     save_path = os.path.join(save_dir, f'{batch_name}_clustering_analysis.png')
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print(f"Comprehensive plot saved to: {save_path}")


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

def plot_clusters_2d(X, labels, feature_names=None, title="GPU HDBSCAN Clustering Results", save_path=None):
    """
    Plot 2D clustering results using matplotlib and the first two features (pulse width and frequency).
    
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
        Path to save the plot (PNG)
    """
    
    # Use first two dimensions
    X_2d = X[:, :2]
    
    # Set default feature names
    if feature_names is None:
        feature_names = ['Pulse Width', 'Frequency']
    
    # Get unique cluster labels
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_count = np.sum(labels == -1)
    
    print(f"Found {n_clusters} clusters and {noise_count} noise points")

    # Setup plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c='black', marker='x', s=50, alpha=0.6, label='Noise')
        else:
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



def predict_completion_time(sample_sizes, times, target_size):
    """Predict completion time using polynomial fitting without scipy"""
    # Filter out None values and convert to numpy arrays
    valid_indices = [i for i, t in enumerate(times) if t is not None]
    if len(valid_indices) < 2:
        return None

    valid_sizes = np.array([sample_sizes[i] for i in valid_indices])
    valid_times = np.array([times[i] for i in valid_indices])

    try:
        # Try quadratic fit
        coeffs = np.polyfit(valid_sizes, valid_times, deg=2)  # a*x^2 + b*x + c
        prediction = np.polyval(coeffs, target_size)

        # Ensure prediction is reasonable
        if prediction > 0 and prediction < 7200:
            return prediction

        # Fallback: linear extrapolation
        if len(valid_times) >= 2:
            slope = (valid_times[-1] - valid_times[-2]) / (valid_sizes[-1] - valid_sizes[-2])
            prediction = valid_times[-1] + slope * (target_size - valid_sizes[-1])
            return max(0, prediction)

    except:
        pass

    return None

def create_speed_benchmark_plot_new_metrics(results_df, output_dir, timeout):
    """Create professional benchmark visualization, now over Number of Emitters."""
    
    # Set style
    plt.style.use('ggplot')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Clustering Algorithm Performance Benchmark', fontsize=16, fontweight='bold')
    
    # ---- extract x-axis values ----
    emitters = results_df['Emitters'].values
    
    # ---- 1. Execution Time vs #Emitters ----
    ax1.set_title('Execution Time vs Number of Emitters', fontweight='bold')
    ax1.set_xlabel('Number of Emitters')
    ax1.set_ylabel('Execution Time (seconds)')
    
    # gather times
    gpu_times     = [row['GPU_Time']     if not row['GPU_Timeout']     else None for _, row in results_df.iterrows()]
    sklearn_times = [row['Sklearn_Time'] if not row['Sklearn_Timeout'] else None for _, row in results_df.iterrows()]
    dbscan_times  = [row['DBSCAN_Time']  if not row['DBSCAN_Timeout']  else None for _, row in results_df.iterrows()]
    
    # masks
    mask_gpu     = [t is not None for t in gpu_times]
    mask_sklearn = [t is not None for t in sklearn_times]
    mask_dbscan  = [t is not None for t in dbscan_times]

    # plot actual
    line_gpu     = ax1.plot(emitters[mask_gpu],     np.array(gpu_times)[mask_gpu],     'o-', label='GPU HDBSCAN',     linewidth=2, markersize=6)[0] if any(mask_gpu) else None
    line_sklearn = ax1.plot(emitters[mask_sklearn], np.array(sklearn_times)[mask_sklearn], 's-', label='Sklearn HDBSCAN', linewidth=2, markersize=6)[0] if any(mask_sklearn) else None
    line_dbscan  = ax1.plot(emitters[mask_dbscan],  np.array(dbscan_times)[mask_dbscan],  '^-', label='DBSCAN',          linewidth=2, markersize=6)[0] if any(mask_dbscan) else None

    # build valid data for predictions
    valid_gpu     = [(e, t) for e, t in zip(emitters,     gpu_times)     if t is not None]
    valid_sklearn = [(e, t) for e, t in zip(emitters, sklearn_times) if t is not None]
    valid_dbscan  = [(e, t) for e, t in zip(emitters,  dbscan_times)  if t is not None]

    # predicted points
    pred_gpu_e,     pred_gpu_t     = [], []
    pred_sklearn_e, pred_sklearn_t = [], []
    pred_dbscan_e,  pred_dbscan_t  = [], []

    for n_emit, g_t, s_t, d_t in zip(emitters, gpu_times, sklearn_times, dbscan_times):
        # GPU
        if g_t is None and len(valid_gpu) >= 2:
            xs, ys = zip(*valid_gpu)
            p = predict_completion_time(xs, ys, n_emit)
            if p and p > 0:
                pred_gpu_e.append(n_emit); pred_gpu_t.append(p)
                c = line_gpu.get_color() if line_gpu else 'blue'
                ax1.plot(n_emit, p, 'o', color=c, markersize=8, alpha=0.7)
                ax1.annotate(f'~{p/60:.1f}min', (n_emit, p), xytext=(5,5), textcoords='offset points', fontsize=8)

        # Sklearn
        if s_t is None and len(valid_sklearn) >= 2:
            xs, ys = zip(*valid_sklearn)
            p = predict_completion_time(xs, ys, n_emit)
            if p and p > 0:
                pred_sklearn_e.append(n_emit); pred_sklearn_t.append(p)
                c = line_sklearn.get_color() if line_sklearn else 'orange'
                ax1.plot(n_emit, p, 's', color=c, markersize=8, alpha=0.7)
                ax1.annotate(f'~{p/60:.1f}min', (n_emit, p), xytext=(5,5), textcoords='offset points', fontsize=8)

        # DBSCAN
        if d_t is None and len(valid_dbscan) >= 2:
            xs, ys = zip(*valid_dbscan)
            p = predict_completion_time(xs, ys, n_emit)
            if p and p > 0:
                pred_dbscan_e.append(n_emit); pred_dbscan_t.append(p)
                c = line_dbscan.get_color() if line_dbscan else 'green'
                ax1.plot(n_emit, p, '^', color=c, markersize=8, alpha=0.7)
                ax1.annotate(f'~{p/60:.1f}min', (n_emit, p), xytext=(5,5), textcoords='offset points', fontsize=8)

    # connect predicted lines
    def _connect(pred_e, pred_t, mask, line, valid):
        if pred_e and line:
            actual_e = emitters[mask]
            actual_t = np.array(valid) if isinstance(valid, list) else np.array([t for (_,t) in valid])
            # last actual point
            last_e, last_t = actual_e[-1], actual_t[-1]
            xs = [last_e] + pred_e
            ys = [last_t] + pred_t
            ax1.plot(xs, ys, linestyle='--', color=line.get_color(), linewidth=1.5)

    _connect(pred_gpu_e,     pred_gpu_t,     mask_gpu,     line_gpu,     [t for (_,t) in valid_gpu])
    _connect(pred_sklearn_e, pred_sklearn_t, mask_sklearn, line_sklearn, [t for (_,t) in valid_sklearn])
    _connect(pred_dbscan_e,  pred_dbscan_t,  mask_dbscan,  line_dbscan,  [t for (_,t) in valid_dbscan])

    # timeout
    ax1.axhline(timeout, color='red', linestyle='--', alpha=0.7, label=f'Timeout ({timeout/60:.0f} min)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)


    # ---- 2. Memory Usage vs #Emitters ----
    ax2.set_title('Memory Usage vs Number of Emitters', fontweight='bold')
    for col, marker in [('GPU_Memory','o-'), ('Sklearn_Memory','s-'), ('DBSCAN_Memory','^-')]:
        m = results_df[col].notna() & (results_df[col] != 0)
        if m.any():
            algo = col.replace('_Memory','')
            ax2.plot(emitters[m], results_df.loc[m, col], marker, label=algo, linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Emitters')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)


    # ---- 3. Clustering Quality (Cluster Correctness) ----
    ax3.set_title('Clustering Quality (Cluster Correctness)', fontweight='bold')
    for algo in ['GPU', 'Sklearn', 'DBSCAN']:
        ccol = f'{algo}_Correct_Clusters'
        icol = f'{algo}_Incorrect_Clusters'
        m = results_df[ccol].notna() & results_df[icol].notna()
        if m.any():
            correct = results_df.loc[m, ccol]
            incorrect = results_df.loc[m, icol]
            ratio = correct / (correct + incorrect)
            ax3.plot(emitters[m], ratio, 'o-', label=algo, linewidth=2, markersize=6)

    ax3.set_xlabel('Number of Emitters')
    ax3.set_ylabel('Cluster Correctness')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)


    # ---- 4. Number of Clusters Found vs #Emitters ----
    ax4.set_title('Number of Clusters Found', fontweight='bold')
    for col, marker in [('GPU_Clusters','o-'), ('Sklearn_Clusters','s-'), ('DBSCAN_Clusters','^-')]:
        m = results_df[col].notna()
        if m.any():
            algo = col.replace('_Clusters','')
            ax4.plot(emitters[m], results_df.loc[m, col], marker, label=algo, linewidth=2, markersize=6)

    ax4.set_xlabel('Number of Emitters')
    ax4.set_ylabel('Number of Clusters')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'speed_benchmark_comprehensive.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def create_speed_benchmark_plot(results_df, output_dir, timeout):
    """Create professional benchmark visualization"""
    
    # Set style
    plt.style.use('ggplot')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Clustering Algorithm Performance Benchmark', fontsize=16, fontweight='bold')
    
    sample_sizes = results_df['Samples'].values
    
    # 1. Execution Time Comparison
    ax1.set_title('Execution Time vs Sample Size', fontweight='bold')
    
    # Prepare data for plotting
    gpu_times = []
    sklearn_times = []
    dbscan_times = []
    
    for _, row in results_df.iterrows():
        gpu_times.append(row['GPU_Time'] if not row['GPU_Timeout'] else None)
        sklearn_times.append(row['Sklearn_Time'] if not row['Sklearn_Timeout'] else None)
        dbscan_times.append(row['DBSCAN_Time'] if not row['DBSCAN_Timeout'] else None)
    
    # Plot actual measurements
    gpu_mask = [t is not None for t in gpu_times]
    sklearn_mask = [t is not None for t in sklearn_times]
    dbscan_mask = [t is not None for t in dbscan_times]
    
    # Plot lines and store colors
    gpu_line = None
    sklearn_line = None
    dbscan_line = None
    
    if any(gpu_mask):
        gpu_line = ax1.plot(np.array(sample_sizes)[gpu_mask], np.array(gpu_times)[gpu_mask], 
                           'o-', label='GPU HDBSCAN', linewidth=2, markersize=6)[0]
    
    if any(sklearn_mask):
        sklearn_line = ax1.plot(np.array(sample_sizes)[sklearn_mask], np.array(sklearn_times)[sklearn_mask], 
                               's-', label='Sklearn HDBSCAN', linewidth=2, markersize=6)[0]
    
    if any(dbscan_mask):
        dbscan_line = ax1.plot(np.array(sample_sizes)[dbscan_mask], np.array(dbscan_times)[dbscan_mask], 
                              '^-', label='DBSCAN', linewidth=2, markersize=6)[0]

    # Store predicted points
    gpu_pred_x, gpu_pred_y = [], []
    sklearn_pred_x, sklearn_pred_y = [], []
    dbscan_pred_x, dbscan_pred_y = [], []

    # Build valid data points for prediction
    valid_gpu_data = [(s, t) for s, t in zip(sample_sizes, gpu_times) if t is not None]
    valid_sklearn_data = [(s, t) for s, t in zip(sample_sizes, sklearn_times) if t is not None]
    valid_dbscan_data = [(s, t) for s, t in zip(sample_sizes, dbscan_times) if t is not None]

    for i, (size, gpu_time, sklearn_time, dbscan_time) in enumerate(zip(sample_sizes, gpu_times, sklearn_times, dbscan_times)):
        # GPU HDBSCAN predictions
        if gpu_time is None and len(valid_gpu_data) >= 2:
            valid_sizes = [s for s, t in valid_gpu_data]
            valid_times = [t for s, t in valid_gpu_data]
            prediction = predict_completion_time(valid_sizes, valid_times, size)
            if prediction and prediction > 0:
                gpu_pred_x.append(size)
                gpu_pred_y.append(prediction)
                color = gpu_line.get_color() if gpu_line else 'blue'
                ax1.plot(size, prediction, 'o', color=color, markersize=8, alpha=0.7)
                ax1.annotate(f'~{prediction/60:.1f}min', (size, prediction), 
                             xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Sklearn HDBSCAN predictions
        if sklearn_time is None and len(valid_sklearn_data) >= 2:
            valid_sizes = [s for s, t in valid_sklearn_data]
            valid_times = [t for s, t in valid_sklearn_data]
            prediction = predict_completion_time(valid_sizes, valid_times, size)
            if prediction and prediction > 0:
                sklearn_pred_x.append(size)
                sklearn_pred_y.append(prediction)
                color = sklearn_line.get_color() if sklearn_line else 'orange'
                ax1.plot(size, prediction, 's', color=color, markersize=8, alpha=0.7)
                ax1.annotate(f'~{prediction/60:.1f}min', (size, prediction), 
                             xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # DBSCAN predictions
        if dbscan_time is None and len(valid_dbscan_data) >= 2:
            valid_sizes = [s for s, t in valid_dbscan_data]
            valid_times = [t for s, t in valid_dbscan_data]
            prediction = predict_completion_time(valid_sizes, valid_times, size)
            if prediction and prediction > 0:
                dbscan_pred_x.append(size)
                dbscan_pred_y.append(prediction)
                color = dbscan_line.get_color() if dbscan_line else 'green'
                ax1.plot(size, prediction, '^', color=color, markersize=8, alpha=0.7)
                ax1.annotate(f'~{prediction/60:.1f}min', (size, prediction), 
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Plot predicted lines (dashed) - connect from last actual point to predictions
    if gpu_pred_x and gpu_line:
        # Get last actual point
        actual_gpu_x = np.array(sample_sizes)[gpu_mask]
        actual_gpu_y = np.array(gpu_times)[gpu_mask]
        if len(actual_gpu_x) > 0:
            # Connect last actual point to first predicted point, then all predicted points
            all_pred_x = [actual_gpu_x[-1]] + gpu_pred_x
            all_pred_y = [actual_gpu_y[-1]] + gpu_pred_y
            ax1.plot(all_pred_x, all_pred_y, linestyle='--', color=gpu_line.get_color(), linewidth=1.5)
        else:
            # If no actual points, just connect predicted points
            ax1.plot(gpu_pred_x, gpu_pred_y, linestyle='--', color=gpu_line.get_color(), linewidth=1.5)
    
    if sklearn_pred_x and sklearn_line:
        actual_sklearn_x = np.array(sample_sizes)[sklearn_mask]
        actual_sklearn_y = np.array(sklearn_times)[sklearn_mask]
        if len(actual_sklearn_x) > 0:
            all_pred_x = [actual_sklearn_x[-1]] + sklearn_pred_x
            all_pred_y = [actual_sklearn_y[-1]] + sklearn_pred_y
            ax1.plot(all_pred_x, all_pred_y, linestyle='--', color=sklearn_line.get_color(), linewidth=1.5)
        else:
            ax1.plot(sklearn_pred_x, sklearn_pred_y, linestyle='--', color=sklearn_line.get_color(), linewidth=1.5)
    
    if dbscan_pred_x and dbscan_line:
        actual_dbscan_x = np.array(sample_sizes)[dbscan_mask]
        actual_dbscan_y = np.array(dbscan_times)[dbscan_mask]
        if len(actual_dbscan_x) > 0:
            all_pred_x = [actual_dbscan_x[-1]] + dbscan_pred_x
            all_pred_y = [actual_dbscan_y[-1]] + dbscan_pred_y
            ax1.plot(all_pred_x, all_pred_y, linestyle='--', color=dbscan_line.get_color(), linewidth=1.5)
        else:
            ax1.plot(dbscan_pred_x, dbscan_pred_y, linestyle='--', color=dbscan_line.get_color(), linewidth=1.5)
    
    # Add timeout line
    ax1.axhline(y=timeout, color='red', linestyle='--', alpha=0.7, label=f'Timeout ({timeout/60:.0f}min)')
    
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory Usage Comparison
    ax2.set_title('Memory Usage vs Sample Size', fontweight='bold')
    
    for col in ['GPU_Memory', 'Sklearn_Memory', 'DBSCAN_Memory']:
        mask = results_df[col].notna() & (results_df[col] != 0) 
        if mask.any():
            ax2.plot(results_df.loc[mask, 'Samples'], results_df.loc[mask, col], 
                    'o-', label=col.replace('_Memory', ''), linewidth=2, markersize=6)
    
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Clustering Quality (Homogeneity)
    ax3.set_title('Clustering Quality (Homogeneity Score)', fontweight='bold')
    
    for col in ['GPU_Homogeneity', 'Sklearn_Homogeneity', 'DBSCAN_Homogeneity']:
        mask = results_df[col].notna()
        if mask.any():
            ax3.plot(results_df.loc[mask, 'Samples'], results_df.loc[mask, col], 
                    'o-', label=col.replace('_Homogeneity', ''), linewidth=2, markersize=6)
    
    ax3.set_xlabel('Number of Samples')
    ax3.set_ylabel('Homogeneity Score')
    ax3.set_xscale('log')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Number of Clusters Found
    ax4.set_title('Number of Clusters Found', fontweight='bold')
    
    for col in ['GPU_Clusters', 'Sklearn_Clusters', 'DBSCAN_Clusters']:
        mask = results_df[col].notna()
        if mask.any():
            ax4.plot(results_df.loc[mask, 'Samples'], results_df.loc[mask, col], 
                    'o-', label=col.replace('_Clusters', ''), linewidth=2, markersize=6)
    
    ax4.set_xlabel('Number of Samples')
    ax4.set_ylabel('Number of Clusters')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_benchmark_comprehensive.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()