from my_hdbscan.hdbscan_scratch import *
from sklearn.cluster import HDBSCAN  # closest alternative in sklearn
# import contrib_hdbscan.hdbscan as contrib
import hdbscan
# from cpp_hdbscan import *


if __name__ == "__main__":
    # Generate different types of synthetic datasets and test clustering
    
    # Set parameters
    min_samples = 3
    min_cluster_size = 50
    
    # Performance comparison results storage
    performance_results = []
    all_comparisons = []
    
    # Test on different datasets
    for data_type in ['anisotropic']:
    # for data_type in ['blobs', 'circles', 'moons', 'anisotropic']:
        print(f"\nTesting on {data_type} dataset...")
        
        # Generate data
        X, true_labels = generate_custom_data(data_type=data_type, n_samples=500)
        
        # Convert numpy array to list of tuples for our implementation
        data_pts = [tuple(x) for x in X]

        # Run conotrib HDBSCAN implementation with performance tracking
        print("Running new HDBSCAN implementation...")
        def run_new_hdbscan(X):
            labels, *_ = hdbscan.hdbscan(
                X,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
            return labels

        new_result, new_time, new_memory = track_performance(
            run_new_hdbscan, X
        )
        
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


        performance_results.append({
            'Dataset': data_type,
            'Custom Time (s)': custom_time,
            'Sklearn Time (s)': sklearn_time,
            'New Time (s)': new_time,
            'Time Ratio (Sk/Custom)': sklearn_time / custom_time if custom_time > 0 else float('inf'),
            'Time Ratio (New/Custom)': new_time / custom_time if custom_time > 0 else float('inf'),
            'Custom Memory (MB)': custom_memory,
            'Sklearn Memory (MB)': sklearn_memory,
            'New Memory (MB)': new_memory,
            'Memory Ratio (Sk/Custom)': sklearn_memory / custom_memory if custom_memory > 0 else float('inf'),
            'Memory Ratio (New/Custom)': new_memory / custom_memory if custom_memory > 0 else float('inf'),
        })

         # Store the clustering results for final combined plot
        all_comparisons.append({
            'X': X,
            'custom_labels': custom_labels,
            'sklearn_labels': sklearn_result,
            'new_labels': new_result,
            'title': f'{data_type.capitalize()} Dataset'
        })


        
        # # Optionally still save individual plots if needed
        # fig = plot_clusters_comparsion(X, custom_labels, sklearn_result, title=f'HDBSCAN - {data_type}')
        # fig.savefig(f'hdbscan_comparison_{data_type}.png')
        # plt.close(fig)

    # Save the figure
    final_fig = plot_all_comparisons(all_comparisons)
    final_fig.savefig('hdbscan_comparison_all_datasets.png', bbox_inches='tight')
    plt.close(final_fig)
    
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