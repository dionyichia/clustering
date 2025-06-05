if __name__ == "__main__":
    # Generate different types of synthetic datasets and test clustering
    
    # Set parameters
    min_samples = 3
    min_cluster_size = 50
    
    # Performance comparison results storage
    performance_results = []
    
    # Test on different datasets
    # for data_type in ['blobs']:
    for data_type in ['blobs', 'circles', 'moons', 'anisotropic']:
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