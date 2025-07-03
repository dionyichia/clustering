#!/usr/bin/env python3
"""
Compute DBSCAN parameters (eps and min_samples) for multiple CSVs by running HDBSCAN.

Usage:
    python optimal_dbscan_params.py <csv_folder> [--min_cluster_size N] [--min_samples M] [--features col1 col2 ...] [--output OUT_TXT]

Outputs a text file with rows:
Name of File Min_Samples Epsilon
file1.csv 5 0.123456
file2.csv 6 0.234567
...
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate DBSCAN eps and min_samples for multiple CSVs via HDBSCAN."
    )
    parser.add_argument(
        "csv_folder", help="Path to folder containing CSV files."
    )
    parser.add_argument(
        "--min_cluster_size", type=int, default=10,
        help="min_cluster_size for HDBSCAN (default: 10)."
    )
    parser.add_argument(
        "--min_samples", type=int,
        help=(
            "Explicit min_samples for DBSCAN/HDBSCAN. "
            "If omitted, will be chosen via persistence scan."
        )
    )
    parser.add_argument(
        "--features", nargs="+",
        default=["PW(microsec)", "FREQ(MHz)", "AZ_S0(deg)", "EL_S0(deg)"],
        help=(
            "List of column names to use as features. "
            "If omitted, defaults to all columns or the defaults above."
        )
    )
    parser.add_argument(
        "--output", default="dbscan_params.txt",
        help="Output text file path (default: dbscan_params.txt)."
    )
    return parser.parse_args()


def find_best_min_samples(X, min_cluster_size, k_values):
    """
    Scan HDBSCAN over different min_samples (k) values, return the k that maximizes mean cluster persistence.
    """
    persistence_scores = {}
    for k in k_values:
        c = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=k).fit(X)
        scores = getattr(c, 'cluster_persistence_', None)
        persistence_scores[k] = float(np.mean(scores)) if scores is not None and len(scores) > 0 else 0.0
    best_k = max(persistence_scores, key=persistence_scores.get)
    return best_k


def compute_params(X, args):
    # Determine DBSCAN min_samples
    if args.min_samples is not None:
        db_min_samples = args.min_samples
    else:
        D = X.shape[1]
        candidate_ks = list(range(D + 1, 2 * D + 1))
        db_min_samples = find_best_min_samples(X, args.min_cluster_size, candidate_ks)

    # Fit HDBSCAN with chosen min_samples to compute core distances
    clusterer = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=db_min_samples
    ).fit(X)

    # Extract or compute core distances
    try:
        core_dists = clusterer.core_distances_
    except AttributeError:
        nbrs = NearestNeighbors(n_neighbors=db_min_samples, n_jobs=1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        core_dists = distances[:, -1]

    # Estimate eps via elbow method
    k_dist = np.sort(core_dists)
    n_pts = len(k_dist)
    idxs = np.arange(n_pts)
    p1 = np.array([0, k_dist[0]])
    p2 = np.array([n_pts - 1, k_dist[-1]])
    line_vec = p2 - p1
    vecs = np.vstack((idxs, k_dist)).T - p1
    cross = np.abs(line_vec[0] * vecs[:,1] - line_vec[1] * vecs[:,0])
    distances_to_line = cross / np.linalg.norm(line_vec)
    elbow_idx = int(np.argmax(distances_to_line))
    eps = float(k_dist[elbow_idx])

    return db_min_samples, eps


def main():
    args = parse_args()

    # Validate input directory
    if not os.path.isdir(args.csv_folder):
        print(f"Error: '{args.csv_folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Gather CSV files
    csv_files = [f for f in os.listdir(args.csv_folder) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in '{args.csv_folder}'.", file=sys.stderr)
        sys.exit(1)

    results = []
    for fname in sorted(csv_files):
        path = os.path.join(args.csv_folder, fname)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading {fname}: {e}", file=sys.stderr)
            continue

        # Select features
        if args.features:
            missing = set(args.features) - set(df.columns)
            if missing:
                print(f"Skipping {fname}: missing columns {missing}", file=sys.stderr)
                continue
            X = df[args.features].values
        else:
            X = df.values

        # Compute parameters
        min_samps, epsilon = compute_params(X, args)
        results.append((fname, min_samps, epsilon))
        print(f"Processed {fname}: min_samples={min_samps}, epsilon={epsilon:.6f}")

    # Write output
    try:
        with open(args.output, 'w') as out:
            out.write("Name of File Min_Samples Epsilon\n")
            for fname, min_samps, eps in results:
                out.write(f"{fname} {min_samps} {eps:.6f}\n")
        print(f"Wrote results for {len(results)} files to '{args.output}'.")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
