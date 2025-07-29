#include "kd_tree/include/util.hpp"
#include "kd_tree/include/types.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <limits>
#include <set>
#include <filesystem>
//------------------------------------------------------------------------------
// Scale all points so that x and y each span [0,1]. 
// If all x's (or y's) are equal, they get mapped to 0.0.
//------------------------------------------------------------------------------
void normalizePoints(std::vector<Point>& pts) {
  if (pts.empty()) return;
  size_t D = pts[0].size();
  std::vector<double> minV(D, std::numeric_limits<double>::infinity());
  std::vector<double> maxV(D, -std::numeric_limits<double>::infinity());
  for (auto& p : pts) {
    for (size_t i = 0; i < D; ++i) {
      minV[i] = std::min(minV[i], p[i]);
      maxV[i] = std::max(maxV[i], p[i]);
    }
  }
  for (auto& p : pts) {
    for (size_t i = 0; i < D; ++i) {
      double range = maxV[i] - minV[i];
      p[i] = (range > 0.0 ? (p[i] - minV[i]) / range : 0.0);
    }
  }
}
/**
 * Read features from CSV, skipping specified columns and always using last column as emitter ID.
 * @param filename: CSV file to read
 * @param dimensions: number of feature dimensions to extract (after skipping columns)
 * @param labels: output vector for emitter IDs
 * @param skip_columns: set of column indices to skip (0-based, excluding the last column)
 * @return vector of Points with selected features
 */
std::vector<Point>
readPointsFromFile(const std::string& filename,
                   int dimensions,
                   std::vector<int>& labels,
                   const std::set<int>& skip_columns)
{
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Unable to open file: " + filename);
    std::vector<Point> pts;
    std::string line;
    // skip header (first line)
    std::getline(in, line);
    while (std::getline(in, line)) {
        if (line.empty() || line[0]=='#') continue;
        // tokenize on comma
        std::vector<std::string> tok;
        {
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                tok.push_back(cell);
            }
        }

        // Count available feature columns (excluding last column and skip columns)
        int available_features = 0;
        for (int i = 0; i < (int)tok.size() - 1; ++i) {  // -1 to exclude last column (EmitterID)
            if (skip_columns.find(i) == skip_columns.end()) {
                available_features++;
            }
        }

        // Check if we have enough feature columns
        if (available_features < dimensions) {
            std::cerr << "Warning: not enough feature columns (" 
                      << available_features << " available, " << dimensions 
                      << " requested), skipping line\n";
            continue;
        }

        // Need at least one column for EmitterID
        if ((int)tok.size() < 2) {
            std::cerr << "Warning: too few total columns (" 
                      << tok.size() << "), need at least 2, skipping line\n";
            continue;
        }

        // Parse features (skip specified columns and last column)
        Point p;
        p.reserve(dimensions);
        bool bad = false;
        int features_added = 0;

        for (int i = 0; i < (int)tok.size() - 1 && features_added < dimensions; ++i) {
            // Skip if this column is in the skip set
            if (skip_columns.find(i) != skip_columns.end()) {
                continue;
            }

            try {
                p.push_back(std::stod(tok[i]));
                features_added++;
            } catch (...) {
                std::cerr << "Warning: invalid float in col " 
                          << i << " ('" << tok[i] << "'), skipping line\n";
                bad = true;
                break;
            }
        }

        if (bad || features_added != dimensions) continue;

        // parse last token as label (emitter ID)
        int lbl;
        try {
            lbl = std::stoi(tok.back());
        } catch (...) {
            std::cerr << "Warning: invalid emitter ID '" 
                      << tok.back() << "', skipping line\n";
            continue;
        }
        pts.push_back(std::move(p));
        labels.push_back(lbl);
    }
    return pts;
}

/**
 * Read features from CSV, using the first 'dimensions' columns as features.
 * @param filename: CSV file to read
 * @param dimensions: number of feature dimensions to extract from start of each row
 * @param skip_columns: set of column indices to skip (0-based)
 * @return vector of Points with selected features
 */
std::vector<Point>
readPointsFromFile(const std::string& filename,
                   int dimensions,
                   const std::set<int>& skip_columns)
{
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Unable to open file: " + filename);
    std::vector<Point> pts;
    std::string line;
    // skip header (first line)
    std::getline(in, line);
    
    while (std::getline(in, line)) {
        if (line.empty() || line[0]=='#') continue;
        
        // tokenize on comma
        std::vector<std::string> tok;
        {
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                tok.push_back(cell);
            }
        }
        
        // Count available feature columns
        int available_features = 0;
        for (int i = 0; i < (int)tok.size(); ++i) {
            if (skip_columns.find(i) == skip_columns.end()) {
                available_features++;
            }
        }
        
        // Check if we have enough feature columns
        if (available_features < dimensions) {
            std::cerr << "Warning: not enough feature columns (" 
                      << available_features << " available, " << dimensions 
                      << " requested), skipping line\n";
            continue;
        }
        
        // Parse features - take first 'dimensions' non-skipped columns
        Point p;
        p.reserve(dimensions);
        bool bad = false;
        int features_added = 0;
        
        for (int i = 0; i < (int)tok.size() && features_added < dimensions; ++i) {
            // Skip if this column is in the skip set
            if (skip_columns.find(i) != skip_columns.end()) {
                continue;
            }
            
            try {
                p.push_back(std::stod(tok[i]));
                features_added++;
            } catch (...) {
                std::cerr << "Warning: invalid float in col " 
                          << i << " ('" << tok[i] << "'), skipping line\n";
                bad = true;
                break;
            }
        }
        
        if (bad || features_added != dimensions) continue;
        
        pts.push_back(std::move(p));
    }
    
    return pts;
}
// Print CLI Usage Instructions
void printUsage(char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n";
    std::cerr << "Options:\n";
    std::cerr << "  --dimensions <int>      Number of feature dimensions to use\n";
    std::cerr << "  --minpts <int>          Minimum points for core distance\n";
    std::cerr << "  --input <filename>      Input CSV file\n";
    std::cerr << "  --distMetric <int>      Distance metric (1:Manhattan, 2:Euclidean, 3:Chebyshev, 4:Minkowski, 5:DSO)\n";
    std::cerr << "  --clusterMethod <int>   Cluster Method Choice (1:Excess Of Mass, 2:Leaf)\n";
    std::cerr << "  --minkowskiP <float>    P-value for Minkowski distance\n";
    std::cerr << "  --minclustersize <int>  Minimum cluster size\n";
    std::cerr << "  --skip-toa              Skip TOA column (index 0)\n";
    std::cerr << "  --skip-amp              Skip Amplitude column (index 3)\n";
    std::cerr << "  --skip-columns <list>   Skip specific columns (comma-separated indices)\n";
    std::cerr << "  --noBenchMark           Controls Which readPointsFromFile function is used";
    std::cerr << "  --quiet, -q             Suppress debug output\n";
    std::cerr << "  --help, -h              Show this help message\n";
}

// // Compute normalized inverse standard deviation weights
// std::vector<double> computeNormalizedInverseStdDevWeights(const std::vector<Point>& points) {
//     if (points.empty()) throw std::invalid_argument("Points cannot be empty.");
//     const size_t D = points[0].size();
//     const size_t N = points.size();

//     std::vector<double> means(D, 0.0);
//     std::vector<double> stds(D, 0.0);
    
//     // Step 1: Compute mean per dimension
//     for (const Point& p : points) {
//         for (size_t d = 0; d < D; ++d) {
//             means[d] += p[d];
//         }
//     }
//     for (size_t d = 0; d < D; ++d) {
//         means[d] /= N;
//     }

//     // Step 2: Compute std deviation per dimension
//     for (const Point& p : points) {
//         for (size_t d = 0; d < D; ++d) {
//             double diff = p[d] - means[d];
//             stds[d] += diff * diff;
//         }
//     }
//     for (size_t d = 0; d < D; ++d) {
//         stds[d] = std::sqrt(stds[d] / N);
//     }

//     // Step 3: Compute inverse std and normalize
//     std::vector<double> inv_std(D, 0.0);
//     double sum = 0.0;
//     for (size_t d = 0; d < D; ++d) {
//         if (stds[d] == 0.0) throw std::runtime_error("Standard deviation is zero in dimension.");
//         inv_std[d] = 1.0 / stds[d];
//         sum += inv_std[d];
//     }

//     for (size_t d = 0; d < D; ++d) {
//         inv_std[d] /= sum;  // normalize so weights sum to 1
//     }

//     return inv_std;
// }

// Helper function to compute percentile
double computePercentile(std::vector<double> values, double percentile) {
    if (values.empty()) throw std::invalid_argument("Values cannot be empty.");
    
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    
    if (percentile <= 0.0) return values[0];
    if (percentile >= 100.0) return values[n - 1];
    
    double index = (percentile / 100.0) * (n - 1);
    size_t lower_index = static_cast<size_t>(std::floor(index));
    size_t upper_index = static_cast<size_t>(std::ceil(index));
    
    if (lower_index == upper_index) {
        return values[lower_index];
    }
    
    double weight = index - lower_index;
    return values[lower_index] * (1.0 - weight) + values[upper_index] * weight;
}

// Compute weights based on std/range with normalization
std::vector<double> computeNormalizedStdRangeWeights(
    const std::vector<Point>& points,
    const std::vector<double>& stds
) {
    if (points.empty()) throw std::invalid_argument("Points cannot be empty.");
    if (stds.empty()) throw std::invalid_argument("Standard deviations cannot be empty.");
    
    const size_t D = points[0].size();
    if (stds.size() != D) {
        throw std::invalid_argument("Standard deviations size must match point dimensions.");
    }
    
    std::vector<double> weights(D, 0.0);
    
    // For each dimension, compute the appropriate range
    for (size_t d = 0; d < D; ++d) {
        // Extract values for this dimension
        std::vector<double> values;
        values.reserve(points.size());
        for (const Point& p : points) {
            values.push_back(p[d]);
        }
        
        // Compute percentiles
        double p10 = computePercentile(values, 10.0);
        double p25 = computePercentile(values, 25.0);
        double p75 = computePercentile(values, 75.0);
        double p90 = computePercentile(values, 90.0);
        
        double range_90_10 = p90 - p10;
        double range_75_25 = p75 - p25;
        
        // Choose range based on your criteria
        // If they differ significantly, use IQR (75th - 25th)
        // You can adjust this threshold as needed
        double range;
        double ratio = std::abs(range_90_10 - range_75_25) / std::max(range_90_10, range_75_25);
        
        if (ratio > 0.2) {  // Threshold for "significant difference" - adjust as needed
            range = range_75_25;
        } else {
            range = range_90_10;
        }
        
        if (range == 0.0) {
            throw std::runtime_error("Range is zero in dimension " + std::to_string(d));
        }
        
        // Compute std/range ratio
        double std_range_ratio = stds[d] / range;
        
        if (std_range_ratio == 0.0) {
            throw std::runtime_error("Standard deviation is zero in dimension " + std::to_string(d));
        }
        
        // Take reciprocal for weight (larger reciprocal = more weight)
        weights[d] = 1.0 / std_range_ratio;
    }
    
    // Normalize weights to sum to 1
    double sum = 0.0;
    for (double w : weights) {
        sum += w;
    }
    
    if (sum == 0.0) {
        throw std::runtime_error("Sum of weights is zero.");
    }
    
    for (size_t d = 0; d < D; ++d) {
        weights[d] /= sum;
    }
    
    return weights;
}


void outputClusterLabels(const std::vector<std::vector<int>>& clusters, int total_points) {
    // Create label array initialized to -1 (noise)
    std::vector<int> labels(total_points, -1);
    
    // Assign cluster labels
    for (int cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        for (int point_id : clusters[cluster_id]) {
            if (point_id >= 0 && point_id < total_points) {
                labels[point_id] = cluster_id;
            }
        }
    }
    
    // ALWAYS output cluster labels (needed for Python parsing)
    std::cout << "CLUSTER_LABELS:";
    for (int i = 0; i < labels.size(); ++i) {
        std::cout << " " << labels[i];
    }
    std::cout << std::endl;
    
    // Output cluster statistics (conditional)
    DEBUG_PRINT("CLUSTER_STATS:" << std::endl);
    DEBUG_PRINT("  Total points: " << total_points << std::endl);
    DEBUG_PRINT("  Number of clusters: " << clusters.size() << std::endl);
    int noise_count = std::count(labels.begin(), labels.end(), -1);
    DEBUG_PRINT("  Noise points: " << noise_count << std::endl);
    DEBUG_PRINT("  Clustered points: " << (total_points - noise_count) << std::endl);
}

void writeMSTEdges(const std::string& filename,
                   const std::vector<Edge>& mst_edges) {
    std::filesystem::path debug_dir = "debug_output";
    std::filesystem::create_directories(debug_dir);
    
    std::filesystem::path filepath = debug_dir / filename;
    
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Failed to open file " << filepath << " for writing\n";
        return;
    }

    // Write CSV header
    out << "u,v,weight\n";
    
    // Write each edge
    for (const auto& edge : mst_edges) {
        out << edge.u << "," << edge.v << "," << edge.weight << "\n";
    }
    
    out.close();
    std::cout << "MST edges written to " << filepath << "\n";
}

void writeMRDGraph(const std::string& filename,
                   const std::vector<std::vector<std::pair<int, double>>>& knn_graph) {
   std::filesystem::path debug_dir = "debug_output";
   std::filesystem::create_directories(debug_dir);
   std::filesystem::path filepath = debug_dir / filename;
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Failed to open file " << filename << " for writing\n";
        return;
    }

    for (int i = 0; i < knn_graph.size(); ++i) {
        out << i;  // query point index
        for (const auto& [nbr_idx, dist] : knn_graph[i]) {
            out << "," << nbr_idx << "," << dist;
        }
        out << "\n";
    }
    out.close();
}