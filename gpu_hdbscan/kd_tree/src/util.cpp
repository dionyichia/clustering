#include "kd_tree/include/util.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <limits>
#include <set>
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
    std::cerr << "  --minkowskiP <float>    P-value for Minkowski distance\n";
    std::cerr << "  --minclustersize <int>  Minimum cluster size\n";
    std::cerr << "  --skip-toa              Skip TOA column (index 0)\n";
    std::cerr << "  --skip-amp              Skip Amplitude column (index 3)\n";
    std::cerr << "  --skip-columns <list>   Skip specific columns (comma-separated indices)\n";
    std::cerr << "  --noBenchMark           Controls Which readPointsFromFile function is used";
    std::cerr << "  --quiet, -q             Suppress debug output\n";
    std::cerr << "  --help, -h              Show this help message\n";
}
