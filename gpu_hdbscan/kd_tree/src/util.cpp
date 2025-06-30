#include "kd_tree/include/util.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <limits>

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
 * Read `dimensions` features + 1 label (last column) from CSV.
 * If a line has fewer than dimensions+1 columns, it's skipped.
 * Always reads the last column as emitter ID, regardless of total columns.
 */
std::vector<Point>
readPointsFromFile(const std::string& filename,
                   int dimensions,
                   std::vector<int>& labels)
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
        // need at least dimensions features + 1 label (last column)
        if ((int)tok.size() < dimensions + 1) {
            std::cerr << "Warning: too few columns (" 
                      << tok.size() << "), need at least " 
                      << dimensions + 1 << ", skipping line\n";
            continue;
        }
        // parse first `dimensions` features
        Point p;
        p.reserve(dimensions);
        bool bad = false;
        for (int i = 0; i < dimensions; ++i) {
            try {
                p.push_back(std::stod(tok[i]));
            } catch (...) {
                std::cerr << "Warning: invalid float in col " 
                          << i << " ('" << tok[i] << "'), skipping line\n";
                bad = true;
                break;
            }
        }
        if (bad) continue;
        // parse *last* token as label (emitter ID)
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

// Print CLI Usage Instructions
void printUsage(char* prog) {
  std::cerr << R"(Usage:
    )" << prog << R"( --dimensions D --minpts K --input [FILENAME] --distMetric M --minkowskiP p --minclustersize min_cluster_size]

Parameters:
  --dimensions D   Number of features per data point (integer > 0)
  --minpts K       Minimum points for core‐distance (integer > 0)
  --input FILE     Path to input file
  --distMetric M   Distance metric (1= Manhattan,2= Euclidean,3 = Chebyshev,4 = Minkowski)
  --minkowskiP p   P-Value for Minkowski Distance
  --minclustersize min_cluster_size minimum cluster size value for cluster extraction
)";
}
