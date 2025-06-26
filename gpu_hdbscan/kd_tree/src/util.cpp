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
 * Read points from a text or CSV file.
 *
 * Each non-empty line should contain features,
 * either separated by whitespace:
 *    1.23  4.56
 * or by comma:
 *    1.23,4.56
 *
 * Lines beginning with '#' or empty lines are skipped.
 */
std::vector<Point> readPointsFromFile(const std::string& filename, int dimensions) {
  std::ifstream in(filename);
  if (!in) throw std::runtime_error("Unable to open file");
  std::vector<Point> pts;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream iss(line);
    Point p; double val;
    while (iss >> val) p.push_back(val);
    if ((int)p.size() != dimensions) {
      std::cerr << "Warning: skipping line with wrong dimension\n";
    } else {
      pts.push_back(std::move(p));
    }
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
