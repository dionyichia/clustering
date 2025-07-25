#pragma once
#include "types.hpp"
#include <string>
#include <vector>
#include <set>
/// Normalize each dimension of pts to [0,1]
void normalizePoints(std::vector<Point>& pts);

/// Read whitespace‐ or comma‐separated points from a file
std::vector<Point>
readPointsFromFile(const std::string& filename,
                   int dimensions,
                   std::vector<int>& labels,
                   const std::set<int>& skip_columns={});

// to be used by python
std::vector<Point>
readPointsFromFile(const std::string& filename,
                   int dimensions,
                   const std::set<int>& skip_columns);
/// Print usage and warnings
void printUsage(char* prog);

std::vector<double> computeNormalizedInverseStdDevWeights(const std::vector<Point>& points);
double computePercentile(std::vector<double> values, double percentile);
std::vector<double> computeNormalizedStdRangeWeights(
    const std::vector<Point>& points,
    const std::vector<double>& stds
);