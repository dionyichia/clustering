#pragma once
#include "kd_tree/include/types.hpp"
#include <string>
#include <vector>

/// Normalize each dimension of pts to [0,1]
void normalizePoints(std::vector<Point>& pts);

/// Read whitespace‐ or comma‐separated points from a file
std::vector<Point> readPointsFromFile(const std::string& filename, int dimensions);

/// Print usage and warnings
void printUsage(char* prog);