#pragma once
#include <vector>
#include <memory>

// your basic Point and PI types
using Point = std::vector<double>;
using PI    = std::pair<Point,int>;

// forward‐declare DistanceMetric so other headers can see it
enum class DistanceMetric;