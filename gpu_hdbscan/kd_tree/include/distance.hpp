#pragma once
#include "types.hpp"

//-------------------------------------------------------------------------
// Metric selector
//-------------------------------------------------------------------------
enum class DistanceMetric {
  EuclideanSquared,
  Manhattan,
  Chebyshev,
  Minkowski,
  DSO
};


/// Compute distance between a and b
// if metric==Minkowski, you must supply a positive “p” (e.g. 3, 4, even non-integer).
double distance(
  const Point& a,
  const Point& b,
  DistanceMetric metric,
  float p = 2.0f
);

/// Convert enum to text
const char* metricName(DistanceMetric m);
