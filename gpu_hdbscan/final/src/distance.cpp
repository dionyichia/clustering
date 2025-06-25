#include "distance.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>


double distance(
    const Point& a,
    const Point& b,
    DistanceMetric metric,
    float p
) {
    assert(a.size() == b.size());
    const size_t D = a.size();

    switch (metric) {
      case DistanceMetric::EuclideanSquared: {
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i) {
          double d = a[i] - b[i];
          sum += d*d;
        }
        return sum;
      }

      case DistanceMetric::Manhattan: {
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i) {
          sum += std::abs(a[i] - b[i]);
        }
        return sum;
      }

      case DistanceMetric::Chebyshev: {
        double mx = 0.0;
        for (size_t i = 0; i < D; ++i) {
          mx = std::max(mx, std::abs(a[i] - b[i]));
        }
        return mx;
      }

      case DistanceMetric::Minkowski: {
        if (p <= 0.0) 
          throw std::invalid_argument("p must be > 0 for Minkowski");
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i) {
          sum += std::pow(std::abs(a[i] - b[i]), p);
        }
        return std::pow(sum, 1.0 / p);
      }

      default:
        throw std::invalid_argument("Unknown DistanceMetric");
    }
}

const char* metricName(DistanceMetric m) {
    switch (m) {
      case DistanceMetric::EuclideanSquared: return "EuclideanSquared";
      case DistanceMetric::Manhattan:        return "Manhattan";
      case DistanceMetric::Chebyshev:        return "Chebyshev";
      case DistanceMetric::Minkowski:         return "Minkowski";
    }
    return "Unknown";
}
