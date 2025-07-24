#include "kd_tree/include/distance.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>


double distance(
    const Point& a,
    const Point& b,
    DistanceMetric metric,
    float p,
    const std::vector<double>* weights = nullptr
) {
    assert(a.size() == b.size());
    const size_t D = a.size();

    switch (metric) {
      case DistanceMetric::EuclideanSquared: {
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i) {
          double d = a[i] - b[i];
          double w = weights ? (*weights)[i] : 1.0;
          sum += w * d * d;
        }
        return sum;
      }

      case DistanceMetric::Manhattan: {
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i) {
          double d = std::abs(a[i] - b[i]);
          double w = weights ? (*weights)[i] : 1.0;
          sum += w * d;
        }
        return sum;
      }

      case DistanceMetric::Chebyshev: {
        double mx = 0.0;
        for (size_t i = 0; i < D; ++i) {
          double d = std::abs(a[i] - b[i]);
          double w = weights ? (*weights)[i] : 1.0;
          mx = std::max(mx, w * d);
        }
        return mx;
      }

      case DistanceMetric::Minkowski: {
        if (p <= 0.0) 
          throw std::invalid_argument("p must be > 0 for Minkowski");
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i) {
          double d = std::abs(a[i] - b[i]);
          double w = weights ? (*weights)[i] : 1.0;
          sum += std::pow(w * d, p);
        }
        return std::pow(sum, 1.0 / p);
      }

      case DistanceMetric::DSO: {
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i) {
          double d = ((a[i] - b[i]) / a[i]);
          double w = weights ? (*weights)[i] : 1.0;
          sum += w * d * d;
        }
        return sum;
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
      case DistanceMetric::Minkowski:        return "Minkowski";
      case DistanceMetric::DSO:              return "DSO";
    }
    return "Unknown";
}
