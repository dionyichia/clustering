#pragma once
#include <vector>
#include <memory>


struct __attribute__ ((packed)) Edge{
    uint u;
    uint v;
    float weight;
    
    __host__ __device__
    Edge(uint _u = 0, uint _v = 0, float _weight = 0)
        : u(_u), v(_v), weight(_weight) {}

    __host__ __device__
    bool operator<(const Edge& other) const {
        if (weight != other.weight)  return weight < other.weight;
        if (u      != other.u)       return u      < other.u;
        return v      < other.v;
    }
};

// your basic Point and PI types
using Point = std::vector<double>;
using PI    = std::pair<Point,int>;

// forward‐declare DistanceMetric so other headers can see it
enum class DistanceMetric;