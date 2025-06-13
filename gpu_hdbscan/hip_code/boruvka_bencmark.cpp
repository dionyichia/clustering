#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <fstream>
#include <set>
#include <queue>
#include <string>
#include <functional>


// Include your existing structs and constants
typedef unsigned long long ullong;

struct __attribute__ ((packed)) Edge
{
    uint u;
    uint v;
    uint weight;
    
    // Constructor for convenience
    Edge(uint _u = 0, uint _v = 0, uint _weight = 0) : u(_u), v(_v), weight(_weight) {}
    
    // For sorting edges by weight
    bool operator<(const Edge& other) const {
        if (weight != other.weight) return weight < other.weight;
        if (u != other.u) return u < other.u;
        return v < other.v;
    }
};

struct MST {
    char* mst;
    ullong weight;
};

// Union-Find data structure for CPU implementation
class UnionFind {
private:
    std::vector<ullong> parent;
    std::vector<int> rank;
    
public:
    UnionFind(ullong n) : parent(n), rank(n, 0) {
        for (ullong i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    ullong find(ullong x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // Path compression
        }
        return parent[x];
    }
    
    bool unite(ullong x, ullong y) {
        ullong px = find(x), py = find(y);
        if (px == py) return false;
        
        // Union by rank
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
        return true;
    }
};

// CPU Reference Implementation using Kruskal's algorithm
MST kruskal_mst_cpu(ullong n_vertices, ullong n_edges, Edge* edges) {
    MST result;
    result.weight = 0;
    result.mst = new char[n_edges]();

    // Create array of pairs (edge, original index)
    struct EdgePair {
        Edge edge;
        ullong index;
    };
    EdgePair* edge_pairs = new EdgePair[n_edges];
    for (ullong i = 0; i < n_edges; i++) {
        edge_pairs[i].edge = edges[i];
        edge_pairs[i].index = i;
    }

    // Insertion sort edge_pairs by edge.weight
    for (ullong i = 1; i < n_edges; i++) {
        EdgePair key = edge_pairs[i];
        ullong j = i;
        while (j > 0 && edge_pairs[j - 1].edge.weight > key.edge.weight) {
            edge_pairs[j] = edge_pairs[j - 1];
            j--;
        }
        edge_pairs[j] = key;
    }

    UnionFind uf(n_vertices);
    ullong edges_added = 0;

    for (ullong i = 0; i < n_edges; i++) {
        const Edge& e = edge_pairs[i].edge;
        ullong orig_idx = edge_pairs[i].index;

        if (uf.unite(e.u, e.v)) {
            result.mst[orig_idx] = 1;
            result.weight += e.weight;
            edges_added++;

            if (edges_added == n_vertices - 1) break;
        }
    }

    delete[] edge_pairs;
    return result;
}

// Alternative CPU Boruvka implementation for comparison
MST boruvka_mst_cpu(ullong n_vertices, ullong n_edges, Edge* edges) {
    MST result;
    result.weight = 0;
    result.mst = new char[n_edges]();
    
    UnionFind uf(n_vertices);
    std::vector<ullong> cheapest(n_vertices);
    
    ullong components = n_vertices;
    
    while (components > 1) {
        // Reset cheapest edges
        std::fill(cheapest.begin(), cheapest.end(), ULONG_MAX);
        
        // Find cheapest edge for each component
        for (ullong i = 0; i < n_edges; i++) {
            ullong comp_u = uf.find(edges[i].u);
            ullong comp_v = uf.find(edges[i].v);
            
            if (comp_u == comp_v) continue;
            
            // Check if this is the cheapest edge for component u
            if (cheapest[comp_u] == ULONG_MAX || 
                edges[i].weight < edges[cheapest[comp_u]].weight) {
                cheapest[comp_u] = i;
            }
            
            // Check if this is the cheapest edge for component v
            if (cheapest[comp_v] == ULONG_MAX || 
                edges[i].weight < edges[cheapest[comp_v]].weight) {
                cheapest[comp_v] = i;
            }
        }
        
        // Add cheapest edges to MST
        std::set<ullong> added_edges;
        for (ullong i = 0; i < n_vertices; i++) {
            ullong edge_idx = cheapest[i];
            if (edge_idx != ULONG_MAX && added_edges.find(edge_idx) == added_edges.end()) {
                ullong comp_u = uf.find(edges[edge_idx].u);
                ullong comp_v = uf.find(edges[edge_idx].v);
                
                if (comp_u != comp_v && uf.unite(comp_u, comp_v)) {
                    result.mst[edge_idx] = 1;
                    result.weight += edges[edge_idx].weight;
                    added_edges.insert(edge_idx);
                    components--;
                }
            }
        }
    }
    
    return result;
}

// Test data generators
class GraphGenerator {
public:
    // Generate random graph
    static std::vector<Edge> generateRandomGraph(ullong n_vertices, ullong n_edges, 
                                               uint min_weight = 1, uint max_weight = 1000,
                                               uint seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<uint> vertex_dist(0, n_vertices - 1);
        std::uniform_int_distribution<uint> weight_dist(min_weight, max_weight);
        
        std::vector<Edge> edges;
        std::set<std::pair<uint, uint>> edge_set;
        
        while (edges.size() < n_edges) {
            uint u = vertex_dist(gen);
            uint v = vertex_dist(gen);
            
            if (u == v) continue;
            if (u > v) std::swap(u, v);
            
            if (edge_set.find({u, v}) == edge_set.end()) {
                edge_set.insert({u, v});
                edges.emplace_back(u, v, weight_dist(gen));
            }
        }
        
        return edges;
    }
    
    // Generate complete graph
    static std::vector<Edge> generateCompleteGraph(ullong n_vertices, 
                                                 uint min_weight = 1, uint max_weight = 1000,
                                                 uint seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<uint> weight_dist(min_weight, max_weight);
        
        std::vector<Edge> edges;
        for (uint u = 0; u < n_vertices; u++) {
            for (uint v = u + 1; v < n_vertices; v++) {
                edges.emplace_back(u, v, weight_dist(gen));
            }
        }
        
        return edges;
    }
    
    // Generate grid graph
    static std::vector<Edge> generateGridGraph(ullong width, ullong height,
                                             uint min_weight = 1, uint max_weight = 1000,
                                             uint seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<uint> weight_dist(min_weight, max_weight);
        
        std::vector<Edge> edges;
        
        for (ullong i = 0; i < height; i++) {
            for (ullong j = 0; j < width; j++) {
                uint u = i * width + j;
                
                // Right edge
                if (j < width - 1) {
                    uint v = u + 1;
                    edges.emplace_back(u, v, weight_dist(gen));
                }
                
                // Down edge
                if (i < height - 1) {
                    uint v = u + width;
                    edges.emplace_back(u, v, weight_dist(gen));
                }
            }
        }
        
        return edges;
    }
    
    // Generate sparse graph (tree + some extra edges)
    static std::vector<Edge> generateSparseGraph(ullong n_vertices, double density = 0.1,
                                                uint min_weight = 1, uint max_weight = 1000,
                                                uint seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<uint> vertex_dist(0, n_vertices - 1);
        std::uniform_int_distribution<uint> weight_dist(min_weight, max_weight);
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        
        std::vector<Edge> edges;
        std::set<std::pair<uint, uint>> edge_set;
        
        // First, create a spanning tree to ensure connectivity
        for (uint i = 1; i < n_vertices; i++) {
            uint parent = vertex_dist(gen) % i;
            uint u = std::min(parent, i);
            uint v = std::max(parent, i);
            edge_set.insert({u, v});
            edges.emplace_back(u, v, weight_dist(gen));
        }
        
        // Add additional edges based on density
        ullong max_edges = n_vertices * (n_vertices - 1) / 2;
        ullong target_edges = std::min((ullong)(max_edges * density), max_edges);
        
        while (edges.size() < target_edges) {
            uint u = vertex_dist(gen);
            uint v = vertex_dist(gen);
            
            if (u == v) continue;
            if (u > v) std::swap(u, v);
            
            if (edge_set.find({u, v}) == edge_set.end()) {
                edge_set.insert({u, v});
                edges.emplace_back(u, v, weight_dist(gen));
            }
        }
        
        return edges;
    }
};

// Benchmark utilities
class Benchmark {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;
    
public:
    Benchmark(const std::string& benchmark_name) : name(benchmark_name) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    ~Benchmark() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
        std::cout << name << " took: " << duration / 1000.0 << " ms" << std::endl;
    }
    
    static double timeFunction(std::function<void()> func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
};

// Test verification
bool verifyMST(const std::vector<Edge>& edges, const MST& mst1, const MST& mst2, 
               ullong n_vertices, const std::string& name1, const std::string& name2) {
    std::cout << "\n=== MST Verification ===" << std::endl;
    std::cout << name1 << " weight: " << mst1.weight << std::endl;
    std::cout << name2 << " weight: " << mst2.weight << std::endl;
    
    // Check if weights match
    if (mst1.weight != mst2.weight) {
        std::cout << "ERROR: MST weights don't match!" << std::endl;
        return false;
    }
    
    // Count edges in each MST
    ullong edges1 = 0, edges2 = 0;
    for (ullong i = 0; i < edges.size(); i++) {
        if (mst1.mst[i]) edges1++;
        if (mst2.mst[i]) edges2++;
    }
    
    std::cout << name1 << " edges: " << edges1 << std::endl;
    std::cout << name2 << " edges: " << edges2 << std::endl;
    
    // MST should have exactly n-1 edges
    if (edges1 != n_vertices - 1 || edges2 != n_vertices - 1) {
        std::cout << "ERROR: MST should have " << n_vertices - 1 << " edges!" << std::endl;
        return false;
    }
    
    // Verify connectivity using Union-Find
    UnionFind uf1(n_vertices), uf2(n_vertices);
    for (ullong i = 0; i < edges.size(); i++) {
        if (mst1.mst[i]) {
            uf1.unite(edges[i].u, edges[i].v);
        }
        if (mst2.mst[i]) {
            uf2.unite(edges[i].u, edges[i].v);
        }
    }
    
    // Check if all vertices are connected
    ullong root1 = uf1.find(0), root2 = uf2.find(0);
    for (ullong i = 1; i < n_vertices; i++) {
        if (uf1.find(i) != root1 || uf2.find(i) != root2) {
            std::cout << "ERROR: MST is not connected!" << std::endl;
            return false;
        }
    }
    
    std::cout << "SUCCESS: Both MSTs are valid and have the same weight!" << std::endl;
    return true;
}

// Declare your GPU function (from your existing code)
extern MST boruvka_mst(const ullong n_vertices, const ullong n_edges, Edge* edge_list);

// Test suite
void runTestSuite() {
    std::cout << "=== Running MST Test Suite ===" << std::endl;
    
    struct TestCase {
        std::string name;
        std::function<std::vector<Edge>()> generator;
        ullong n_vertices;
    };
    
    std::vector<TestCase> test_cases = {
        {"Small Random Graph", []() { return GraphGenerator::generateRandomGraph(10, 20); }, 10},
        {"Medium Random Graph", []() { return GraphGenerator::generateRandomGraph(100, 500); }, 100},
        {"Large Random Graph", []() { return GraphGenerator::generateRandomGraph(1000, 5000); }, 1000},
        {"Complete Small Graph", []() { return GraphGenerator::generateCompleteGraph(20); }, 20},
        {"Grid Graph 10x10", []() { return GraphGenerator::generateGridGraph(10, 10); }, 100},
        {"Sparse Graph", []() { return GraphGenerator::generateSparseGraph(500, 0.05); }, 500},
    };
    
    for (const auto& test_case : test_cases) {
        std::cout << "\n--- Testing: " << test_case.name << " ---" << std::endl;
        
        auto edges = test_case.generator();
        Edge* edge_array = edges.data();
        
        // Run CPU implementations
        MST cpu_kruskal, cpu_boruvka, gpu_result;
        
        // double cpu_kruskal_time = Benchmark::timeFunction([&]() {
        //     cpu_kruskal = kruskal_mst_cpu(test_case.n_vertices, edges.size(), edge_array);
        // });
        double cpu_kruskal_time = 0;
        
        double cpu_boruvka_time = Benchmark::timeFunction([&]() {
            cpu_boruvka = boruvka_mst_cpu(test_case.n_vertices, edges.size(), edge_array);
        });
        
        // Run GPU implementation
        double gpu_time = Benchmark::timeFunction([&]() {
            gpu_result = boruvka_mst(test_case.n_vertices, edges.size(), edge_array);
        });
        
        std::cout << "CPU Kruskal time: " << cpu_kruskal_time << " ms" << std::endl;
        std::cout << "CPU Boruvka time: " << cpu_boruvka_time << " ms" << std::endl;
        std::cout << "GPU Boruvka time: " << gpu_time << " ms" << std::endl;
        
        // Verify results
        bool kruskal_match = verifyMST(edges, cpu_kruskal, cpu_boruvka, 
                                      test_case.n_vertices, "CPU Kruskal", "CPU Boruvka");
        bool boruvka_match = verifyMST(edges, cpu_boruvka, gpu_result, 
                                      test_case.n_vertices, "CPU Boruvka", "GPU Boruvka");
        
        if (kruskal_match && boruvka_match) {
            double kruskal_speedup = cpu_kruskal_time / gpu_time;
            double boruvka_speedup = cpu_boruvka_time / gpu_time;
            std::cout << "CPU Boruvka vs CPU Kruskal: " << kruskal_speedup << "x" << std::endl;
            std::cout << "Speedup vs CPU Boruvka: " << boruvka_speedup << "x" << std::endl;
        } else {
            std::cout << "TEST FAILED!" << std::endl;
        }
        
        // Cleanup
        delete[] cpu_kruskal.mst;
        delete[] cpu_boruvka.mst;
        // Note: gpu_result.mst is freed in your boruvka_mst function
    }
}

// Performance benchmark with different graph sizes
void runPerformanceBenchmark() {
    std::cout << "\n=== Performance Benchmark ===" << std::endl;
    
    std::vector<std::pair<ullong, ullong>> sizes = {
        // {100, 500},
        {500, 2500},
        {1000, 5000},
        {2000, 10000},
        {5000, 25000},
        {10000, 50000},
        {50000, 250000},
        {100000, 500000},
    };
    
    std::cout << "Vertices\tEdges\tCPU Kruskal (ms)\tCPU Boruvka (ms)\tGPU Boruvka (ms)\tSpeedup vs Kruskal\tSpeedup vs Boruvka" << std::endl;
    
    for (const auto& size : sizes) {
        ullong n_vertices = size.first;
        ullong n_edges = size.second;
        
        std::cout << "generating graph" << std::endl;
        auto edges = GraphGenerator::generateCompleteGraph(n_vertices);
        Edge* edge_array = edges.data();
        std::cout << "graph generated, num_edges: `" << edges.size() << std::endl;
        
        // Warm up GPU
        MST warmup = boruvka_mst(n_vertices, edges.size(), edge_array);
        
        // Benchmark
        std::cout << "kruskal_mst_cpu start " << std::endl;
        double cpu_kruskal_time = Benchmark::timeFunction([&]() {
            MST result = kruskal_mst_cpu(n_vertices, edges.size(), edge_array);
            delete[] result.mst;
        });
        std::cout << "kruskal_mst_cpu done " << std::endl;
        
        double cpu_boruvka_time = Benchmark::timeFunction([&]() {
            MST result = boruvka_mst_cpu(n_vertices, edges.size(), edge_array);
            delete[] result.mst;
        });
        std::cout << "boruvka_mst_cpu done " << std::endl;
        
        double gpu_time = Benchmark::timeFunction([&]() {
            MST result = boruvka_mst(n_vertices, edges.size(), edge_array);
        });
        std::cout << "boruvka_gpu_mst done " << std::endl;
        
        double kruskal_speedup = cpu_kruskal_time / gpu_time;
        double boruvka_speedup = cpu_boruvka_time / gpu_time;
        
        std::cout << n_vertices << "\t\t" << n_edges << "\t" 
                  << cpu_kruskal_time << "\t\t\t" << cpu_boruvka_time << "\t\t\t"
                  << gpu_time << "\t\t\t" << kruskal_speedup << "\t\t\t" 
                  << boruvka_speedup << std::endl;
    }
}

// Save test data to file
void saveTestData(const std::string& filename, const std::vector<Edge>& edges, ullong n_vertices) {
    std::ofstream file(filename);
    file << n_vertices << " " << edges.size() << std::endl;
    for (const auto& edge : edges) {
        file << edge.u << " " << edge.v << " " << edge.weight << std::endl;
    }
    file.close();
    std::cout << "Test data saved to " << filename << std::endl;
}

// Load test data from file
std::pair<std::vector<Edge>, ullong> loadTestData(const std::string& filename) {
    std::ifstream file(filename);
    ullong n_vertices, n_edges;
    file >> n_vertices >> n_edges;
    
    std::vector<Edge> edges;
    for (ullong i = 0; i < n_edges; i++) {
        uint u, v, weight;
        file >> u >> v >> weight;
        edges.emplace_back(u, v, weight);
    }
    
    file.close();
    return {edges, n_vertices};
}

int main() {
    // Initialize GPU
    extern void initGPUs();
    initGPUs();
    
    // Run tests
    // runTestSuite();
    
    // Run performance benchmark
    runPerformanceBenchmark();
    
    // Generate and save some test data
    auto large_graph = GraphGenerator::generateRandomGraph(50000, 250000);
    saveTestData("large_test_graph.txt", large_graph, 5000);
    
    auto complete_graph = GraphGenerator::generateCompleteGraph(100);
    saveTestData("complete_test_graph.txt", complete_graph, 100);
    
    return 0;
}