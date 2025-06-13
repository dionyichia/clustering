# Makefile for Boruvka MST Testing and Benchmarking

# Compiler settings
HIPCC = hipcc
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
HIPFLAGS = -std=c++17 -O3 --offload-arch=gfx1030 --offload-arch=gfx1031 --offload-arch=gfx1032

# Directories
SRCDIR = .
BUILDDIR = build
TESTDIR = tests

# Files
GPU_SRC = _hdbscan_boruvka_scratch_gpu.hip
TEST_SRC = boruvka_test.cpp
MAIN_SRC = main.cpp

# Targets
GPU_OBJ = $(BUILDDIR)/boruvka_gpu.o
TEST_OBJ = $(BUILDDIR)/boruvka_test.o
MAIN_OBJ = $(BUILDDIR)/main.o

EXECUTABLE = $(BUILDDIR)/boruvka_test
BENCHMARK_EXEC = $(BUILDDIR)/boruvka_benchmark

# Default target
all: $(EXECUTABLE)

# Create build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)
	mkdir -p $(TESTDIR)

# Compile GPU code
$(GPU_OBJ): $(GPU_SRC) | $(BUILDDIR)
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Compile test code
$(TEST_OBJ): $(TEST_SRC) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile main
$(MAIN_OBJ): $(MAIN_SRC) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link executable
$(EXECUTABLE): $(GPU_OBJ) $(TEST_OBJ) $(MAIN_OBJ)
	$(HIPCC) $(HIPFLAGS) $^ -o $@

# Separate your GPU code into a .hip file
gpu_only: $(GPU_OBJ)

# Run tests
test: $(EXECUTABLE)
	./$(EXECUTABLE)

# Run only correctness tests (faster)
test_correctness: $(EXECUTABLE)
	./$(EXECUTABLE) --correctness-only

# Run only performance benchmarks
benchmark: $(EXECUTABLE)
	./$(EXECUTABLE) --benchmark-only

# Generate test data
generate_data: $(EXECUTABLE)
	./$(EXECUTABLE) --generate-data

# Clean build files
clean:
	rm -rf $(BUILDDIR)
	rm -rf $(TESTDIR)/*.txt

# Install dependencies (Ubuntu/Debian)
install_deps:
	sudo apt update
	sudo apt install -y rocm-dev hip-dev

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build the test executable"
	@echo "  test         - Run all tests and benchmarks"
	@echo "  test_correctness - Run only correctness tests"
	@echo "  benchmark    - Run only performance benchmarks"
	@echo "  generate_data - Generate test data files"
	@echo "  clean        - Clean build files"
	@echo "  install_deps - Install ROCm dependencies"
	@echo "  help         - Show this help message"

.PHONY: all test test_correctness benchmark generate_data clean install_deps help