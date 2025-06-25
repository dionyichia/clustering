# Unified Makefile for GPU HDBSCAN Project
# Compiles code from final/, hip_code/, and single_linkage/ directories

# Compiler settings
HIPCC = hipcc
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Ifinal/include
HIPFLAGS = -std=c++17 -O3 --offload-arch=gfx1030 --offload-arch=gfx1031 --offload-arch=gfx1032 -Ifinal/include

# Directories
BUILDDIR = build
FINAL_DIR = final
HIP_DIR = hip_code
SINGLE_DIR = single_linkage

# Source files from each directory
FINAL_SOURCES = $(wildcard $(FINAL_DIR)/src/*.cpp)
# FINAL_KDTREE = $(FINAL_DIR)/kdtree.cpp
HIP_SOURCES = $(wildcard $(HIP_DIR)/*.cpp)
HIP_GPU_SOURCES = $(wildcard $(HIP_DIR)/*.hip)
SINGLE_SOURCES = $(wildcard $(SINGLE_DIR)/*.cpp)

# Object files
FINAL_OBJECTS = $(patsubst $(FINAL_DIR)/src/%.cpp,$(BUILDDIR)/final_%.o,$(FINAL_SOURCES))
# FINAL_KDTREE_OBJ = $(BUILDDIR)/final_kdtree.o
HIP_OBJECTS = $(patsubst $(HIP_DIR)/%.cpp,$(BUILDDIR)/hip_%.o,$(HIP_SOURCES))
HIP_GPU_OBJECTS = $(patsubst $(HIP_DIR)/%.hip,$(BUILDDIR)/hip_gpu_%.o,$(HIP_GPU_SOURCES))
SINGLE_OBJECTS = $(patsubst $(SINGLE_DIR)/%.cpp,$(BUILDDIR)/single_%.o,$(SINGLE_SOURCES))

# Filter out main.cpp objects to avoid multiple main functions
FINAL_OBJECTS := $(filter-out $(BUILDDIR)/final_main.o,$(FINAL_OBJECTS))
SINGLE_OBJECTS := $(filter-out $(BUILDDIR)/single_main.o,$(SINGLE_OBJECTS))

# Main executable
MAIN_OBJ = $(BUILDDIR)/main.o
EXECUTABLE = $(BUILDDIR)/gpu_hdbscan

# All object files for linking
ALL_OBJECTS = $(FINAL_OBJECTS) $(HIP_OBJECTS) $(HIP_GPU_OBJECTS) $(SINGLE_OBJECTS) $(MAIN_OBJ)

# Default target
all: $(EXECUTABLE)

# Create build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Compile final/ source files
$(BUILDDIR)/final_%.o: $(FINAL_DIR)/src/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile final/kdtree.cpp
# $(BUILDDIR)/final_kdtree.o: $(FINAL_DIR)/kdtree.cpp | $(BUILDDIR)
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile hip_code/ C++ files
$(BUILDDIR)/hip_%.o: $(HIP_DIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile hip_code/ GPU files (.hip)
$(BUILDDIR)/hip_gpu_%.o: $(HIP_DIR)/%.hip | $(BUILDDIR)
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Compile single_linkage/ files
$(BUILDDIR)/single_%.o: $(SINGLE_DIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile main.cpp
$(MAIN_OBJ): main.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link everything together
$(EXECUTABLE): $(ALL_OBJECTS)
	$(HIPCC) $(HIPFLAGS) $^ -o $@

# Individual component targets
final_only: $(FINAL_OBJECTS)
	@echo "Compiled final/ components"

hip_only: $(HIP_OBJECTS) $(HIP_GPU_OBJECTS)
	@echo "Compiled hip_code/ components"

single_only: $(SINGLE_OBJECTS)
	@echo "Compiled single_linkage/ components"

# Static library targets (optional - for modular linking)
libfinal.a: $(FINAL_OBJECTS) | $(BUILDDIR)
	ar rcs $(BUILDDIR)/libfinal.a $^

libhip.a: $(HIP_OBJECTS) $(HIP_GPU_OBJECTS) | $(BUILDDIR)
	ar rcs $(BUILDDIR)/libhip.a $^

libsingle.a: $(SINGLE_OBJECTS) | $(BUILDDIR)
	ar rcs $(BUILDDIR)/libsingle.a $^

# Build static libraries
libs: libfinal.a libhip.a libsingle.a

# Alternative linking using static libraries
$(EXECUTABLE)_libs: $(MAIN_OBJ) libfinal.a libhip.a libsingle.a
	$(HIPCC) $(HIPFLAGS) $(MAIN_OBJ) -L$(BUILDDIR) -lfinal -lhip -lsingle -o $@

# Run the executable
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Debug build with debug symbols
debug: CXXFLAGS += -g -DDEBUG
debug: HIPFLAGS += -g -DDEBUG
debug: $(EXECUTABLE)

# Clean build files
clean:
	rm -rf $(BUILDDIR)

# Show build information
info:
	@echo "=== Build Configuration ==="
	@echo "Final sources: $(FINAL_SOURCES)"
	@echo "HIP sources: $(HIP_SOURCES)"
	@echo "HIP GPU sources: $(HIP_GPU_SOURCES)"
	@echo "Single linkage sources: $(SINGLE_SOURCES)"
	@echo "Final objects: $(FINAL_OBJECTS)"
	@echo "HIP objects: $(HIP_OBJECTS) $(HIP_GPU_OBJECTS)"
	@echo "Single objects: $(SINGLE_OBJECTS)"
	@echo "All objects: $(ALL_OBJECTS)"

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build the unified executable"
	@echo "  final_only   - Compile only final/ components"
	@echo "  hip_only     - Compile only hip_code/ components"
	@echo "  single_only  - Compile only single_linkage/ components"
	@echo "  libs         - Build static libraries for each component"
	@echo "  run          - Build and run the executable"
	@echo "  debug        - Build with debug symbols"
	@echo "  clean        - Clean build files"
	@echo "  info         - Show build configuration"
	@echo "  help         - Show this help message"

.PHONY: all final_only hip_only single_only libs run debug clean info help