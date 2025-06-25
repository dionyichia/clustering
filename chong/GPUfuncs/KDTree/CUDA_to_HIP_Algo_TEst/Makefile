################################################################################
# HIP-based Makefile with build/ sandbox directory
################################################################################

# toolchain
HIPCC      ?= hipcc
CXX        := $(HIPCC)

# flags
CXXSTD     := -std=c++14
OMP_FLAGS  := -fopenmp

# add this:
INCLUDES   := -Iinclude

CXXFLAGS   := $(CXXSTD) $(OMP_FLAGS) $(INCLUDES)
LDFLAGS    := $(OMP_FLAGS)

# directories
SRC_DIR    := src
BUILD_DIR  := build
OBJ_DIR    := $(BUILD_DIR)/obj

# sources and objects
SRCS       := buildKdTree.cu \
              Gpu.cu \
              KdTreeGPUsms.cu \
              mergeSort.cu \
              removeDups.cu

OBJS       := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(SRCS))

# final binary
BIN        := $(BUILD_DIR)/kdTreeGPUsms

.PHONY: all clean run

all: $(BIN)

# link step
$(BIN): $(OBJS) | $(BUILD_DIR)
	@echo "[LD] $@"
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# compile each .cu → .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@echo "[CC] $<"
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# ensure directories exist
$(BUILD_DIR) $(OBJ_DIR):
	@mkdir -p $@

# run
run: all
	@$(BIN)

# clean everything under build/
clean:
	@rm -rf $(BUILD_DIR)
