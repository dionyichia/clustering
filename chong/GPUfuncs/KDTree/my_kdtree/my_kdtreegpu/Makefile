# Makefile for my_kdtree

# Compiler driver
HIPCC        := hipcc

# Directories
INCLUDE_DIR  := include
SRC_DIR      := src
BUILD_DIR    := build

# Flags
CXXFLAGS     := -std=c++17 -O3 -I$(INCLUDE_DIR)
CXXFLAGS += -Wno-unused-result

# Source files
CPP_SRCS     := $(wildcard $(SRC_DIR)/*.cpp)
HIP_SRCS     := $(wildcard $(SRC_DIR)/*.hip.cpp)
SRCS         := $(CPP_SRCS) $(HIP_SRCS)

# Object files
CPP_OBJS     := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
HIP_OBJS     := $(patsubst $(SRC_DIR)/%.hip.cpp,$(BUILD_DIR)/%.o,$(HIP_SRCS))
OBJS         := $(CPP_OBJS) $(HIP_OBJS)

# Final binary
TARGET       := $(BUILD_DIR)/kdtree

# Default target
all: $(TARGET)

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile host-side C++ sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

# Compile HIP kernels
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.hip.cpp | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

# Link everything
$(TARGET): $(OBJS)
	$(HIPCC) $(CXXFLAGS) $^ -o $@

# Remove generated files
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
