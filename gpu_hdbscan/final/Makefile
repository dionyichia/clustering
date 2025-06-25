# toolchain
CXX        := hipcc
CXXFLAGS   := -std=c++17 -Iinclude -O3

# source files
SRC        := src/main.cpp src/util.cpp src/distance.cpp src/kdtree.cpp

# build directory and targets
BUILD_DIR  := build
TARGET     := my_kdtree_app
BIN        := $(BUILD_DIR)/$(TARGET)

# mirror src/*.cpp → build/*.o
OBJ        := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SRC))

# default
all: $(BIN)

# ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# link final executable
# order‐only prerequisite '$(BUILD_DIR)' makes sure the folder exists
$(BIN): $(OBJ) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# compile each .cpp → .o under build/
# again use the order‐only prereq to create the folder
$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# remove everything under build/
clean:
	rm -rf $(BUILD_DIR)
