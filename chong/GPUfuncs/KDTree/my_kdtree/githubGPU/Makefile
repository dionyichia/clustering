################################################################################
# HIP-based Makefile (converted from NVIDIA CUDA original)
# Files are now in the root directory (not src/) and use HIP instead of CUDA
################################################################################

# HIP Compiler settings
HIPCC := hipcc
GCC   ?= g++

# Detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr "[:lower:]" "[:upper:]")
OSLOWER = $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
OS_SIZE = $(shell uname -m | sed -e "s/x86_64/64/" -e "s/armv7l/32/" -e "s/aarch64/64/")
OS_ARCH = $(shell uname -m)

# HIP flags (equivalent to the original NVCC flags)
HIPCCFLAGS   := -m${OS_SIZE}
CCFLAGS      :=
LDFLAGS      :=

# Extra user flags
EXTRA_HIPCCFLAGS  ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?=

# Debug build flags
ifeq ($(dbg),1)
    HIPCCFLAGS += -g -O0
    TARGET := debug
else
    HIPCCFLAGS += -O3
    TARGET := release
endif

# Combine all flags
ALL_CCFLAGS :=
ALL_CCFLAGS += $(HIPCCFLAGS)
ALL_CCFLAGS += $(EXTRA_HIPCCFLAGS)
ALL_CCFLAGS += $(CCFLAGS)
ALL_CCFLAGS += $(EXTRA_CCFLAGS)
ALL_CCFLAGS += -fopenmp

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(LDFLAGS)
ALL_LDFLAGS += $(EXTRA_LDFLAGS)

# Common includes and libraries
INCLUDES  := -I.
LIBRARIES := -lgomp

# Enable sample (equivalent to original check)
SAMPLE_ENABLED := 1

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

# Target rules
all: build

build: kdTreeGPUsms

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

# Create obj directory if it doesn't exist
obj:
	mkdir -p obj

# Individual object file rules (files now in root directory, not src/)
obj/buildKdTree.o: buildKdTree.cu buildKdTree_common.h | obj
	$(EXEC) $(HIPCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<

obj/Gpu.o: Gpu.cu buildKdTree_common.h Gpu.h mergeSort_common.h KdNode.h removeDups_common.h | obj
	$(EXEC) $(HIPCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<

obj/KdTreeGPUsms.o: KdTreeGPUsms.cu buildKdTree_common.h Gpu.h mergeSort_common.h KdNode.h removeDups_common.h | obj
	$(EXEC) $(HIPCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<

obj/mergeSort.o: mergeSort.cu mergeSort_common.h KdNode.h | obj
	$(EXEC) $(HIPCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<

obj/removeDups.o: removeDups.cu removeDups_common.h | obj
	$(EXEC) $(HIPCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<

# Link the final executable
kdTreeGPUsms: obj/removeDups.o obj/KdTreeGPUsms.o obj/Gpu.o obj/mergeSort.o obj/buildKdTree.o
	$(EXEC) $(HIPCC) $(ALL_LDFLAGS) -o $@ $^ $(LIBRARIES)

# Run target with default parameters (similar to your first Makefile)
run: build
	$(EXEC) ./kdTreeGPUsms 

# Clean build artifacts
clean:
	rm -f obj/removeDups.o obj/KdTreeGPUsms.o obj/Gpu.o obj/mergeSort.o obj/buildKdTree.o kdTreeGPUsms
	rmdir obj 2>/dev/null || true

clobber: clean

# Additional utility targets
.PHONY: all build check.deps run clean clobber help list-sources

# List source files
list-sources:
	@echo "Source files:"
	@echo "  buildKdTree.cu"
	@echo "  Gpu.cu" 
	@echo "  KdTreeGPUsms.cu"
	@echo "  mergeSort.cu"
	@echo "  removeDups.cu"
	@echo ""
	@echo "Header files:"
	@echo "  buildKdTree_common.h"
	@echo "  Gpu.h"
	@echo "  mergeSort_common.h"
	@echo "  KdNode.h"
	@echo "  removeDups_common.h"

# Help target
help:
	@echo "Available targets:"
	@echo "  all/build     - Build the kdTreeGPUsms executable"
	@echo "  run           - Build and run with default parameters"
	@echo "  check.deps    - Check if dependencies are met"
	@echo "  list-sources  - List expected source and header files"
	@echo "  clean         - Remove object files and executable"
	@echo "  clobber       - Same as clean"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Build options:"
	@echo "  dbg=1         - Build debug version"
	@echo ""
	@echo "Example usage:"
	@echo "  make                    # Build release version"
	@echo "  make dbg=1              # Build debug version"
	@echo "  make run                # Build and run"
	@echo "  make clean && make      # Clean build"