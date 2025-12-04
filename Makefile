# Compiler
NVCC = nvcc

# Flags
NVCCFLAGS = -O3 -std=c++17

# Source layout
SRC_DIR = src
SRCS = $(wildcard $(SRC_DIR)/*.cu)
MAIN = main.cu

# Object files
OBJS = $(SRCS:.cu=.o)
MAIN_OBJ = $(MAIN:.cu=.o)

# Output binary
TARGET = app

# Default rule
all: $(TARGET)

# Build final binary
$(TARGET): $(OBJS) $(MAIN_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compile .cu → .o in src/
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile main.cu → main.o
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Cleanup
clean:
	rm -f $(TARGET) $(OBJS) $(MAIN_OBJ)

.PHONY: all clean
