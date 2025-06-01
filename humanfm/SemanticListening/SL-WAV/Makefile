CXX = g++
CXXFLAGS = -O2 -Wall -Wextra -std=c++17 -Isrc
LDFLAGS = -lfftw3 -lonnxruntime

# Source and object directories
SRCDIR = src
OBJDIR = obj

# Source files (in src directory)
SRCS = wav_processor.cpp static_buffers.cpp SL_Model.cpp InferenceWrapper.cpp ringbuffer.cpp
# Object files (in obj directory)
OBJS = $(SRCS:%.cpp=$(OBJDIR)/%.o)

TARGET = wav_processor

# Default target
all: $(TARGET)

# Create target executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compile source files to object files in obj directory
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create obj directory if it doesn't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(TARGET)

# Phony targets
.PHONY: all clean 