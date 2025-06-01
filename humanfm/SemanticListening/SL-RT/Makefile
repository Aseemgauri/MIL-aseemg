CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Isrc
INCLUDES = -I/usr/include/onnxruntime -I/usr/include/onnxruntime/core/session
LIBS = -ljack -lfftw3 -lonnxruntime -lpthread

# Source and object directories
SRCDIR = src
OBJDIR = obj

# Source files (in src directory)
SOURCES = realtime_processor.cpp audio_manager.cpp static_buffers.cpp SL_Model.cpp InferenceWrapper.cpp
# Object files (in obj directory)
OBJECTS = $(SOURCES:%.cpp=$(OBJDIR)/%.o)

TARGET = realtime_processor

# Header dependencies (in src directory)
HEADERS = $(SRCDIR)/audio_manager.h $(SRCDIR)/audio_config.h $(SRCDIR)/static_buffers.h $(SRCDIR)/unified_ringbuffer.h $(SRCDIR)/audio_data_structures.h $(SRCDIR)/SL_Model.h $(SRCDIR)/InferenceWrapper.h

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build object files in obj directory
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Create obj directory if it doesn't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(TARGET)

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y libjack-jackd2-dev libfftw3-dev libonnxruntime-dev

.PHONY: all clean install-deps 