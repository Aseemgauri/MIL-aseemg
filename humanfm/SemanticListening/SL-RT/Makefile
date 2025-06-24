CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Isrc
INCLUDES = -I/usr/include/onnxruntime -I/usr/include/onnxruntime/core/session
LIBS = -ljack -lfftw3 -lonnxruntime -lpthread -ljsoncpp

# Source and object directories
SRCDIR = src
OBJDIR = obj

# Source files for main program (in src directory)
MAIN_SOURCES = realtime_processor.cpp audio_manager.cpp static_buffers.cpp SL_Model.cpp InferenceWrapper.cpp humanfm_client.cpp
MAIN_OBJECTS = $(MAIN_SOURCES:%.cpp=$(OBJDIR)/%.o)

# Source files for runtime test (in src directory)
TEST_SOURCES = runtime_test.cpp SL_Model.cpp InferenceWrapper.cpp
TEST_OBJECTS = $(TEST_SOURCES:%.cpp=$(OBJDIR)/%.o)

# Target executables
MAIN_TARGET = realtime_processor
TEST_TARGET = runtime_test

# Header dependencies (in src directory)
HEADERS = $(SRCDIR)/audio_manager.h $(SRCDIR)/audio_config.h $(SRCDIR)/static_buffers.h $(SRCDIR)/unified_ringbuffer.h $(SRCDIR)/audio_data_structures.h $(SRCDIR)/SL_Model.h $(SRCDIR)/InferenceWrapper.h $(SRCDIR)/humanfm_client.hpp

# Default target
all: $(MAIN_TARGET) $(TEST_TARGET)

# Build the main executable
$(MAIN_TARGET): $(MAIN_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build the runtime test executable
$(TEST_TARGET): $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build object files in obj directory
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Create obj directory if it doesn't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(MAIN_TARGET) $(TEST_TARGET)

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y libjack-jackd2-dev libfftw3-dev libonnxruntime-dev

.PHONY: all clean install-deps 