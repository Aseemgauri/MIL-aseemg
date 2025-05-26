CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
INCLUDES = -I/usr/include/onnxruntime -I/usr/include/onnxruntime/core/session
LIBS = -ljack -lfftw3 -lonnxruntime -lpthread

# Source files
SOURCES = realtime_processor.cpp audio_manager.cpp static_buffers.cpp ringbuffer.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = realtime_processor

# Header dependencies
HEADERS = audio_manager.h audio_ringbuffer.h audio_config.h static_buffers.h ringbuffer.h

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y libjack-jackd2-dev libfftw3-dev libonnxruntime-dev

.PHONY: all clean install-deps 