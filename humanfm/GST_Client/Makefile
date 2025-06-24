# Compiler
CXX = g++

# GStreamer flags via pkg-config
GST_CFLAGS := $(shell pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0)
GST_LIBS   := $(shell pkg-config --libs gstreamer-1.0 gstreamer-app-1.0)

# JACK flags via pkg-config
JACK_CFLAGS := $(shell pkg-config --cflags jack)
JACK_LIBS   := $(shell pkg-config --libs jack)

# Compilation flags (Add -g for debugging)
CXXFLAGS = -std=c++17 -g -Wall $(GST_CFLAGS) $(JACK_CFLAGS)

# Target executable
TARGET = chunks_audio

# Source files
SRCS = chunks.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Build the program.
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(GST_LIBS) $(JACK_LIBS)

# Compile .cpp files into .o files.
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files.
clean:
	rm -f $(OBJS) $(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all (default) - Build chunks_audio program"
	@echo "  clean         - Remove built files"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Usage: make"
	@echo "Output: ./chunks_audio"

.PHONY: clean help
