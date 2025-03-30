# Compiler and flags
CXX = g++
CXXFLAGS = -pthread
LDFLAGS = -lrt -lm -lasound
LIBS = libportaudio.a

# Targets
OUTPUT_NORMAL = loopback
OUTPUT_LATENT = loopback_latent

# Source files
SRCS_NORMAL = loopback.cpp ringbuffer.cpp
SRCS_LATENT = loopback_latent.cpp ringbuffer.cpp

# Default target: Compile both executables
all: $(OUTPUT_NORMAL) $(OUTPUT_LATENT)

# Build for 'normal' mode
normal: $(OUTPUT_NORMAL)

$(OUTPUT_NORMAL): $(SRCS_NORMAL:.cpp=.o)
	$(CXX) $^ $(LIBS) $(LDFLAGS) $(CXXFLAGS) -o $@

# Build for 'latent' mode
latent: $(OUTPUT_LATENT)

$(OUTPUT_LATENT): $(SRCS_LATENT:.cpp=.o)
	$(CXX) $^ $(LIBS) $(LDFLAGS) $(CXXFLAGS) -o $@

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f *.o $(OUTPUT_NORMAL) $(OUTPUT_LATENT)

# Phony targets
.PHONY: all normal latent clean
