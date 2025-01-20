# Compiler and flags
CXX = g++
CXXFLAGS = -pthread
LDFLAGS = -ljack

# Targets
TARGETS = loopback_jack loopback_jack_latent loopback_jack_filtered

# Source files
SRC_LOOPBACK = loopback_jack.cpp
SRC_LATENT = loopback_jack_latent.cpp
SRC_FILTERED = loopback_jack_filtered.cpp

# Default target: Build all executables
all: $(TARGETS)

# Build loopback_jack
loopback_jack: $(SRC_LOOPBACK)
	$(CXX) $(SRC_LOOPBACK) -o $@ $(LDFLAGS) $(CXXFLAGS)

# Build loopback_jack_latent
loopback_jack_latent: $(SRC_LATENT)
	$(CXX) $(SRC_LATENT) -o $@ $(LDFLAGS) $(CXXFLAGS)

# Build loopback_jack_filtered
loopback_jack_filtered: $(SRC_FILTERED)
	$(CXX) $(SRC_FILTERED) -o $@ $(LDFLAGS) $(CXXFLAGS)

# Clean up build files
clean:
	rm -f $(TARGETS)

# Phony targets
.PHONY: all clean
