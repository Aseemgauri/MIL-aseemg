CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17

all: wav_processor

wav_processor: wav_processor.cpp
	$(CXX) $(CXXFLAGS) -o wav_processor wav_processor.cpp -lfftw3

clean:
	rm -f wav_processor 