CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11

all: generate_test_wav

generate_test_wav: generate_test_wav.cpp
	$(CXX) $(CXXFLAGS) -o generate_test_wav generate_test_wav.cpp

clean:
	rm -f generate_test_wav 