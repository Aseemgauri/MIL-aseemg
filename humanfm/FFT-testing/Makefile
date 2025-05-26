CXX = g++
CXXFLAGS = -O2 -Wall -Wextra -std=c++17
LDFLAGS = -lfftw3 -lonnxruntime

SRCS = wav_processor.cpp static_buffers.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = wav_processor

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) 