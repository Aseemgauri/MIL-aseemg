CXX = g++
CXXFLAGS = -O2 -Wall -Wextra
LDFLAGS = -lfftw3

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