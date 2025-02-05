CC      = /usr/bin/g++
CFLAGS  = -Wall -Wextra -std=c++20 -ffast-math -march=native
OPTFLAGS = -O3
LDFLAGS = -lm -pthread -lrt -ljack -lz -lonnxruntime -lzita-resampler -lpthread

OBJDIR = obj
SOURCEDIR = src
HEADERDIR = src

SOURCES = $(wildcard $(SOURCEDIR)/*.cpp)
OBJECTS = $(patsubst $(SOURCEDIR)/%.cpp, $(OBJDIR)/%.o, $(SOURCES))

$(OBJDIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

BINARY = SpeechField

.PHONY: all debug clean

all: $(BINARY)

debug:
	$(eval DFLAGS = -g)
	$(eval OPTFLAGS = -O0)
	make all

$(info    DEPS is $(OBJDIR)/$(OBJECTS))
$(info $(SOURCES))

$(BINARY): $(OBJDIR)/$(OBJECTS)
	$(CC) $(CFLAGS) $(OPTFLAGS) $(DFLAGS) -I$(HEADERDIR) -I$(SOURCEDIR) \
	$(OBJECTS) $(LDFLAGS) -o $@

$(OBJDIR)/%.o: $(SOURCEDIR)/%.cpp $(HEADERDIR)/%.h
	$(CC) $(CFLAGS) $(OPTFLAGS) $(DFLAGS) -I$(HEADERDIR) -I$(SOURCEDIR) -c $< -o $@

clean:
	rm -f $(OBJDIR)/*.o $(BINARY)
