CXX = clang++
CXXFLAGS = -std=c++11 -Wall -O2
LDFLAGS = -lOpenCL

TARGET = rl_host
SOURCES = rl_host.cc
OBJECTS = $(SOURCES:.cc=.o)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)
