# C++ compiler (g++ or clang++)
CXX = g++

# Compiler flags: C++17, warnings, optimization, and libs/ on the include path
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -Ilibs

# cpp/solver: the command-line interface of the library (JSON output on stdout)
TARGET = solver

# cpp/check: the tests of the algorithms against brute force (make check && ./check)
CHECKER = check

# Headers on which every object file depends
HEADERS = libs/gameSolver.h libs/gameSolver.tpp libs/shapleyMC.h libs/shapleyMC.tpp \
          libs/banzhafMC.h libs/banzhafMC.tpp libs/shapleyBinary.h libs/utils.h libs/sampling.h

SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

$(CHECKER): check.o
	$(CXX) $(CXXFLAGS) -o $(CHECKER) check.o

# Object files, rebuilt when a header changes
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Remove the build products
clean:
	rm -f $(OBJS) check.o $(TARGET) $(CHECKER)

.PHONY: all clean
