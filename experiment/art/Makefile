CXX = g++
CXXFLAGS = -m64 -Og -g -Wall -Werror -std=c++11

SRCS = $(shell find . -name "*.cpp")
OBJS = $(addsuffix .o, $(basename $(SRCS)))

main: $(OBJS)

clean:
	rm -f main
	rm -f *.o
	rm -f *.ppm

.PHONY: clean

