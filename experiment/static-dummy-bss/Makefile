CC = gcc
CFLAGS = -m64 -Og -g -Wall -Werror

SRCS = $(shell find . -name "*.c")
OBJS = $(addsuffix .o, $(basename $(SRCS)))

readelf: main
	readelf -a main | grep dummy

main: $(OBJS)

clean:
	rm -f main
	rm -f *.o

.PHONY: readelf clean

