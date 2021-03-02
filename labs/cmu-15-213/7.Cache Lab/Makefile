#
# Student makefile for Cache Lab
# Note: requires a 64-bit x86-64 system
#
CC = gcc
CFLAGS = -g -Wall -Werror -std=c99 -m64 #-fsanitize=address -fno-omit-frame-pointer
RUSTC = rustc
RUSTFLAGS = -g

all: csim test-trans tracegen
	# Generate a handin tar file each time you compile
	-tar -cvf ${USER}-handin.tar  csim.c trans.c

csim: csim.c cachelab.c cachelab.h
	$(CC) $(CFLAGS) -o csim csim.c cachelab.c -lm

test-trans: test-trans.c trans.o cachelab.c cachelab.h
	$(CC) $(CFLAGS) -o test-trans test-trans.c cachelab.c trans.o

tracegen: tracegen.c trans.o cachelab.c
	$(CC) $(CFLAGS) -O0 -o tracegen tracegen.c trans.o cachelab.c

trans.o: trans.c
	$(CC) $(CFLAGS) -O0 -c trans.c

csim-rs: csim.rs libcachelab.a
	$(CC) $(CFLAGS) -c cachelab.c -lm
	$(RUSTC) $(RUSTFLAGS) csim.rs -L.

trans-rs.o: trans.rs
	$(RUSTC) $(RUSTFLAGS) --emit=obj trans.rs

libcachelab.a: cachelab.c cachelab.h
	$(CC) $(CFLAGS) cachelab.c -lm
	ar rcs libcachelab.a cachelab.o
#
# Clean the src dirctory
#
clean:
	rm -rf *.o
	rm -f *.tar
	rm -f csim
	rm -f test-trans tracegen
	rm -f trace.all trace.f*
	rm -f .csim_results .marker
	rm -f *.a
