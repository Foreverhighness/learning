CC = gcc
CFLAGS = -g -Wall -Werror -std=c99 -m64
mmap: mmap.c
	$(CC) $(CFLAGS) -Og -o mmap mmap.c
.PHONY: clean
clean:
	rm -f mmap
