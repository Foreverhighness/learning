CC = gcc
CFLAGS = -m64 -Og -g -Wall -Werror

all: main a.out

main: main.o libadder.so
	$(CC) -Wl,--verbose,-rpath,'$$ORIGIN' -o main main.o libadder.so

a.out: main.o libadder.so
	 ld -rpath '$$ORIGIN' -dynamic-linker /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 /usr/lib/x86_64-linux-gnu/Scrt1.o /usr/lib/x86_64-linux-gnu/crti.o main.o libadder.so -lc /usr/lib/x86_64-linux-gnu/crtn.o

main.o: main.c
	$(CC) $(CFLAGS) -c -o main.o main.c

libadder.so: adder.c
	$(CC) $(CFLAGS) -shared -o libadder.so adder.c

.PHONY: all clean
clean:
	rm -f libadder.so
	rm -f main.o
	rm -f main
	rm -f a.out
