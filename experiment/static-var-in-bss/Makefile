.PHONY: all clean obj

obj: all
	objdump -t cOg cO0 rs | grep VAR | tr -s ' '

all: cOg cO0 rs

cOg: main.c
	gcc -Og -g -o cOg main.c

cO0: main.c
	gcc -O0 -g -o cO0 main.c

rs: main.rs
	rustc -C opt-level=0 -o rs main.rs

clean:
	rm -f cOg
	rm -f cO0
	rm -f rs
	rm -f *.o
