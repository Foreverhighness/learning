.PHONY: all clean obj

all: cOg cO0 rs

cOg: main.c
	gcc -Og -g -o cOg main.c
	
cO0: main.c
	gcc -O0 -g -o cO0 main.c

rs: main.rs
	rustc -C opt-level=0 -o rs main.rs

obj: cOg cO0 rs
	objdump -t cOg cO0 rs | grep VAR | tr -s ' '

clean:
	rm -f cOg
	rm -f cO0
	rm -f rs
	rm -f *.o
