NVCC      := nvcc
NVCCFLAGS := -arch=native -O3 -g -Werror=all-warnings -std=c++17

SRCS = $(shell find . -name "*.cu")
OBJS = $(addsuffix .o, $(basename $(SRCS)))

run: main
	./main

prof: main
	nsys nvprof $^

main: main.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

query:
	nvidia-smi --query-gpu=compute_cap --format=csv,noheader

clean:
	rm -f main
	rm -f *.o
	rm -f *.nsys-rep
	rm -f *.sqlite

.PHONY: run prof query clean
