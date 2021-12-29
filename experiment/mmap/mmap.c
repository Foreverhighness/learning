#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <linux/mman.h>

#define KiB * 1024LL
#define MiB * 1024LL * 1024LL
#define GiB * 1024LL * 1024LL * 1024LL

#define handle_error(msg) \
  do { perror(msg); exit(EXIT_FAILURE); } while (0)

void wait_input() {
  puts("wait for input...");
  if ('q' == getchar()) {
    exit(0);
  }
}

int main() {
  size_t len = 32 GiB;
  printf("Hello world, %lu!\n", len);
  char *addr;

  int prot = PROT_READ | PROT_WRITE;
  int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB;// | MAP_LOCKED;

  wait_input();

  addr = mmap(NULL, len, prot, flags, -1, 0);

  if (addr == MAP_FAILED) handle_error("mmap");
  // if (mlock(addr, len) == -1) handle_error("mlock");

  for (size_t i = 0; i < len; i += 8 KiB) {
    if (i % (1 GiB) == 0) wait_input();
    addr[i] = 1;
  }

  munmap(addr, len);
  return 0;
}

