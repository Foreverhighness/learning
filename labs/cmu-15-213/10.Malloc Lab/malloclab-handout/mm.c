/*
 * mm-seglist.c
 */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"

#ifndef DRIVER
team_t team = {
    /* Team name */
    "fh",
    /* First member's full name */
    "Foreverhighness",
    /* First member's email address */
    "Foreverhighness@github.com",
    /* Second member's full name (leave blank if none) */
    "",
    /* Second member's email address (leave blank if none) */
    ""};
#endif
#ifdef DRIVER
void mm_checkheap(int i) {}
#endif

/* single word (4) or double word (8) alignment */
#define ALIGNMENT 8

/* rounds up to the nearest multiple of ALIGNMENT */
#define ALIGN(size) (((size) + (ALIGNMENT - 1)) & ~0x7)

#define SIZE_T_SIZE (ALIGN(sizeof(size_t)))

// This is the implicit free list implementation.

// clang-format off
/* Basic constants and macros */
#define WSIZE      4        /* Word and header/footer size (bytes) */
#define DSIZE      8        /* Double word size (bytes) */
#define CHUNKSIZE (1 << 12) /* Extend heap by this amount (bytes) */

#define NUM_CLASSES 20
#define PROLOGUE_SIZE (DSIZE + NUM_CLASSES * WSIZE)

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

/* Pack a size and allocated bit into a word */
#define PACK(size, alloc) ((size) | (alloc))

/* Read and write a word at address p */
#define GET(p)      (*(uint32_t*)(p))
#define PUT(p, val) (*(uint32_t*)(p) = (val))

/* Read the size and allocated fields from address p */
#define GET_SIZE(p)  (GET(p) & ~0x7)
#define GET_ALLOC(p) (GET(p) & 0x1)

/* Given block ptr bp, compute address of its header and footer */
#define HDRP(bp) ((uint8_t*)(bp) - WSIZE)
#define FTRP(bp) ((uint8_t*)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)

/* Given block ptr bp, compute address of its pred and succ */
#define SUCCP(bp) ((uint8_t*)(bp))
#define PREDP(bp) ((uint8_t*)(bp) + WSIZE)

/* Given block ptr bp, compute its class*/
#define GET_HEAD(bp, class) ((uint8_t*)(bp) + (class) * WSIZE)

/* Given block ptr bp, compute address of next and previous blocks */
#define NEXT_BLKP(bp) ((uint8_t*)(bp) + GET_SIZE(((uint8_t*)(bp) - WSIZE)))
#define PREV_BLKP(bp) ((uint8_t*)(bp) - GET_SIZE(((uint8_t*)(bp) - DSIZE)))
// clang-format on

/* Global variables */
static void *heap_listp;

/* Function prototypes for internal helper routines */
static void *extend_heap(size_t);
static void *coalesce(void *);
static void *find_fit(size_t);
static void *place(void *, size_t);

/* Function prototypes for explicit free list implementation */
static void *insert_block(void *);
static void *remove_block(void *);

/* Function prototypes for seglist implementation */
static int get_class(size_t);

#define mm_check_heap(verbose, lineno)
#ifdef DEBUG
#undef mm_check_heap
#define mm_check_heap(verbose, lineno)       \
  if (verbose) printf("in %s:\n", __func__); \
  check_heap(verbose, lineno)
#endif

/* Function prototypes for debug */
static void print_block(void *);
static void check_block(void *);
static void check_heap(bool, int);

static void check_list(void *, int);

int mm_init(void) {
  if ((heap_listp = mem_sbrk(DSIZE + PROLOGUE_SIZE)) == (void *)-1) {
    return -1;
  }

  PUT(heap_listp, 0);                                    /* Alignment padding */
  PUT(heap_listp + (1 * WSIZE), PACK(PROLOGUE_SIZE, 1)); /* Prologue header */
  // PUT(heap_listp + (2 * WSIZE), 0);                  /* Prologue successor */
  // PUT(heap_listp + (3 * WSIZE), 0);                  /* Prologue predcessor */
  for (int i = 0; i < NUM_CLASSES; ++i) {
    PUT(heap_listp + ((i + 2) * WSIZE), 0);
  }
  PUT(heap_listp + ((NUM_CLASSES + 2) * WSIZE), PACK(PROLOGUE_SIZE, 1)); /* Prologue footer */
  PUT(heap_listp + ((NUM_CLASSES + 3) * WSIZE), PACK(0, 1));             /* Epilogue header */

  heap_listp += (2 * WSIZE);

  /* Extend the empty heap with a free block of CHUNKSIZE bytes */
  if (extend_heap(CHUNKSIZE / WSIZE) == NULL) {
    return -1;
  }
  mm_check_heap(true, __LINE__);

  return 0;
}

void *mm_malloc(size_t size) {
  size_t aligned_size;
  size_t extend_size;
  uint8_t *bp = heap_listp;

  if (bp == NULL) {
    mm_init();
  }

  if (size == 0) {
    return NULL;
  }

  /* Adjust block size to include overhead and alignment requirements. */
  aligned_size = ALIGN(size + DSIZE);
  assert(aligned_size >= 16);

  /* Search the free list for a fit */
  if ((bp = find_fit(aligned_size)) != NULL) {
    return place(bp, aligned_size);
  }

  /* No fit found. Get more memory and place the block */
  extend_size = MAX(aligned_size, CHUNKSIZE);
  if ((bp = extend_heap(extend_size / WSIZE)) == NULL) {
    return NULL;
  }
  return place(bp, aligned_size);
}

void mm_free(void *ptr) {
  if (ptr == NULL) {
    return;
  }

  if (heap_listp == NULL) {
    mm_init();
  }

  size_t size = GET_SIZE(HDRP(ptr));

  PUT(HDRP(ptr), PACK(size, 0));
  PUT(FTRP(ptr), PACK(size, 0));
  coalesce(ptr);
  mm_check_heap(true, __LINE__);
}

void *mm_realloc(void *ptr, size_t size) {
  size_t old_size;
  // size_t aligned_size;
  void *new_ptr;

  /* If size == 0 then this is just free, and we return NULL. */
  if (size == 0) {
    mm_free(ptr);
    return NULL;
  }

  /* If oldptr is NULL, then this is just malloc. */
  if (ptr == NULL) {
    return mm_malloc(size);
  }

  old_size = GET_SIZE(HDRP(ptr));
  // aligned_size = ALIGN(size + DSIZE);

  // if (old_size >= aligned_size) {
  //   if (old_size - aligned_size >= 2 * DSIZE) {
  //     PUT(HDRP(ptr), PACK(aligned_size, 1));
  //     PUT(FTRP(ptr), PACK(aligned_size, 1));
  //     ptr = NEXT_BLKP(ptr);
  //     PUT(HDRP(ptr), PACK(old_size - aligned_size, 0));
  //     PUT(FTRP(ptr), PACK(old_size - aligned_size, 0));
  //     coalesce(ptr);
  //     insert_block(ptr);
  //     ptr = PREV_BLKP(ptr);
  //   }
  //   return ptr;
  // }

  /* Not enough space in existing block, so we need to allocate a new block */
  new_ptr = mm_malloc(size);

  /* If malloc fails, then we return NULL */
  if (new_ptr == NULL) {
    return NULL;
  }

  /* Copy the old data into the new block */
  memcpy(new_ptr, ptr, MIN(old_size, size));

  /* Free the old block */
  mm_free(ptr);

  return new_ptr;
}

static void *insert_block(void *bp) {
  size_t size = GET_SIZE(HDRP(bp));
  void *head = GET_HEAD(heap_listp, get_class(size));

  PUT(SUCCP(bp), GET(SUCCP(head)));
  PUT(PREDP(bp), (uint32_t)head);
  PUT(SUCCP(head), (uint32_t)bp);
  if (GET(SUCCP(bp)) != 0) {
    PUT(PREDP(GET(SUCCP(bp))), (uint32_t)bp);
  }
  return bp;
}

static void *remove_block(void *bp) {
  if (GET(SUCCP(bp)) != 0) {
    PUT(PREDP(GET(SUCCP(bp))), GET(PREDP(bp)));
  }
  if (GET(PREDP(bp)) != 0) {
    PUT(SUCCP(GET(PREDP(bp))), GET(SUCCP(bp)));
  }
  PUT(PREDP(bp), 0);
  PUT(SUCCP(bp), 0);
  return bp;
}

static void *coalesce(void *bp) {
  assert(!GET_ALLOC(HDRP(bp)));
  bool prev_alloc = GET_ALLOC(FTRP(PREV_BLKP(bp)));
  bool next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
  size_t size = GET_SIZE(HDRP(bp));
  size_t prev_size;

  if (prev_alloc && next_alloc) {
    insert_block(bp);
    return bp;
  } else if (prev_alloc && !next_alloc) {
    size += GET_SIZE(HDRP(NEXT_BLKP(bp)));
    remove_block(NEXT_BLKP(bp));
    PUT(HDRP(bp), PACK(size, 0));
    PUT(FTRP(bp), PACK(size, 0));
    insert_block(bp);
  } else if (!prev_alloc && next_alloc) {
    prev_size = GET_SIZE(HDRP(PREV_BLKP(bp)));
    size += prev_size;
    PUT(FTRP(bp), PACK(size, 0));
    PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
    bp = PREV_BLKP(bp);
    remove_block(bp);
    insert_block(bp);
  } else {
    prev_size = GET_SIZE(HDRP(PREV_BLKP(bp)));
    size += prev_size + GET_SIZE(FTRP(NEXT_BLKP(bp)));
    remove_block(NEXT_BLKP(bp));
    PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
    PUT(FTRP(NEXT_BLKP(bp)), PACK(size, 0));
    bp = PREV_BLKP(bp);
    remove_block(bp);
    insert_block(bp);
  }

  mm_check_heap(true, __LINE__);
  return bp;
}

// Returns the block pointer to a free block.
static void *extend_heap(size_t words) {
  uint8_t *bp;
  size_t size;

  /* Allocate an even number of words to maintain alignment */
  size = (words % 2) ? (words + 1) * WSIZE : words * WSIZE;
  if ((long)(bp = mem_sbrk(size)) == -1) {
    return NULL;
  }

  /* Initialize free block header/footer and the epilogue header */
  PUT(HDRP(bp), PACK(size, 0));         /* Free block header */
  PUT(FTRP(bp), PACK(size, 0));         /* Free block footer */
  PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 1)); /* New epilogue header */

  /* Coalesce if the previous block was free */
  return coalesce(bp);
}

static void *place(void *bp, size_t aligned_size) {
  size_t old_size = GET_SIZE(HDRP(bp));

  remove_block(bp);
  if (old_size - aligned_size >= 96) {
    PUT(HDRP(bp), PACK(old_size - aligned_size, 0));
    PUT(FTRP(bp), PACK(old_size - aligned_size, 0));
    insert_block(bp);
    bp = NEXT_BLKP(bp);
    PUT(HDRP(bp), PACK(aligned_size, 1));
    PUT(FTRP(bp), PACK(aligned_size, 1));
  } else if (old_size - aligned_size >= 2 * DSIZE) {
    PUT(HDRP(bp), PACK(aligned_size, 1));
    PUT(FTRP(bp), PACK(aligned_size, 1));
    bp = NEXT_BLKP(bp);
    PUT(HDRP(bp), PACK(old_size - aligned_size, 0));
    PUT(FTRP(bp), PACK(old_size - aligned_size, 0));
    insert_block(bp);
    bp = PREV_BLKP(bp);
  } else {
    PUT(HDRP(bp), PACK(old_size, 1));
    PUT(FTRP(bp), PACK(old_size, 1));
  }
  return bp;
}

static void *find_fit(size_t aligned_size) {
  /* First-fit search */
  for (int class = get_class(aligned_size); class < NUM_CLASSES; ++class) {
    void *head = GET_HEAD(heap_listp, class);
    for (void *bp = (void *)(GET(SUCCP(head))); bp != NULL; bp = (void *)(GET(SUCCP(bp)))) {
      if (GET_SIZE(HDRP(bp)) >= aligned_size) {
        return bp;
      }
    }
  }
  return NULL;  // Not fit
}

static void print_block(void *bp) {
  size_t hsize, halloc, fsize, falloc;

  hsize = GET_SIZE(HDRP(bp));
  halloc = GET_ALLOC(HDRP(bp));
  fsize = GET_SIZE(FTRP(bp));
  falloc = GET_ALLOC(FTRP(bp));

  if (hsize == 0) {
    printf("%p: EOF\n", bp);
    return;
  }

  printf("%p: header: [%zu:%c] footer: [%zu:%c]", bp, hsize, halloc ? 'a' : 'f', fsize, falloc ? 'a' : 'f');
  if (!halloc) {
    printf(" pred: %p, succ: %p", (void *)(GET(PREDP(bp))), (void *)(GET(SUCCP(bp))));
  }
  printf("\n");
}

static void check_block(void *bp) {
  if ((size_t)bp % DSIZE != 0) {
    printf("Error: %p is not doubleword aligned\n", bp);
  }
  if (GET(HDRP(bp)) != GET(FTRP(bp))) {
    printf("Error: header does not match footer\n");
  }
  if (!GET_ALLOC(HDRP(bp)) && GET(SUCCP(bp)) != 0 && bp != (void *)(GET(PREDP(GET(SUCCP(bp)))))) {
    printf("Error: self does not match succ's pred\n");
  }
  if (!GET_ALLOC(HDRP(bp)) && GET(PREDP(bp)) != 0 && bp != (void *)(GET(SUCCP(GET(PREDP(bp)))))) {
    printf("Error: self does not match pred's succ\n");
  }
}

static void print_list(void *head, int class) {
  if (head == NULL || GET(SUCCP(head)) == 0) {
    return;
  }
  printf("class%02d head:(%p) -> (%p) \n", class, head, (void *)GET(SUCCP(head)));
  while (GET(SUCCP(head)) != 0) {
    head = (void *)(GET(SUCCP(head)));
    print_block(head);
  }
  puts("");
}

static void check_heap(bool verbose, int lineno) {
  void *bp = heap_listp;
  if (verbose) {
    printf("checkheap called from %d\n", lineno);
    printf("Heap (%p):\n", bp);
  }

  if (GET(HDRP(bp)) != PACK(PROLOGUE_SIZE, 1)) {
    printf("Bad prologue header\n");
  }

  for (int class = 0; class < NUM_CLASSES; ++class) {
    if (verbose) {
      print_list(GET_HEAD(heap_listp, class), class);
    }
    check_list(GET_HEAD(heap_listp, class), class);
  }

  check_block(bp);

  for (bp = NEXT_BLKP(bp); GET_SIZE(HDRP(bp)) > 0; bp = NEXT_BLKP(bp)) {
    if (verbose) {
      print_block(bp);
    }
    check_block(bp);
  }

  if (verbose) {
    print_block(bp);
  }

  if (GET(HDRP(bp)) != PACK(0, 1)) {
    printf("Bad epilogue header\n");
  }
}

static int get_class(size_t size) {
  //                                   0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,   14,   15,  16*,
  //                                   17, 18,    19
  // static const int thresholds[] = {16, 24, 32, 36, 40, 48, 56, 64, 80, 96, 112, 128, 256, 512, 1024, 2048, 4096,
  // 8192, 16384, 32768, 65536};
  assert(size >= 16);
  if (size <= 64) {
    return (size - 16) / 8;
  }
  if (size <= 128) {
    return (size - 64) / 16 + 7;
  }
  return MIN(19, 32 - __builtin_clz(size - 1) + 4);
}

static void check_list(void *head, int class) {
  void *bp = head;
  if (head == NULL) {
    return;
  }

  /* Check if the list is circular */
  void *fast = head;
  bp = head;
  while (fast != NULL && GET(SUCCP(fast)) != 0) {
    fast = (void *)(GET(SUCCP(fast)));
    fast = (void *)(GET(SUCCP(fast)));
    bp = (void *)(GET(SUCCP(bp)));
    if (bp == fast) {
      printf("Error: circular list\n");
      return;
    }
  }

  /* Next/prev pointers in consecutive free blocks are consistent */
  for (bp = (void *)GET(SUCCP(head)); bp != NULL; bp = (void *)(GET(SUCCP(bp)))) {
    if (bp != (void *)(GET(SUCCP(GET(PREDP(bp)))))) {
      printf("Error: list is not sane\n");
    }
  }

  /* Free list contains no allocated blocks */
  for (bp = (void *)GET(SUCCP(head)); bp != NULL; bp = (void *)(GET(SUCCP(bp)))) {
    if (GET_ALLOC(HDRP(bp))) {
      printf("Error: free list contains allocated block\n");
    }
  }

  /* All free block are in the free list */
  int cnt1 = 0, cnt2 = 0;
  for (bp = (void *)GET(SUCCP(head)); bp != NULL; bp = (void *)(GET(SUCCP(bp)))) {
    ++cnt1;
  }
  for (bp = NEXT_BLKP(heap_listp); GET_SIZE(HDRP(bp)) > 0; bp = NEXT_BLKP(bp)) {
    if (!GET_ALLOC(HDRP(bp)) && get_class(GET_SIZE(HDRP(bp))) == class) {
      ++cnt2;
    }
  }
  if (cnt1 != cnt2) {
    printf("Error: There is free block not in list, free list:%d, expect:%d\n", cnt1, cnt2);
  }
}