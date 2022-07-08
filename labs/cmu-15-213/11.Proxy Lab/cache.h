#include <stdbool.h>

#include "csapp.h"

#ifdef DEBUG
#include <assert.h>
#endif

/* Recommended max cache and object sizes */
#define MAX_CACHE_SIZE 1049000
#define MAX_OBJECT_SIZE 102400
#define MAX_HOST_SIZE 307200

/* cache element definition */
typedef struct {
  char *buf;
  ssize_t len;
} cache_element_t;

/* alloc free on cache_element_t */
cache_element_t *cache_element_new();
void cache_element_free(void *self);

/* operation on cache_element_t */
void cache_element_update(cache_element_t *self, char *usrbuf, ssize_t len);

typedef struct Node node_t;
typedef struct List list_t;

/* cache definition */
typedef struct {
  sem_t lock;
  ssize_t size;
  list_t *head, *tail;
} cache_t;

/* initialize global cache */
void cache_init();

/* operation on cache_t */
cache_element_t *cache_get(const char *key1, const char *key2);
void cache_put(const char *key1, const char *key2, cache_element_t *value);

/* global cache */
extern cache_t *cache;
