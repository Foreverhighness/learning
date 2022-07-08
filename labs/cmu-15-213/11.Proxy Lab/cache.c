#include "cache.h"

cache_t *cache;

static bool cache_element_vaild(cache_element_t *self);
static void cache_element_realloc(cache_element_t *self);

cache_element_t *cache_element_new() {
  cache_element_t *element = Malloc(sizeof(cache_element_t));
  memset(element, 0, sizeof(cache_element_t));
  element->buf = Malloc(MAX_OBJECT_SIZE);
  return element;
}

void cache_element_free(void *ptr) {
  cache_element_t *self = ptr;
  Free(self->buf);
  Free(self);
}

void cache_element_update(cache_element_t *self, char *usrbuf, ssize_t len) {
  if (!cache_element_vaild(self)) return;
  if (self->len + len <= MAX_OBJECT_SIZE) {
    memcpy(self->buf + self->len, usrbuf, len);
  }
  self->len += len;
}

static bool cache_element_vaild(cache_element_t *self) { return self->len <= MAX_OBJECT_SIZE; }

static void cache_element_realloc(cache_element_t *self) {
  if (!cache_element_vaild(self)) return;
  self->buf = Realloc(self->buf, self->len);
}
struct Node {
#ifdef DEBUG
  char *key1;
#endif
  char *key2;
  cache_element_t *value;
  node_t *next;
};

static node_t *node_new(const char *key2, cache_element_t *value) {
  node_t *self = Malloc(sizeof(node_t));
  self->key2 = strdup(key2);
  self->value = value;
  self->next = NULL;
  return self;
}
static void node_free(node_t *self) {
#ifdef DEBUG
  printf("free up node (%s, %s), size: %ld\n", self->key1, self->key2, self->value->len);
#endif
  self->next = NULL;
  Free(self->key2);
  self->key2 = NULL;
  cache_element_free(self->value);
  self->value = NULL;
  Free(self);
}

static cache_element_t *node_cloned_elem(node_t *node) {
  if (node == NULL) return NULL;
  cache_element_t *other = node->value, *self = Malloc(sizeof(cache_element_t *));
  self->len = other->len;
  self->buf = Malloc(other->len);
  memcpy(self->buf, other->buf, self->len);
  return self;
}

struct List {
  char *key1;
  node_t *node;
  list_t *prev, *next;
  ssize_t size;
};

static list_t *list_new(const char *key1, node_t *node) {
  list_t *self = Malloc(sizeof(list_t));
  self->key1 = strdup(key1);
  self->node = node;
  self->prev = self->next = NULL;
  self->size = node->value->len;
  return self;
}
static void list_free(list_t *self) {
#ifdef DEBUG
  printf("free up list (%s), size: %ld\n", self->key1, self->size);
#endif
  self->prev = self->next = NULL;
  Free(self->key1);
  self->key1 = NULL;
  node_t *nxt = self->node, *cur;
  while (nxt != NULL) {
    cur = nxt;
    nxt = nxt->next;
    node_free(cur);
  }
  self->node = NULL;
}
static void list_insert(list_t *self, node_t *node) {
  self->size += node->value->len;
  node->next = self->node;
  self->node = node;
}
static void list_free_head(list_t *list) {
  node_t *node = list->node;
  list->size -= node->value->len;
  list->node = node->next;
  node_free(node);
}

static node_t *list_find_node(list_t *self, const char *key2) {
  if (self == NULL) return NULL;
  for (node_t *p = self->node; p != NULL; p = p->next) {
    if (strcasecmp(key2, p->key2) == 0) {
      return p;
    }
  }
  return NULL;
}

static void lock() { P(&(cache->lock)); }
static void unlock() { V(&(cache->lock)); }

void cache_init() {
  cache = Malloc(sizeof(cache_t));
  memset(cache, 0, sizeof(cache_t));

  Sem_init(&(cache->lock), 0, 1);

  cache->head = Malloc(sizeof(list_t));
  memset(cache->head, 0, sizeof(list_t));
  cache->tail = Malloc(sizeof(list_t));
  memset(cache->tail, 0, sizeof(list_t));

  cache->head->next = cache->tail;
  cache->tail->prev = cache->head;
}

static void cache_remove(list_t *list) {
  cache->size -= list->size;
  list->prev->next = list->next;
  list->next->prev = list->prev;
  list->prev = NULL;
  list->next = NULL;
}
static void cache_add2tail(list_t *list) {
  cache->size += list->size;
  list->prev = cache->tail->prev;
  list->next = cache->tail;
  list->prev->next = list;
  list->next->prev = list;
}
static void cache_move2tail(list_t *list) {
  if (list == NULL) return;
  cache_remove(list);
  cache_add2tail(list);
}
static void cache_free_head() {
  list_t *list = cache->head->next;
  cache->size -= list->size;
  cache->head->next = list->next;
  list->next->prev = cache->head;
  list_free(list);
}

static list_t *cache_find_list(const char *key1) {
#ifdef DEBUG
  assert(cache != NULL);
#endif
  for (list_t *p = cache->head->next; p != cache->tail; p = p->next) {
    if (strcasecmp(key1, p->key1) == 0) {
      return p;
    }
  }
  return NULL;
}

/* cache get */
cache_element_t *cache_get(const char *key1, const char *key2) {
  lock();
  list_t *list = cache_find_list(key1);
  node_t *node = list_find_node(list, key2);
  cache_move2tail(list);
  unlock();
  return node_cloned_elem(node);
}

void cache_put(const char *key1, const char *key2, cache_element_t *value) {
  if (!cache_element_vaild(value)) {
    cache_element_free(value);
    return;
  }
  cache_element_realloc(value);

  list_t *list;
  node_t *node = node_new(key2, value);
  lock();
  if ((list = cache_find_list(key1)) == NULL) {
    list = list_new(key1, node);
    cache_add2tail(list);
  } else {
#ifdef DEBUG
    assert(list_find_node(list, key2) == NULL);
#endif
    while (list->size + node->value->len > MAX_HOST_SIZE) {
      list_free_head(list); /* break LRU rule */
    }
    list_insert(list, node);
  }
  while (cache->size > MAX_CACHE_SIZE) {
    cache_free_head();
  }
#ifdef DEBUG
  node->key1 = list->key1;
#endif
  unlock();
}