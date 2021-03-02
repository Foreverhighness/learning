#include <assert.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cachelab.h"

int hit, miss, eviction;
bool verbose;
typedef unsigned long long Key;

typedef struct Node {
    Key key;
    struct Node* prv;
    struct Node* nxt;
} Node;
Node* nodeNew(Key v) {
    Node* node = (Node*)calloc(sizeof(Node), 1);
    if (node == NULL) {
        fprintf(stderr, "Memory is not enough for create a new node.");
        exit(EXIT_FAILURE);
    }
    // memset(node, 0, sizeof(Node));
    node->key = v;
    // node->prv = NULL;
    // node->nxt = NULL;
    return node;
}
bool nodeDelete(Node* node) {
    if (node == NULL) return false;
    node->prv = NULL;
    node->nxt = NULL;
    free(node);
    node = NULL;
    return true;
}
Node* nodeRemove(Node* node) {
    node->prv->nxt = node->nxt;
    node->nxt->prv = node->prv;
    return node;
}

typedef Node* Value;
// typedef Node* HashValue;
// typedef NodeValue HashKey;
// typedef struct {
//     int capacity;
// } HashMap;
// int hash(HashKey key) {
//     return 0;
// }
// HashValue hashMapGet(HashMap* obj, HashKey key) {
//     int key = hash(key);
// }

typedef struct {
    int size, capacity;
    Node* dummyhead;
    Node* dummytail;
} LRUCache;
// bool isEmpty(const LRUCache* obj) { return obj->size == 0; }
bool isFull(const LRUCache* obj) { return obj->size == obj->capacity; }
LRUCache* lRUCacheCreate(int capacity) {
    assert(capacity > 0);
    LRUCache* lru = (LRUCache*)calloc(sizeof(LRUCache), 1);
    if (lru == NULL) {
        fprintf(stderr, "malloc LRUCache failed.\n");
        exit(EXIT_FAILURE);
    }
    // memset(lru, 0, sizeof(LRUCache));
    lru->capacity = capacity;
    // lru->size = 0;
    lru->dummyhead = nodeNew(0);
    lru->dummytail = nodeNew(0);
    // lru->dummyhead->key = 0;
    lru->dummyhead->nxt = lru->dummytail;
    // lru->dummyhead->prv = NULL;
    // lru->dummytail->key = 0;
    // lru->dummytail->nxt = NULL;
    lru->dummytail->prv = lru->dummyhead;
    return lru;
}
// Node* mapFind(LRUCache* obj, Key key) {}
Node* lRUCacheGet(LRUCache* obj, Key key) {
    Node* cur = obj->dummyhead->nxt;
    const Node* end = obj->dummytail;
    int cnt = 0;
    while (cur != end) {
        assert(cur);
        if (cur->key == key) {
            ++hit;
            if (verbose) printf(" hit");
            return cur;
        }
        ++cnt;
        cur = cur->nxt;
    }
    assert(cnt == obj->size);
    ++miss;
    if (verbose) printf(" miss");
    return NULL;
}
void addTotail(LRUCache* obj, Node* node) {
    node->prv = obj->dummytail->prv;
    node->nxt = obj->dummytail;
    node->prv->nxt = node;
    node->nxt->prv = node;
}
void lRUCacheUpdate(LRUCache* obj, Node* node) {
    nodeRemove(node);
    addTotail(obj, node);
}
void lRUCachePut(LRUCache* obj, Key key) {
    assert(obj->size < obj->capacity);
    ++obj->size;
    addTotail(obj, nodeNew(key));
}
void lRUCacheFree(LRUCache* obj) {
    Node* cur = obj->dummyhead;
    while (cur != NULL) {
        Node* nxt = cur->nxt;
        nodeDelete(cur);
        cur = nxt;
        --obj->size;
    }
    assert(obj->size == -2);
    free(obj);
    obj = NULL;
}
void lRUCacheDestroyHead(LRUCache* obj) {
    assert(isFull(obj));
    --obj->size;
    ++eviction;
    if (verbose) printf(" eviction");
    Node* node = obj->dummyhead->nxt;
    nodeDelete(nodeRemove(node));
}

typedef struct {
    int s, E, b;
    LRUCache** lrus;
} Cache;
Cache* cacheCreate(int s, int E, int b) {
    Cache* cache = (Cache*)malloc(sizeof(Cache));
    cache->s = s;
    cache->E = E;
    cache->b = b;
    cache->lrus = (LRUCache**)malloc(sizeof(LRUCache*) << s);
    for (int i = 0; i < (1 << s); ++i) {
        cache->lrus[i] = lRUCacheCreate(E);
    }
    return cache;
}
void cacheFree(Cache* cache) {
    assert(cache);
    for (int i = 0; i < (1 << cache->s); ++i) {
        lRUCacheFree(cache->lrus[i]);
    }
    free(cache->lrus);
    cache->lrus = NULL;
    free(cache);
    cache = NULL;
}

void parse_args(int argc, char* const argv[], Cache** cache, FILE** fd) {
    // extern int optind;
    extern char* optarg;
    int opt;
    int s = 0, E = 0, b = 0;
    char* filename = NULL;
    const char* info =
        "Usage: %s [-hv] -s <num> -E <num> -b <num> -t <file>\n"
        "Options:\n"
        "  -h         Print this help message.\n"
        "  -v         Optional verbose flag.\n"
        "  -s <num>   Number of set index bits.\n"
        "  -E <num>   Number of lines per set.\n"
        "  -b <num>   Number of block offset bits.\n"
        "  -t <file>  Trace file.\n"
        ""
        "Examples:\n"
        "  linux>  %s -s 4 -E 1 -b 4 -t traces/yi.trace\n"
        "  linux>  %s -v -s 8 -E 2 -b 4 -t traces/yi.trace\n";
    // "\n"
    // "  • -h: Optional help flag that prints usage info\n"
    // "  • -v: Optional verbose flag that displays trace info\n"
    // "  • -s <s>: Number of set index bits (S = 2s is the number of
    // sets)\n" "  • -E <E>: Associativity (number of lines per set)\n" "  •
    // -b <b>: Number of block bits (B = 2b is the block size)\n" "  • -t
    // <tracefile>: Name of the valgrind trace to replay\n";
    while ((opt = getopt(argc, argv, "hvs:E:b:t:")) != -1) {
        switch (opt) {
            case 'h':
#ifdef DEBUG
                fprintf(stderr, "Get -h args\n");
#endif
                printf(info, argv[0], argv[0], argv[0]);
                exit(EXIT_SUCCESS);
                break;
            case 'v':
#ifdef DEBUG
                fprintf(stderr, "Get -v args\n");
#endif
                verbose = true;
                break;
            case 's':
                s = atoi(optarg);
#ifdef DEBUG
                fprintf(stderr, "Get -s=%s\n", optarg);
#endif
                break;
            case 'E':
#ifdef DEBUG
                fprintf(stderr, "Get -E=%s\n", optarg);
#endif
                E = atoi(optarg);
                break;
            case 'b':
#ifdef DEBUG
                fprintf(stderr, "Get -b=%s\n", optarg);
#endif
                b = atoi(optarg);
                break;
            case 't':
#ifdef DEBUG
                fprintf(stderr, "Get -t=%s\n", optarg);
#endif
                filename = optarg;
                break;
            default:
                fprintf(stderr, info, argv[0], argv[0], argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    if (s <= 0 || E <= 0 || b <= 0) {
        fprintf(stderr, "s, E, b should be positive number\n");
        exit(EXIT_FAILURE);
    }
    if ((*fd = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Couldn't open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    *cache = cacheCreate(s, E, b);
}
void process(Cache* cache, FILE* fd) {
    const unsigned int s = cache->s;
    const unsigned int b = cache->b;
    const unsigned int smask = (1 << s) - 1;
    char line[128];
    while (fgets(line, 128, fd) != NULL) {
        char operation[4];
        Key address, size;
        sscanf(line, "%s%llx,%llx", operation, &address, &size);
#ifdef DEBUG
        fprintf(stderr, "Get %s %llx,%llx\n", operation, address, size);
#endif
        if (line[0] == 'I') continue;
        char op = operation[0];
        if (verbose) printf("%c %llx,%llx", op, address, size);
        LRUCache* obj = cache->lrus[(address >> b) & smask];
        Key tag = address >> (s + b);
        Node* node = lRUCacheGet(obj, tag);
        switch (op) {
            case 'L':
            case 'S':
            case 'M':
                if (node == NULL) {
                    if (isFull(obj)) lRUCacheDestroyHead(obj);
                    lRUCachePut(obj, tag);
                } else {
                    lRUCacheUpdate(obj, node);
                }
                if (op == 'M') {
                    ++hit;
                    if (verbose) printf(" hit");
                }
                break;
            default:
                assert(false);
        }
        if (verbose) puts("");
    }
}
void clearup(Cache* cache, FILE* fd) {
    cacheFree(cache);
    assert(fd);
    fclose(fd);
}

int main(int argc, char* const argv[]) {
    Cache* cache;
    FILE* fd;
    parse_args(argc, argv, &cache, &fd);
    process(cache, fd);
    clearup(cache, fd);
    printSummary(hit, miss, eviction);
    return 0;
}
