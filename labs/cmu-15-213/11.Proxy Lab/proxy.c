
#include "cache.h"

/* len of "http://" */
#define SCHEMELEN 7

/* You won't lose style points for including this long line in your code */
static const char *user_agent_hdr =
    "User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:10.0.3) Gecko/20120305 "
    "Firefox/10.0.3\r\n";
static const char *conn_hdr = "Connection: close\r\n";
static const char *porxy_conn_hdr = "Proxy-Connection: close\r\n";
static char *http_port = "80";

/* print error helper function*/
static void print_error(char *cause, int lineno, char *errstr);
static void clienterror(int fd, char *cause, char *errnum, char *shortmsg, char *longmsg);

/* robust error handler */
static void try_writen(int fd, char *buf, ssize_t len, const char *format, ...);
static void Rclose(int fd);
static ssize_t Rrio_readlineb(rio_t *rp, char *buf, size_t maxlen);

/* main routine */
static void handle_proxy(int clientfd);

/* helper fucntion */
static void parse_requestline(int clientfd, char *buf, char *hostname, char **port, char *filename);
static int open_serverfd(int clientfd, char *hostname, char *filename);
static void process_headers(int serverfd, rio_t *rp, char *buf, char *host);
static void check_send(const int serverfd, char *hdr, bool *host);

/* SIGPIPE handler */
static void noop(int sig) {}

/* Simple warraper */
static void cleanup(void *arg) { Rclose(*(int *)arg); }
static void *doit(void *arg) {
  int fd = *(int *)arg;
  free(arg);
  handle_proxy(fd);
  return NULL;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("usage: %s <port>\n", argv[0]);
    return 0;
  }

  int listenfd = Open_listenfd(argv[1]);

  Signal(SIGPIPE, noop);
  pthread_t tid;

  while (1) {
    int connfd;
    struct sockaddr_storage clientaddr;
    socklen_t clientlen = sizeof(struct sockaddr_storage);
    if ((connfd = accept(listenfd, (SA *)&clientaddr, &clientlen)) < 0) {
      print_error("accept", __LINE__, strerror(connfd));
      continue;
    }

#ifdef DEBUG
    char hostname[MAXLINE], port[MAXLINE];
    Getnameinfo((SA *)&clientaddr, clientlen, hostname, MAXLINE, port, MAXLINE, 0);
    printf("[%d] Accepted connection from (%s, %s)\n", connfd, hostname, port);
#endif

    int *arg = malloc(sizeof(int));
    *arg = connfd;
    Pthread_create(&tid, NULL, &doit, (void *)arg);
  }
  return 0;
}

static void print_error(char *cause, int lineno, char *errstr) {
  fprintf(stderr, "[%d] %s: %s\n", lineno, cause, errstr);
}

static void clienterror(int clientfd, char *cause, char *errnum, char *shortmsg, char *longmsg) {
  char buf[MAXLINE];

  /* Print the HTTP response headers */
  try_writen(clientfd, buf, 0, "HTTP/1.0 %s %s\r\n", errnum, shortmsg);
  try_writen(clientfd, buf, 0, "Content-type: text/html\r\n\r\n");

  /* Print the HTTP response body */
  try_writen(clientfd, buf, 0, "<html><title>Proxy Error</title>");
  try_writen(clientfd, buf, 0, "<body bgcolor=\"ffffff\">\r\n");
  try_writen(clientfd, buf, 0, "%s: %s\r\n", errnum, shortmsg);
  try_writen(clientfd, buf, 0, "<p>%s: %s\r\n", longmsg, cause);
  try_writen(clientfd, buf, 0, "<hr><em>The Proxy server</em>\r\n");
  Pthread_exit(NULL);
}

static void try_writen(int fd, char *buf, ssize_t len, const char *format, ...) {
  if (len == 0) {
    va_list ap;
    va_start(ap, format);
    vsprintf(buf, format, ap);
    va_end(ap);
    len = strlen(buf);
#ifdef DEBUG
    printf("<w:%d> %s", fd, buf);
  } else {
    assert(format == NULL);
#endif
  }

  ssize_t n;
  if ((n = rio_writen(fd, buf, len)) != len) {
    if (n >= 0) {
      sprintf(buf, "try writen error, get: (%ld), expect: (%ld)\n", n, len);
      print_error("rio_writen", __LINE__, buf);
    } else {
      print_error("rio_writen", __LINE__, strerror(errno));
    }
    Pthread_exit(NULL);
  }
}

static void Rclose(const int fd) {
  int retcode;
  if ((retcode = close(fd)) < 0) {
    print_error("close", __LINE__, strerror(retcode));
  }
#ifdef DEBUG
  printf("[%d] Closed\n", fd);
#endif
}

static ssize_t Rrio_readlineb(rio_t *rp, char *buf, size_t maxlen) {
  ssize_t n;
  if ((n = rio_readlineb(rp, buf, maxlen)) <= 0) {
    if (n == 0) {
      print_error("rio_readlineb", __LINE__, "three is no data comming in");
    } else {
      print_error("rio_readlineb", __LINE__, strerror(errno));
    }
    Pthread_exit(NULL);
  }
#ifdef DEBUG
  printf("<r:%d> %s", rp->rio_fd, buf);
#endif
  return n;
}

static void parse_requestline(int clientfd, char *buf, char *hostname, char **port, char *filename) {
  char method[MAXLINE], uri[MAXLINE], version[MAXLINE];
  sscanf(buf, "%s %s %s", method, uri, version);

  /* method not match */
  if (strcasecmp("GET", method) != 0) {
    clienterror(clientfd, method, "501", "Not Implemented", "Proxy does not implement this method");
    return;
  }

  /* (RFC1945 5.1.2) The absoluteURI form is only allowed when the request is being made to a proxy.  */
  /* absoluteURI = scheme ":" *( uchar | reserved ) */
  if (strncasecmp("http://", uri, SCHEMELEN) == 0) {
    char *start = uri + SCHEMELEN;
    char *end = strchr(start, '/');

    size_t n;
    if (end == NULL) {
      n = strlen(start);
      strcpy(filename, "/");
    } else {
      n = end - start;
      strcpy(filename, end);
    }

    memcpy(hostname, start, n);
    hostname[n] = '\0';

    char *port_start;
    if ((port_start = strchr(hostname, ':')) != NULL) {
      n = port_start - hostname;
      hostname[n] = '\0';
      *port = port_start + 1;
    }

#ifdef DEBUG
    printf("host: %s, port: %s, len: %ld\n", hostname, *port, n);
#endif
  } else {
    clienterror(clientfd, uri, "400", "Bad Request", "Proxy only process absoluteURI");
    return;
  }
}

static int open_serverfd(int clientfd, char *hostname, char *port) {
  int serverfd;
  if ((serverfd = open_clientfd(hostname, port)) < 0) {
    print_error("open_clientfd", __LINE__, strerror(errno));
    clienterror(clientfd, hostname, "404", "Not Found", "Proxy cannot connect with server");
    Pthread_exit(NULL);
  }
#ifdef DEBUG
  printf("[%d] Connect to host: %s\n", serverfd, hostname);
#endif
  return serverfd;
}

static void process_headers(int serverfd, rio_t *rp, char *buf, char *hostname) {
  bool hosthdr = false;
  Rrio_readlineb(rp, buf, MAXLINE);
  while (strcmp(buf, "\r\n")) {
    check_send(serverfd, buf, &hosthdr);
    Rrio_readlineb(rp, buf, MAXLINE);
  }
  if (!hosthdr) {
    try_writen(serverfd, buf, 0, "Host: %s", hostname);
  }
  try_writen(serverfd, buf, 0, user_agent_hdr);
  try_writen(serverfd, buf, 0, conn_hdr);
  try_writen(serverfd, buf, 0, porxy_conn_hdr);
  try_writen(serverfd, buf, 0, "\r\n");
}

static void check_send(const int serverfd, char *hdr, bool *host) {
  static const char *fhdrs[] = {
      "Host",
      "User-Agent",
      "Connection",
      "Proxy-Connection",
  };
  int n = sizeof(fhdrs) / sizeof(fhdrs[0]);
  for (int i = 0; i < n; ++i) {
    int len = strlen(fhdrs[i]);
    if (strncmp(hdr, fhdrs[i], len) == 0) {
      if (i == 0) {
        /* break if match Host */
        *host = true;
        break;
      }
      return;
    }
  }

  char buf[MAXLINE];
  try_writen(serverfd, buf, 0, hdr); /* forward headers unchanged. */
}

static void handle_proxy(int clientfd) {
  Pthread_detach(Pthread_self());
  pthread_cleanup_push(cleanup, (void *)&clientfd);

  rio_t client_rio;
  rio_readinitb(&client_rio, clientfd);

  /* Read and parse the Request-Line.  */
  char buf[MAXLINE];
  char hostname[MAXLINE], filename[MAXLINE], *port = http_port;
  Rrio_readlineb(&client_rio, buf, MAXLINE);
  parse_requestline(clientfd, buf, hostname, &port, filename);

  static pthread_once_t once = PTHREAD_ONCE_INIT;
  Pthread_once(&once, cache_init);

  /* fast path if cache hit */
  cache_element_t *element;
  if ((element = cache_get(hostname, filename)) != NULL) {
    pthread_cleanup_push(cache_element_free, element);
    do {
      Rio_readlineb(&client_rio, buf, MAXLINE); /* ignore all headers */
    } while (strcmp(buf, "\r\n"));

    try_writen(clientfd, element->buf, element->len, NULL);
    pthread_cleanup_pop(true);
    return;
  }

  /* Connect to server */
  int serverfd = open_serverfd(clientfd, hostname, port);
  pthread_cleanup_push(cleanup, (void *)&serverfd);

  /* forward request headers to server */
  try_writen(serverfd, buf, 0, "GET %s HTTP/1.0\r\n", filename);
  process_headers(serverfd, &client_rio, buf, hostname);

  rio_t server_rio;
  rio_readinitb(&server_rio, serverfd);

  /* alloc on heap, so it is <Send, Sync>(in Rust) */
  element = cache_element_new();
  pthread_cleanup_push(cache_element_free, element);

  /* forward response to client and caching */
  ssize_t len;
  while ((len = rio_readnb(&server_rio, buf, MAXLINE)) > 0) {
    cache_element_update(element, buf, len);
    try_writen(clientfd, buf, len, NULL);
  }
  if (len < 0) {
    print_error("rio_readnb", __LINE__, strerror(errno));
    return;
  }
  cache_put(hostname, filename, element);

  pthread_cleanup_pop(false);
  pthread_cleanup_pop(true);
  pthread_cleanup_pop(true);
}
