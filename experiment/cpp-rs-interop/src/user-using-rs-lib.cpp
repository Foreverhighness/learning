#include "user-rs-lib.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>

class User {
  std::string name;
  uint64_t comments_count;
  uint8_t uuid[16];

public:
  User(std::string name_) : name{name_}, comments_count{0} {
    arc4random_buf(uuid, sizeof(uuid));

    static_assert(std::is_standard_layout_v<User>);
    static_assert(sizeof(std::string) == 32);
    static_assert(sizeof(User) == sizeof(UserC));
    static_assert(offsetof(User, name) == offsetof(UserC, name));
    static_assert(offsetof(User, comments_count) ==
                  offsetof(UserC, comments_count));
    static_assert(offsetof(User, uuid) == offsetof(UserC, uuid));
  }

  void write_comment(const char *comment, size_t comment_len) {
    printf("%s (", name.c_str());
    for (size_t i = 0; i < sizeof(uuid); i += 1) {
      printf("%x", uuid[i]);
    }
    printf(") says: %.*s\n", (int)comment_len, comment);
    comments_count += 1;
  }

  uint64_t get_comment_count() { return comments_count; }
};

int main() {
  User alice{"alice"};
  const char msg[] = "hello, world!";
  alice.write_comment(msg, sizeof(msg) - 1);

  printf("Comment count: %lu\n", alice.get_comment_count());

  // This prints:
  // alice (fe61252cf5b88432a7e8c8674d58d615) says: hello, world!
  // Comment count: 1
  Rust_write_comment(reinterpret_cast<UserC *>(&alice),
                     reinterpret_cast<const uint8_t *>(msg), sizeof(msg) - 1);
  printf("Comment count: %lu\n", alice.get_comment_count());
}
