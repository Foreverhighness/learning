#include "user-rs-lib.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>

class User {
  std::string name_;
  uint64_t comments_count_;
  uint8_t uuid_[16];

public:
  User(std::string name) : name_{name}, comments_count_{0} {
    arc4random_buf(uuid_, sizeof(uuid_));

    static_assert(std::is_standard_layout_v<User>);
    static_assert(sizeof(std::string) == 32);
    static_assert(sizeof(User) == sizeof(UserC));
    static_assert(offsetof(User, name_) == offsetof(UserC, name));
    static_assert(offsetof(User, comments_count_) ==
                  offsetof(UserC, comments_count));
    static_assert(offsetof(User, uuid_) == offsetof(UserC, uuid));
  }

  void write_comment(const char *comment, size_t comment_len) {
    printf("%s (", name_.c_str());
    for (size_t i = 0; i < sizeof(uuid_); i += 1) {
      printf("%x", uuid_[i]);
    }
    printf(") says: %.*s\n", (int)comment_len, comment);
    comments_count_ += 1;
  }

  auto get_comment_count() -> uint64_t { return comments_count_; }

  auto get_name() -> std::string & { return name_; }
};

auto get_std_string_pointer_and_length(const std::string &str)
    -> ByteSliceView {
  return {
      .ptr = reinterpret_cast<const uint8_t *>(str.data()),
      .len = str.size(),
  };
}

int main() {
  User alice{"alice"};
  {
    const char msg[] = "hello, cpp!";
    alice.write_comment(msg, sizeof(msg) - 1);
    printf("Comment count: %lu\n", alice.get_comment_count());
    // This prints:
    // alice (486af2db45c3f5cc92ff641c85fe4a1d) says: hello, cpp!
    // Comment count: 1
  }

  {
    const char msg[] = "hello, rust!";
    Rust_write_comment(reinterpret_cast<UserC *>(&alice),
                       reinterpret_cast<const uint8_t *>(msg), sizeof(msg) - 1);
    printf("Comment count: %lu\n", alice.get_comment_count());
    // This prints:
    // [48, 6a, f2, db, 45, c3, f5, cc, 92, ff, 64, 1c, 85, fe, 4a, 1d] says: hello, rust!
    // Comment count: 2
  }

  {
    const char msg[] = "hello, ByteSliceView!";
    Rust_write_comment_with_ByteSliceView(
        reinterpret_cast<UserC *>(&alice),
        reinterpret_cast<const uint8_t *>(msg), sizeof(msg) - 1,
        get_std_string_pointer_and_length(alice.get_name()));
    printf("Comment count: %lu\n", alice.get_comment_count());
    // This prints:
    // alice [48, 6a, f2, db, 45, c3, f5, cc, 92, ff, 64, 1c, 85, fe, 4a, 1d] says: hello, ByteSliceView!
    // Comment count: 3
  }
}
