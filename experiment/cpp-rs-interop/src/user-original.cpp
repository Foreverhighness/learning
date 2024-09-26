#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

class User {
  std::string name_;
  uint64_t comments_count_;
  uint8_t uuid_[16];

public:
  User(std::string name_) : name_{name_}, comments_count_{0} {
    arc4random_buf(uuid_, sizeof(uuid_));

    static_assert(std::is_standard_layout_v<User>);
  }

  void write_comment(const char *comment, size_t comment_len) {
    printf("%s (", name_.c_str());
    for (size_t i = 0; i < sizeof(uuid_); i += 1) {
      printf("%x", uuid_[i]);
    }
    printf(") says: %.*s\n", (int)comment_len, comment);
    comments_count_ += 1;
  }

  uint64_t get_comment_count() { return comments_count_; }
};

int main() {
  User alice{"alice"};
  const char msg[] = "hello, original!";
  alice.write_comment(msg, sizeof(msg) - 1);

  printf("Comment count: %lu\n", alice.get_comment_count());

  // This prints:
  // alice (fe61252cf5b88432a7e8c8674d58d615) says: hello, original!
  // Comment count: 1
}
