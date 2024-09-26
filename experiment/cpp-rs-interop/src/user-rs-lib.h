#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

struct UserC {
  uint8_t name[32];
  uint64_t comments_count;
  uint8_t uuid[16];
};

extern "C" {

void Rust_write_comment(UserC *user, const uint8_t *comment, uintptr_t comment_len);

}  // extern "C"
