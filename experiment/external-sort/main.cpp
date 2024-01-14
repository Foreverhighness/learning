#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <random>

namespace fs = std::filesystem;

using elem_t = uint32_t;

constexpr int BATCH_SIZE = 32;
constexpr int NUM_FILES = 1000;
constexpr int NUMBERS = 1024;
constexpr size_t BLOCK_SIZE = NUMBERS * sizeof(elem_t);

const fs::path BUILD_DIR = "build";

// template <typename T>
class BlockReaderIterator {
public:
  using T = elem_t;

  BlockReaderIterator(char *buffer, size_t size, std::istream &is)
      : buffer_(buffer), size_(size), is_(is) {}
  auto next() -> std::optional<T> {
    if (is_over_ && cur_ == len_) {
      return std::nullopt;
    }
    if (cur_ == len_) {
      update_buffer();
    }
    if (is_over_ && cur_ == len_) {
      return std::nullopt;
    }
    assert(cur_ < len_);
    return reinterpret_cast<T *const>(buffer_)[cur_++];
  }

private:
  char *const buffer_;
  const size_t size_;

  std::istream &is_;

  size_t len_{0};
  size_t cur_{0};

  bool is_over_{false};

  auto update_buffer() -> void {
    assert(cur_ == len_);
    assert(!is_over_);

    const bool success = (bool)is_.read(buffer_, size_);
    const auto count = is_.gcount();

    assert(count % sizeof(T) == 0);
    // std::cerr << success << '\n';
    // std::cerr << "len: " << len_ << " -> " << (count / sizeof(T)) << '\n';
    len_ = count / sizeof(T);
    cur_ = 0;

    if (!success || count == 0) {
      is_over_ = true;
    }
  }
};

template <typename It> void kmerge(It first, It last, std::ostream &os) {
  static_assert(std::is_same_v<typename It::value_type, BlockReaderIterator>);
  using BRIT = typename It::value_type;
  using PI = std::pair<typename BRIT::T, It>;

  auto comp = [](const PI &lhs, const PI &rhs) {
    return lhs.first > rhs.first;
  };
  std::priority_queue<PI, std::vector<PI>, decltype(comp)> pq(comp);

  for (auto it = first; it != last; std::advance(it, 1)) {
    if (const auto value = it->next()) {
      pq.emplace(PI{value.value(), it});
    }
  }

  while (!pq.empty()) {
    const auto [val, it] = pq.top();
    pq.pop();
    if (const auto value = it->next()) {
      pq.emplace(PI{value.value(), it});
    }
    os.write(reinterpret_cast<const char *>(&val), sizeof(val));
  }
}

[[maybe_unused]] static void test(std::ostream &os) {
  constexpr int num_data = 9;
  std::vector<elem_t> data(num_data);
  std::iota(data.begin(), data.end(), 1);
  std::random_shuffle(data.begin(), data.end());

  std::mt19937 g{std::random_device{}()};
  constexpr int num_file = 3;
  std::vector<std::vector<elem_t>> files(num_file);
  for (const elem_t val : data) {
    files.at(g() % num_file).emplace_back(val);
  }

  for (int i = 0; i != num_file; ++i) {
    std::cout << i << ": [";
    bool first = true;
    for (const elem_t val : files.at(i)) {
      if (first) {
        first = false;
      } else {
        std::cout << ", ";
      }
      std::cout << val;
    }
    std::cout << "]\n";
  }

  std::vector<BlockReaderIterator> vec;
  std::vector<std::stringstream> inputs(files.size());
  std::vector<std::vector<char>> buffers(files.size());
  constexpr size_t size = 256;

  for (size_t i = 0; i != files.size(); ++i) {
    auto &file = files[i];
    auto &input = inputs[i];
    auto &buffer = buffers[i];

    std::sort(file.begin(), file.end());
    std::for_each(file.cbegin(), file.cend(), [&](const elem_t val) {
      input.write(reinterpret_cast<const char *>(&val), sizeof(val));
    });
    buffer.resize(size);

    vec.emplace_back(buffer.data(), buffer.size(), input);
  }

  kmerge(vec.begin(), vec.end(), os);
}

static auto get_run_file_path(const int round, const int run_number)
    -> fs::path {
  const auto run_filename{std::to_string(round) + '-' +
                          std::to_string(run_number) + ".run"};
  return BUILD_DIR / run_filename;
}

[[maybe_unused]] static void output(elem_t *addr, int len) {
  for (int i = 0; i != len; ++i) {
    std::cout << addr[i] << " \n"[i + 1 == len];
  }
}

class Sorter {
public:
  Sorter(char *memory, const size_t size)
      : memory_(reinterpret_cast<elem_t *>(memory)),
        cap_(size / sizeof(elem_t)) {
    assert(size % sizeof(elem_t) == 0);
  }
  Sorter() = delete;

  auto finish() -> std::vector<fs::path> {
    do_sort();

    std::vector<fs::path> ret;
    ret.swap(result_);
    return ret;
  }

  void take(const fs::path &path) {
    std::ifstream ifs{path, std::ios::in | std::ios::binary};
    for (;;) {
      const auto n = std::min(BLOCK_SIZE, (cap_ - len_) * sizeof(elem_t));
      const bool success =
          (bool)ifs.read(reinterpret_cast<char *>(memory_ + len_), n);

      const auto size = ifs.gcount();
      if (!success || size == 0) {
        break;
      }
      assert(size % sizeof(elem_t) == 0);

      len_ += size / sizeof(elem_t);

      const bool is_full = len_ == cap_;
      if (is_full) {
        do_sort();
      }
    }
  }

private:
  elem_t *memory_;
  const size_t cap_;

  size_t len_{0};
  int run_number_{0};
  std::vector<fs::path> result_{};

  static constexpr int round = 1;

  auto dump_memory_into_run_file() const -> fs::path {
    const auto filepath = get_run_file_path(round, run_number_);
    std::ofstream ofs{filepath, std::ios::out | std::ios::binary};
    ofs.write(reinterpret_cast<const char *>(memory_), len_ * sizeof(elem_t));
    return filepath;
  }

  void do_sort() {
    std::sort(memory_, memory_ + len_);
    auto run_filepath = dump_memory_into_run_file();
    result_.emplace_back(std::move(run_filepath));
    len_ = 0;
    run_number_ += 1;
  }
};

class Merger {
public:
  Merger(char *memory, const size_t size)
      : memory_(memory), size_(size), B_minus_one_(size / BLOCK_SIZE - 1) {
    assert(size_ % BLOCK_SIZE == 0);
    assert(B_minus_one_ >= 2);

    inputs_.reserve(B_minus_one_);
    output_.rdbuf()->pubsetbuf(memory_ + B_minus_one_ * BLOCK_SIZE, BLOCK_SIZE);
  }

  Merger() = delete;

  auto finish() -> std::vector<fs::path> {
    do_merge();

    round_ += 1;
    run_number_ = 0;

    std::vector<fs::path> ret;
    ret.swap(result_);
    return ret;
  }

  auto take(const fs::path input_path) -> void {
    inputs_.emplace_back(std::move(input_path));
    if (inputs_.size() == B_minus_one_) {
      do_merge();
    }
  }

private:
  char *const memory_;
  const size_t size_;
  const size_t B_minus_one_;

  std::vector<fs::path> result_{};

  std::vector<fs::path> inputs_;
  std::ofstream output_;

  int round_{2};
  int run_number_{0};

  auto cleanup() -> void {
    for_each(inputs_.cbegin(), inputs_.cend(),
             [](const auto &path) { assert(fs::remove(path)); });
    inputs_.clear();
    output_.close();
  }

  auto do_merge() -> void {
    const auto run_filepath = get_run_file_path(round_, run_number_++);
    output_.open(run_filepath, std::ios::out | std::ios::binary);

    const size_t size = inputs_.size();

    std::vector<std::ifstream> inputs;
    inputs.reserve(size);
    for (const auto &path : inputs_) {
      inputs.emplace_back(path, std::ios::in | std::ios::binary);
    }

    std::vector<BlockReaderIterator> iters;
    iters.reserve(size);
    for (size_t i = 0; i != size; ++i) {
      iters.emplace_back(memory_ + i * BLOCK_SIZE, BLOCK_SIZE, inputs[i]);
    }

    kmerge(iters.begin(), iters.end(), output_);
    cleanup();
    result_.emplace_back(std::move(run_filepath));
  }
};

int main() {
  const size_t size = BATCH_SIZE * NUMBERS * sizeof(elem_t);
  auto memory = std::make_unique<char[]>(size);
  Sorter sorter{memory.get(), size};

  std::vector<fs::path> run_files;
  for (int fileno = 0; fileno != NUM_FILES; ++fileno) {
    const auto input_path = BUILD_DIR / (std::to_string(fileno) + ".bin");
    sorter.take(input_path);
  }
  run_files = sorter.finish();

  // merge phase
  Merger merger{memory.get(), size};

  while (run_files.size() != 1) {
    for (const auto &path : run_files) {
      merger.take(path);
    }
    run_files = merger.finish();
  }
  fs::rename(run_files[0], BUILD_DIR / "result.out");

  return 0;
}