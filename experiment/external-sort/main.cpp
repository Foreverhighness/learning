#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <random>

using elem_t = uint32_t;

constexpr int BATCH_SIZE = 32;
constexpr int NUM_FILES = 1000;
constexpr int NUMBERS = 1024;
constexpr size_t BLOCK_SIZE = NUMBERS * sizeof(elem_t);

const std::filesystem::path BUILD_DIR = "build";

template <typename It> void kmerge(It first, It last, std::ostream &os) {
  using PI = typename It::value_type;
  static_assert(
      std::is_same_v<typename PI::first_type, typename PI::second_type>);
  static_assert(
      std::is_same_v<std::remove_reference_t<decltype(*(first->first))>,
                     elem_t>);

  auto comp = [](const PI &lhs, const PI &rhs) {
    return *lhs.first > *rhs.first;
  };
  std::priority_queue<PI, std::vector<PI>, decltype(comp)> pq(comp);

  for (auto it = first; it != last; std::advance(it, 1)) {
    auto &[begin, end] = *it;
    assert(std::is_sorted(begin, end));
    if (begin != end) {
      pq.emplace(*it);
    }
  }

  while (!pq.empty()) {
    const auto [begin, end] = pq.top();
    pq.pop();
    const elem_t val = *begin;
    const auto next = std::next(begin);
    if (next != end) {
      pq.emplace(PI{next, end});
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

  using VIt = std::vector<elem_t>::iterator;
  using PI = std::pair<VIt, VIt>;
  std::vector<PI> vec;
  for (auto &file : files) {
    std::sort(file.begin(), file.end());
    vec.emplace_back(PI{file.begin(), file.end()});
  }

  kmerge(vec.begin(), vec.end(), os);
}

static auto get_run_file_path(const int round, const int run_number)
    -> std::filesystem::path {
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
  Sorter(uint8_t *memory, const size_t size)
      : memory_(reinterpret_cast<elem_t *>(memory)),
        cap_(size / sizeof(elem_t)) {
    assert(size % sizeof(elem_t) == 0);
  }
  Sorter() = delete;

  auto finish() -> std::vector<std::filesystem::path> {
    do_sort();

    std::vector<std::filesystem::path> ret;
    ret.swap(result_);
    return ret;
  }

  void take(const std::filesystem::path &path) {
    std::ifstream ifs{path, std::ios::in | std::ios::binary};
    std::streamsize size;
    do {
      const auto n = std::min(BLOCK_SIZE, (cap_ - len_) * sizeof(elem_t));
      ifs.read(reinterpret_cast<char *>(memory_ + len_), n);
      size = ifs.gcount();
      assert(size % sizeof(elem_t) == 0);

      len_ += size / sizeof(elem_t);

      const bool is_full = len_ == cap_;
      if (is_full) {
        do_sort();
      }
    } while (size != 0);
  }

private:
  elem_t *memory_;
  const size_t cap_;

  size_t len_{0};
  int run_number_{0};
  std::vector<std::filesystem::path> result_{};

  static constexpr int round = 1;

  auto dump_memory_into_run_file() const -> std::filesystem::path {
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

int main() {
  const size_t size = BATCH_SIZE * NUMBERS * sizeof(elem_t);
  auto memory = std::make_unique<uint8_t[]>(size);
  Sorter sorter{memory.get(), size};

  for (int fileno = 0; fileno != NUM_FILES; ++fileno) {
    const auto input_path = BUILD_DIR / (std::to_string(fileno) + ".bin");
    sorter.take(input_path);
  }
  const auto run_files = sorter.finish();

  // merge phase

  return 0;
}