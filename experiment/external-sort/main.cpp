#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <random>

using elem_t = uint32_t;

constexpr char const *BUILD_DIR = "build";
constexpr int BATCH_SIZE = 32;
constexpr int NUM_FILES = 1000;
constexpr int NUMBERS = 1024;
constexpr int BLOCK_SIZE = NUMBERS * sizeof(elem_t);

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

static auto get_run_file_name(const int round, const int run_number)
    -> std::string {
  return std::string{BUILD_DIR} + '/' + std::to_string(round) + '-' +
         std::to_string(run_number) + ".run";
}

[[maybe_unused]] static void output(elem_t *addr, int len) {
  for (int i = 0; i != len; ++i) {
    std::cout << addr[i] << " \n"[i + 1 == len];
  }
}

static void read_file_into_buffer(const int fileno, elem_t *buf) {
  const std::string filename{std::string{BUILD_DIR} + '/' +
                             std::to_string(fileno) + ".bin"};

  std::ifstream ifs{filename, std::ios::in | std::ios::binary};
  ifs.read(reinterpret_cast<char *>(buf), BLOCK_SIZE);
}

static void write_buffer_into_run(const elem_t *buf, const int len,
                                  const int round, const int run_number) {
  const auto filename = get_run_file_name(round, run_number);
  std::ofstream ofs{filename, std::ios::out | std::ios::binary};
  ofs.write(reinterpret_cast<const char *>(buf), len * sizeof(elem_t));
}

int main() {
  auto memory = std::make_unique<elem_t[]>(BATCH_SIZE * NUMBERS);
  int round = 1;
  int num_files = NUM_FILES;
  int runs = (num_files + BATCH_SIZE - 1) / BATCH_SIZE;

  // sort phase
  int fileno = 0;
  for (int run_number = 0; run_number != runs; ++run_number) {

    auto cur = memory.get();
    for (int i = 0; i != BATCH_SIZE && fileno != num_files; ++i, ++fileno) {
      read_file_into_buffer(fileno, cur);
      cur += NUMBERS;
    }

    const auto begin = memory.get(), end = cur;
    std::sort(begin, end);
    write_buffer_into_run(begin, end - begin, round, run_number);
  }

  return 0;
}