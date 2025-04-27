#include <cuda_pipeline.h>
#include <math.h>

#include <iostream>

constexpr size_t size = 32 * 1024;
__shared__ char s[size];  // 32KiB

#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t result = (stmt);                                               \
    if (cudaSuccess != result) {                                               \
      std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] CUDA failed with " \
                << cudaGetErrorString(result) << std::endl;                    \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

template <typename T>
__global__ void pipeline_kernel_sync(T *global, uint64_t *clock,
                                     size_t copy_count) {
  T *shared = reinterpret_cast<T *>(s);

  uint64_t clock_start = clock64();

  for (size_t i = 0; i < copy_count; ++i) {
    shared[blockDim.x * i + threadIdx.x] = global[blockDim.x * i + threadIdx.x];
  }

  uint64_t clock_end = clock64();

  atomicAdd(reinterpret_cast<unsigned long long *>(clock),
            clock_end - clock_start);
}

template <typename T>
__global__ void pipeline_kernel_async(T *global, uint64_t *clock,
                                      size_t copy_count) {
  T *shared = reinterpret_cast<T *>(s);

  uint64_t clock_start = clock64();

  // pipeline pipe;
  for (size_t i = 0; i < copy_count; ++i) {
    __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                            &global[blockDim.x * i + threadIdx.x], sizeof(T));
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);

  uint64_t clock_end = clock64();

  atomicAdd(reinterpret_cast<unsigned long long *>(clock),
            clock_end - clock_start);
}

template <typename T>
void run_test(size_t copy_count = 16) {
  T *global;
  uint64_t *clock;

  int block_size = 32;

  std::cout << "Type size: " << sizeof(T) << " bytes" << std::endl;
  std::cout << "Block size: " << block_size << ", Copy count: " << copy_count
            << std::endl;

  CUDA_CHECK(cudaMallocAsync(&global, size, 0));
  CUDA_CHECK(cudaMallocAsync(&clock, sizeof(uint64_t), 0));

  CUDA_CHECK(cudaMemsetAsync(global, 0x3f, size));
  CUDA_CHECK(cudaMemsetAsync(clock, 0, sizeof(uint64_t)));

  cudaDeviceSynchronize();

  pipeline_kernel_sync<T><<<1, block_size>>>(global, clock, copy_count);
  CUDA_CHECK(cudaGetLastError());

  uint64_t sync_cycles;
  CUDA_CHECK(cudaMemcpyAsync(&sync_cycles, clock, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemsetAsync(clock, 0, sizeof(uint64_t)));
  cudaDeviceSynchronize();

  pipeline_kernel_async<T><<<1, block_size>>>(global, clock, copy_count);
  CUDA_CHECK(cudaGetLastError());
  uint64_t async_cycles;
  CUDA_CHECK(cudaMemcpyAsync(&async_cycles, clock, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemsetAsync(clock, 0, sizeof(uint64_t)));
  cudaDeviceSynchronize();

  std::cout << "Sync version: " << sync_cycles << " cycles" << std::endl;
  std::cout << "Async version: " << async_cycles << " cycles" << std::endl;
  std::cout << "Speedup: " << (1.0f * sync_cycles / async_cycles) << "x"
            << std::endl;

  CUDA_CHECK(cudaFreeAsync(global, 0));
  CUDA_CHECK(cudaFreeAsync(clock, 0));
}

int main(void) {
  // run_test<uint8_t>();
  // run_test<uint16_t>();
  run_test<uint32_t>();
  run_test<uint64_t>();
  return 0;
}
