#include <math.h>

#include <iostream>

// Kernel function to add the elements of two arrays
__global__ void add0(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}
__global__ void add1(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}
__global__ void add2(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}

void init(int N, float *x, float *y) {
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
}

int main(void) {
  int N = 1 << 20;  // 1024 * 1024 = 1M
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  int block_size = 256;
  int num_blocks = (N + block_size - 1) / block_size;

  init(N, x, y);
  add0<<<1, 1>>>(N, x, y);  // Run kernel on 1M elements on the GPU
  cudaDeviceSynchronize();  // Wait for GPU to finish before accessing on host
  init(N, x, y);
  add1<<<1, block_size>>>(N, x, y);
  cudaDeviceSynchronize();
  init(N, x, y);
  add2<<<num_blocks, block_size>>>(N, x, y);
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
