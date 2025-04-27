#include <math.h>

#include <iostream>

constexpr int TILE_DIM = 32;
constexpr float v_a = 2.0f;
constexpr float v_b = 3.0f;

__global__ void simpleMultiply(float *a, float *b, float *c, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int i = 0; i < TILE_DIM; i++) {
    sum += a[row * TILE_DIM + i] * b[i * N + col];
  }
  c[row * N + col] = sum;
}

__global__ void coalescedMultiply(float *a, float *b, float *c, int N) {
  __shared__ float aTile[TILE_DIM][TILE_DIM];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];

  __syncwarp();

  for (int i = 0; i < TILE_DIM; i++) {
    sum += aTile[threadIdx.y][i] * b[i * N + col];
  }
  c[row * N + col] = sum;
}

__global__ void sharedABMultiply(float *a, float *b, float *c, int N) {
  __shared__ float aTile[TILE_DIM][TILE_DIM], bTile[TILE_DIM][TILE_DIM];
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
  bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y * N + col];

  __syncthreads();

  for (int i = 0; i < TILE_DIM; i++) {
    sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
  }
  c[row * N + col] = sum;
}

void init(float *a, float *b, float *c, int N, int M) {
  for (int row = 0; row < M; ++row) {
    for (int i = 0; i < TILE_DIM; ++i) {
      a[row * TILE_DIM + i] = v_a;
    }
  }

  for (int i = 0; i < TILE_DIM; ++i) {
    for (int col = 0; col < N; ++col) {
      b[i * N + col] = v_b;
    }
  }

  cudaMemset(c, 0, N * M);
}

void check(float *c, int N, int M) {
  float maxError = 0.0f;
  for (int i = 0; i < N * M; i++) {
    maxError = fmax(maxError, fabs(c[i] - v_a * v_b * TILE_DIM));
  }
  std::cout << "Max error: " << maxError << std::endl;
}

int main(void) {
  int M = 2 << 10;
  int N = 3 << 10;
  float *a, *b, *c;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&a, M * TILE_DIM * sizeof(float));
  cudaMallocManaged(&b, N * TILE_DIM * sizeof(float));
  cudaMallocManaged(&c, N * M * sizeof(float));

  dim3 block_size(TILE_DIM, TILE_DIM);
  dim3 num_blocks(N / TILE_DIM, M / TILE_DIM);

  init(a, b, c, N, M);
  simpleMultiply<<<num_blocks, block_size>>>(a, b, c, N);
  cudaDeviceSynchronize();  // Wait for GPU to finish before accessing on host
  check(c, N, M);

  init(a, b, c, N, M);
  coalescedMultiply<<<num_blocks, block_size>>>(a, b, c, N);
  cudaDeviceSynchronize();
  check(c, N, M);

  init(a, b, c, N, M);
  sharedABMultiply<<<num_blocks, block_size>>>(a, b, c, N);
  cudaDeviceSynchronize();
  check(c, N, M);

  // Free memory
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  return 0;
}
