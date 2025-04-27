#include <math.h>

#include <iostream>

constexpr int TILE_DIM = 32;
constexpr float v_a = 2.0f;

__global__ void simpleMultiply(float *a, float *c, int M) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int i = 0; i < TILE_DIM; i++) {
    sum += a[row * TILE_DIM + i] * a[col * TILE_DIM + i];
  }
  c[row * M + col] = sum;
}

__global__ void coalescedMultiply(float *a, float *c, int M) {
  __shared__ float aTile[TILE_DIM][TILE_DIM],
      transposedTile[TILE_DIM][TILE_DIM];
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
  transposedTile[threadIdx.x][threadIdx.y] =
      a[(blockIdx.x * blockDim.x + threadIdx.y) * TILE_DIM + threadIdx.x];
  __syncthreads();
  for (int i = 0; i < TILE_DIM; i++) {
    sum += aTile[threadIdx.y][i] * transposedTile[i][threadIdx.x];
  }
  c[row * M + col] = sum;
}

__global__ void bankConflictFreeMultiply(float *a, float *c, int M) {
  __shared__ float aTile[TILE_DIM][TILE_DIM],
      transposedTile[TILE_DIM][TILE_DIM + 1];
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
  transposedTile[threadIdx.x][threadIdx.y] =
      a[(blockIdx.x * blockDim.x + threadIdx.y) * TILE_DIM + threadIdx.x];
  __syncthreads();
  for (int i = 0; i < TILE_DIM; i++) {
    sum += aTile[threadIdx.y][i] * transposedTile[i][threadIdx.x];
  }
  c[row * M + col] = sum;
}

void init(float *a, float *c, int M) {
  for (int row = 0; row < M; ++row) {
    for (int i = 0; i < TILE_DIM; ++i) {
      a[row * TILE_DIM + i] = v_a;
    }
  }

  cudaMemset(c, 0, M * M);
}

void check(float *c, int M) {
  float maxError = 0.0f;
  for (int i = 0; i < M * M; i++) {
    maxError = fmax(maxError, fabs(c[i] - v_a * v_a * TILE_DIM));
  }
  std::cout << "Max error: " << maxError << std::endl;
}

int main(void) {
  int M = 2 << 10;
  float *a, *c;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&a, M * TILE_DIM * sizeof(float));
  cudaMallocManaged(&c, M * M * sizeof(float));

  int block = TILE_DIM;
  dim3 block_size(block, block);
  dim3 num_blocks(M / block, M / block);

  init(a, c, M);
  simpleMultiply<<<num_blocks, block_size>>>(a, c, M);
  cudaDeviceSynchronize();  // Wait for GPU to finish before accessing on host
  check(c, M);

  init(a, c, M);
  coalescedMultiply<<<num_blocks, block_size>>>(a, c, M);
  cudaDeviceSynchronize();
  check(c, M);

  init(a, c, M);
  bankConflictFreeMultiply<<<num_blocks, block_size>>>(a, c, M);
  cudaDeviceSynchronize();
  check(c, M);

  // Free memory
  cudaFree(a);
  cudaFree(c);

  return 0;
}
