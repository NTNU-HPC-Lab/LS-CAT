#include "includes.h"
__global__ void sgemm_kernel_v2(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta)
{
int bid_x = blockIdx.x * blockDim.x;
int bid_y = blockIdx.y * blockDim.y;
int tid_x = threadIdx.x;
int tid_y = threadIdx.y;

float element_c = 0.f;
__shared__ float s_tile_A[BLOCK_DIM][BLOCK_DIM];
__shared__ float s_tile_B[BLOCK_DIM][BLOCK_DIM];

// forward tile with tile size in matrix A
for (int k = 0; k < K; k += BLOCK_DIM)
{
s_tile_A[tid_y][tid_x] = A[ (bid_y + tid_y) * K + tid_x + k ]; // Get sub-matrix from A
s_tile_B[tid_y][tid_x] = B[ (k*BLOCK_DIM + tid_y) * N + bid_x + tid_x ]; // Get sub-matrix from B

__syncthreads();

// compute gemm operation with tiles
for (int e = 0; e < BLOCK_DIM; e++)
element_c += s_tile_A[tid_y][e] * s_tile_B[e][tid_x];

__syncthreads();
}

C[(bid_y + tid_y) * N + (bid_x + tid_x)] = \
alpha * element_c + beta * C[(bid_y + tid_y) * N + (bid_x + tid_x)];
}