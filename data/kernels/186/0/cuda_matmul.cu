#include "includes.h"
__global__ void cuda_matmul(float* A, float* B, float* C, size_t n)
{
float sum = 0.0f;

#ifndef MATMUL_USE_SHARED
int ia = (blockDim.y * blockIdx.y + threadIdx.y) * n;
int ib = blockDim.x * blockIdx.x + threadIdx.x;
int ic = ia + ib;

// Multiply two matrices
for (int k = 0; k < n; k++)
sum += A [ia + k] * B [ib + k * n];
#else
// Base indexes inside A and B
int ia = (blockDim.y * blockIdx.y) * n;
int ib = blockDim.x * blockIdx.x;

// Subindex inside a "tile"
int tileidx = n * threadIdx.y + threadIdx.x;

// Index in C
int ic = ia + ib + tileidx;

int aoff = 0, boff = 0;

// Shared memory for the "tile" sub-matrix of A and B
__shared__ float As [BLOCK_SIZE][BLOCK_SIZE];
__shared__ float Bs [BLOCK_SIZE][BLOCK_SIZE];

// Go through "tiles" of size blockDim.x * blockDim.y
for (; aoff < n; aoff += blockDim.x, boff += blockDim.y * n)
{
// Load the "tile" matrices from global memory to shared memory
As [threadIdx.y][threadIdx.x] = A [ia + aoff + tileidx];
Bs [threadIdx.y][threadIdx.x] = B [ib + boff + tileidx];

// Synchronize to make sure the matrices are loaded
__syncthreads();

// Multiply the two matrices
for (int k = 0; k < BLOCK_SIZE; k++)
sum += As [threadIdx.y][k] * Bs [k][threadIdx.x];

// Synchronize to make sure that the preceding
// computation is done before loading two new
// sub-matrices of A and B in the next iteration
__syncthreads();
}
#endif
// Write the block sub-matrix to global memory
// each thread writes one element
C [ic] = sum;
}